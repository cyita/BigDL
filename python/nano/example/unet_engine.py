import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import copy
from time import time
from openvino.runtime import PartialShape, Shape    # pylint: disable=E0611,E0401
from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, compress_model_weights, create_pipeline
from openvino.tools.pot.graph.model_utils import add_outputs
from openvino.tools.pot.samplers.batch_sampler import BatchSampler
from openvino.tools.pot.engines.utils import process_accumulated_stats, \
    restore_original_node_names, align_stat_names_with_results, \
    add_tensor_names, collect_model_outputs, get_clean_name
from openvino.tools.pot.utils.logger import init_logger, get_logger

init_logger(level='INFO')
logger = get_logger(__name__)


class UNETEngine(IEEngine):
    def __init__(self, config, data_loader=None, metric=None):
        super().__init__(config, data_loader, metric)

    def _predict(self, stats_layout, sampler, print_progress=False,
                 need_metrics_per_sample=False):
        progress_log_fn = logger.info if print_progress else logger.debug
        progress_log_fn('Start inference of %d images', len(sampler))

        compiled_model = self._ie.compile_model(model=self._model,
                                                device_name=self.config.device)
        infer_request = compiled_model.create_infer_request()

        # Start inference
        start_time = time()
        for batch_id, batch in iter(enumerate(sampler)):
            batch_annotations, image_batch, batch_meta = self._process_batch(batch)
            filled_input = self._fill_input(compiled_model, image_batch)
            result = infer_request.infer(filled_input)

            self._process_infer_output(stats_layout, result, batch_annotations, batch_meta,
                                       need_metrics_per_sample)

            # Print progress
            if self._print_inference_progress(progress_log_fn,
                                              batch_id, len(sampler),
                                              start_time, time()):
                start_time = time()
        progress_log_fn('Inference finished')

    def _infer(self, data, ie_network, stats_collect_callback=None):
        # ie_network.reshape(PartialShape(data.shape))
        filled_input = self._fill_input(ie_network, data)
        compiled_model = self._ie.compile_model(model=ie_network,
                                                device_name=self.config.device)
        infer_request = compiled_model.create_infer_request()
        result = infer_request.infer(filled_input)
        # Collect statistics
        if stats_collect_callback:
            stats_collect_callback(self._transform_for_callback(result))

        return result

    def _fill_input(self, model, image_batch):
        """Matches network input name with corresponding input batch
        :param model: IENetwork instance
        :param image_batch: list of ndarray images or list with a dictionary of inputs mapping
        """
        input_info = model.inputs
        batch_dim = self.config.get('batch_dim', 0)

        def is_dynamic_input(input_blob):
            return input_blob.partial_shape.is_dynamic

        def input_dim(input_blob):
            return len(input_blob.partial_shape)

        def process_input(input_blob, input_data):
            is_sampler_batchfied = len(input_data) != 1
            is_loader_batchfied = input_dim(input_blob) == input_data[0].ndim

            if is_loader_batchfied:
                if input_data[0].shape[batch_dim] == 1:
                    input_data = [np.squeeze(d, batch_dim) for d in input_data]
                    is_loader_batchfied = False
            if not is_sampler_batchfied and not is_loader_batchfied:
                is_sampler_batchfied = True

            assert not (is_sampler_batchfied and is_loader_batchfied), (
                "Data have to be batchfied by either 'stat_batch_size' parameter "
                "in quantization algorithm "
                "or a '__getitem__' method of 'DataLoader' not both."
            )

            input_data_batched = np.concatenate(
                [np.expand_dims(i, batch_dim) for i in input_data], axis=batch_dim
            )
            input_data_batched = input_data_batched.squeeze()
            if is_sampler_batchfied:
                if len(input_data_batched.shape) > 0:
                    if input_data_batched.shape[batch_dim] != len(input_data):
                        input_data_batched = np.expand_dims(input_data_batched, batch_dim)

            if is_dynamic_input(input_blob):
                return input_data_batched
            else:
                return np.reshape(input_data_batched, input_blob.shape)

        if isinstance(image_batch[0], dict):
            feed_dict = {}
            input_blobs = {get_clean_name(in_node.get_node().friendly_name): in_node for in_node in input_info}
            for input_name in image_batch[0].keys():
                input_blob = input_blobs[input_name]
                input_blob_name = self._get_input_any_name(input_blob)
                feed_dict[input_blob_name] = process_input(
                    input_blob, [data[input_name] for data in image_batch]
                )
                if input_dim(input_blob) != feed_dict[input_blob_name].ndim:
                    raise ValueError(
                        "Incompatible input dimension. "
                        f"Cannot infer dimension {feed_dict[input_blob_name].ndim} "
                        f"{Shape(feed_dict[input_blob_name].shape)} "
                        f"into {input_dim(input_blob)}. "
                        "Please make sure batch of input is properly configured."
                    )
            return feed_dict

        if len(input_info) == 1:
            input_blob = next(iter(input_info))
            input_blob_name = self._get_input_any_name(input_blob)
            image_batch = {input_blob_name: process_input(input_blob, image_batch)}
            if input_dim(input_blob) != image_batch[input_blob_name].ndim:
                raise ValueError(
                    "Incompatible input dimension. "
                    f"Cannot infer dimension {image_batch[input_blob_name].ndim} "
                    f"{Shape(image_batch[input_blob_name].shape)} "
                    f"into {input_dim(input_blob)}. "
                    "Please make sure batch of input is properly configured."
                )
            if not is_dynamic_input(input_blob) and Shape(image_batch[input_blob_name].shape) != input_info[0].shape:
                raise ValueError(f"Incompatible input shapes. "
                                 f"Cannot infer {Shape(image_batch[input_blob_name].shape)} into {input_info[0].shape}."
                                 f"Try to specify the layout of the model.")
            return image_batch

        if len(input_info) == 2:
            image_info_nodes = list(filter(
                lambda x: len(x.shape) == 2, input_info))

            if len(image_info_nodes) != 1:
                raise Exception('Two inputs networks must contain exactly one ImageInfo node')

            image_info_node = image_info_nodes[0]
            image_info_name = self._get_input_any_name(image_info_node)
            image_tensor_node = next(iter(filter(
                lambda x: x.get_any_name() != image_info_name, input_info)))
            image_tensor_name = image_tensor_node.get_any_name()

            image_tensor = (image_tensor_name, process_input(image_tensor_node, image_batch))
            if not is_dynamic_input(image_tensor_node) and \
                    Shape(image_tensor[1].shape) != image_tensor_node.shape:
                raise ValueError(f"Incompatible input shapes. "
                                 f"Cannot infer {Shape(image_tensor[1].shape)} into {image_tensor_node.shape}."
                                 f"Try to specify the layout of the model.")

            ch, height, width = image_batch[0].shape
            image_info = (image_info_name,
                          np.stack(np.array([(height, width, ch)] * len(image_batch)), axis=0))

            return dict((k, v) for k, v in [image_tensor, image_info])

        raise Exception('Unsupported number of inputs')


    @staticmethod
    def _transform_for_callback(result):
        batch_size = len(list(result.values())[0])
        if batch_size == 1:
            return result
        return [{key: np.expand_dims(value[i], axis=0) for key, value in result.items()}
                for i in range(batch_size)]

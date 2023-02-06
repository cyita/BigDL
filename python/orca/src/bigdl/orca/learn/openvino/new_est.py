#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import os.path

from pyspark.sql import DataFrame
import ray

from bigdl.orca.data import SparkXShards
from bigdl.orca.learn.spark_estimator import Estimator as SparkEstimator
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils import nest
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.data.utils import spark_df_to_pd_sparkxshards
from bigdl.orca.learn.utils import process_xshards_of_pandas_dataframe,\
    add_predict_to_pd_xshards

from openvino.inference_engine import IECore
from openvino.runtime import Core
import numpy as np
from bigdl.orca.learn.utils import openvino_output_to_sdf
from bigdl.dllib.utils.log4Error import invalidInputError

from typing import (List, Optional, Union)


class Estimator(object):
    @staticmethod
    def from_openvino(*, model_path: str) -> "OpenvinoEstimator":
        """
        Load an openVINO Estimator.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        """
        return OpenvinoEstimator(model_path=model_path)

class OpenvinoEstimator(SparkEstimator):
    def __init__(self,
                 *,
                 model_path: str) -> None:
        self.load(model_path)

    def fit(self, data, epochs, batch_size=32, feature_cols=None, label_cols=None,
            validation_data=None, checkpoint_trigger=None):
        """
        Fit is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def predict(self,  # type: ignore[override]
                data: Union["SparkXShards", "DataFrame", "np.ndarray", List["np.ndarray"]],
                feature_cols: Optional[List[str]] = None,
                batch_size: Optional[int] = 4,
                input_cols: Optional[Union[str, List[str]]]=None, original=False,
                ) -> Optional[Union["SparkXShards", "DataFrame", "np.ndarray", List["np.ndarray"]]]:
        def partition_inference(partition):
            core = Core()
            model = core.read_model(model=self.model_bytes, weights=self.weight_bytes)
            # model.reshape("?, 3, 550, 550")
            model.reshape([2, 3, 550, 550])
            local_model = core.compile_model(model, "CPU", self.config)
            infer_request = local_model.create_infer_request()
            r = infer_request.infer([np.random.rand(2, 3, 550, 550)])
            yield r

        if isinstance(data, DataFrame):
            is_df = True
            xshards = spark_df_to_pd_sparkxshards(data)
            pd_sparkxshards = process_xshards_of_pandas_dataframe(xshards,
                                                                  feature_cols=feature_cols)
            a = pd_sparkxshards.collect()
            # transformed_data = pd_sparkxshards.transform_shard(predict_transform, batch_size)
            
            result = data.rdd.mapPartitions(lambda iter: partition_inference(iter))
            b = result.collect()
            return result
            # result_df = openvino_output_to_sdf(data, result, outputs,
            #                                    list(self.output_dict.values()))
            # return result_df

    def evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None):
        """
        Evaluate is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_model(self):
        """
        Get_model is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def save(self, model_path: str):
        """
        Save is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def load(self, model_path: str) -> None:
        """
        Load an openVINO model.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :return:
        """
        self.node_num, self.core_num = get_node_and_core_number()
        invalidInputError(isinstance(model_path, str), "The model_path should be string.")
        invalidInputError(os.path.exists(model_path), "The model_path should be exist.")
        with open(model_path, 'rb') as file:
            self.model_bytes = file.read()

        with open(model_path[:model_path.rindex(".")] + ".bin", 'rb') as file:
            self.weight_bytes = file.read()

        self.config = {
            'CPU_THREADS_NUM': str(self.core_num), 
            "CPU_BIND_THREAD": "HYBRID_AWARE"
        }
        # core = Core()
        # model = core.read_model(model=self.model_bytes, weights=self.weight_bytes)
        # self.model = core.compile_model(model, "CPU", config)
        # self.inputs = [i.get_names() for i in self.model.inputs]
        # self.outputs = self.model.outputs
        # ie = IECore()
        # config = {'CPU_THREADS_NUM': str(self.core_num)}
        # ie.set_config(config, 'CPU')
        # net = ie.read_network(model=self.model_bytes,
        #                       weights=self.weight_bytes, init_from_buffer=True)
        # self.inputs = list(net.input_info.keys())
        # self.output_dict = {k: v.shape for k, v in net.outputs.items()}
        # del net
        # del ie

    def set_tensorboard(self, log_dir: str, app_name: str):
        """
        Set_tensorboard is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def clear_gradient_clipping(self):
        """
        Clear_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def set_constant_gradient_clipping(self, min, max):
        """
        Set_constant_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Set_l2_norm_gradient_clipping is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_train_summary(self, tag=None):
        """
        Get_train_summary is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def get_validation_summary(self, tag=None):
        """
        Get_validation_summary is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

    def load_orca_checkpoint(self, path, version):
        """
        Load_orca_checkpoint is not supported in OpenVINOEstimator
        """
        invalidInputError(False, "not implemented")

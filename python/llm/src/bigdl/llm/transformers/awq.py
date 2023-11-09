
import torch
import torch.nn as nn
import warnings
import platform
import ctypes
from .utils import logger
from awq.modules.linear import WQLinear_GEMM
from bigdl.llm.ggml.quantize import ggml_tensor_qtype

import bigdl.llm.ggml.model.llama.llama_cpp as ggml

Q4_1 = ggml_tensor_qtype["asym_int4"]

def is_awq_linear(module):
    extra_args = None
    result = False

    if isinstance(module, WQLinear_GEMM):
        extra_args = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "w_bit": module.w_bit,
            "group_size": module.group_size,
            "qweight": module.qweight,
            "qzeros": module.qzeros,
            "scales": module.scales
        }
        
        result = True
    return result, extra_args


def ggml_convert_awq(w_bit: int, group_size: int, qweight: torch.Tensor,
                     qzeros: torch.Tensor, scales: torch.Tensor,
                     n: int, k: int):
    qtype = Q4_1
    QK = ggml.ggml_qk_size(qtype)
    block_size_in_bytes = ggml.ggml_type_size(qtype)

    qweight_src = qweight.data.data_ptr()
    qweight_src = ctypes.cast(qweight_src, ctypes.POINTER(ctypes.c_uint32))

    qzeros_src = qzeros.data.data_ptr()
    qzeros_src = ctypes.cast(qzeros_src, ctypes.POINTER(ctypes.c_uint32))

    scales_src = scales.data.data_ptr()
    scales_src = ctypes.cast(scales_src, ctypes.POINTER(ctypes.c_float))

    dst_size = (n // QK) * block_size_in_bytes
    dst_tensor = torch.empty(dst_size, dtype=torch.uint8,
                             device="cpu")
    dst = ctypes.c_void_p(dst_tensor.data.data_ptr())
    
    ggml.ggml_convert_awq(qweight_src, dst, qzeros_src, scales_src, w_bit, group_size, n, k)
    
    return dst_tensor


def awq_convert(model, optimize_model=True,
                convert_shape_only=False, device="cpu",
                modules_to_not_convert=None, replace_embedding=False):
    modules_to_not_convert = [] if modules_to_not_convert is None else modules_to_not_convert

    # if optimize_model:
    #     model = _optimize_pre(model)

    model, has_been_replaced = _replace_awq_with_low_bit_linear(
        model, Q4_1, modules_to_not_convert,
        None, convert_shape_only, replace_embedding,
    )
    if not has_been_replaced:
        warnings.warn(
            "No linear modules were found in "
            "your model. This can happen for some architectures such as gpt2 that uses Conv1D "
            "instead of Linear layers. Please double check your model architecture, or submit "
            "an issue on github if you think this is a bug."
        )
    elif device == "cpu":
        model.to(torch.float32)
    elif device == "meta":
        # Do nothing here for weights are empty.
        pass

    # if optimize_model:
    #     model = _optimize_post(model)
    return model



def _replace_awq_with_low_bit_linear(model, qtype, modules_to_not_convert=None,
                                     current_key_name=None, convert_shape_only=False,
                                     replace_embedding=False):
    from bigdl.llm.transformers.low_bit_linear import LowBitLinear, FP4Params
    from bigdl.llm.transformers.embedding import LLMEmbedding
    from awq.modules.linear import WQLinear_GEMM
    has_been_replaced = False

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []

        is_linear, linear_args = is_awq_linear(module)
        if is_linear and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                new_linear = LowBitLinear(
                    input_features=linear_args["in_features"],
                    output_features=linear_args["out_features"],
                    qtype=qtype,
                    bias=module.bias is not None,
                    mp_group=None,
                )

                device_type = module.qweight.data.device.type
                # Copy the weights
                paramsLowBit = FP4Params(data=module.qweight.data,
                                         requires_grad=False,
                                         quantized=False,
                                         _shape=torch.Size([linear_args["out_features"], linear_args["in_features"]]),
                                         convert_shape_only=convert_shape_only,
                                         qtype=qtype)
                paramsLowBit.convert_awq(**linear_args)
                new_linear._parameters['weight'] = paramsLowBit

                #  fp16 may generalize to other sizes later
                if new_linear is not None:
                    if module.bias is not None:
                        new_linear._parameters['bias'] = nn.Parameter(module.bias.data)\
                            .to(device_type)

                    model._modules[name] = new_linear
                    has_been_replaced = True
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)

                    module.weight = None
        elif replace_embedding and type(module) == nn.Embedding:
            # skip user-defined Embedding layer
            if platform.system().lower() == 'windows':
                model._modules[name] = LLMEmbedding(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                    _weight=module.weight.data,
                )

        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            _, _flag = _replace_awq_with_low_bit_linear(
                module,
                qtype,
                modules_to_not_convert,
                current_key_name,
                convert_shape_only,
                replace_embedding,
            )
            has_been_replaced = _flag or has_been_replaced
    return model, has_been_replaced


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

import torch
import logging
from bigdl.llm.transformers.xpu_customize_fwd import custom_fwd, custom_bwd
from bigdl.llm.utils.common import invalidInputError

LOG = logging.getLogger("bigdl.llm.rope_embedding")


# Fast RoPE for finetuning, split the q and k
def apply_fast_rope_embedding(q, k, position_ids, model_family):
    if q.device.type != "xpu":
        invalidInputError(False,
                          f"only xpu is supported in this function")
    if model_family in ["llama", "baichuan", "internlm", "aquila", "gpt_neox", "mistral",
                        "mixtral"]:
        q_embed = Fast_RoPE_Embedding.apply(q, position_ids)
        k_embed = Fast_RoPE_Embedding.apply(k, position_ids)
        return q_embed, k_embed
    else:
        invalidInputError(False,
                          f"{model_family} is not supported.")


# Fast RoPE for finetuning, split the q and k
class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, position_ids):
        import linear_q4_0
        x_embed = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        linear_q4_0.apply_rotary_embedding_half_x(x, position_ids,
                                                  x_embed, False)
        ctx.save_for_backward(position_ids)
        return x_embed

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        import linear_q4_0
        position_ids, = ctx.saved_tensors
        x_embed = torch.empty(grad_output.shape,
                              dtype=grad_output.dtype,
                              device=grad_output.device)
        dx = linear_q4_0.apply_rotary_embedding_half_x(grad_output,
                                                       position_ids,
                                                       x_embed,
                                                       True)
        return dx, None

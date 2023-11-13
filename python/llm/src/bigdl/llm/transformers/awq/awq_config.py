from dataclasses import dataclass
from transformers.utils.quantization_config import QuantizationConfigMixin
from transformers.utils.quantization_config import AwqBackendPackingMethod, AWQLinearVersion, QuantizationMethod

@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 )
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: AWQLinearVersion = AWQLinearVersion.GEMM,
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ

        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.backend = backend

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.backend != AwqBackendPackingMethod.AUTOAWQ:
            raise ValueError(
                f"Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} - not recognized backend {self.backend}"
            )

        if self.version not in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV]:
            raise ValueError(
                f"Only supported versions are in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV] - not recognized version {self.version}"
            )

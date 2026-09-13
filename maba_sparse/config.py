from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class MabaSparseConfig:
    dim: int = 640
    d_model: int = 640
    n_layers: int = 16
    num_layers: int = 16
    n_heads: int = 10
    num_heads: int = 10
    d_head: int = 64
    d_k: int = 64
    d_v: int = 64
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_emb: int = 320
    kernel_size: int = 4
    conv_kernel_size: int = 4
    chunk_size: int = 16
    eps: float = 1e-6
    inversion_method: str = "adaptive"
    inversion_threshold: float = 0.5
    adaptive_tol: float = 7e-5
    d_c: int = 128
    d_idx: int = 64
    block_size: int = 64
    window_size: int = 128
    top_k: int = 32
    hca_pool_size: int = 64
    dist_lambda: float = 0.5
    intermediate_size: int = 1728
    rms_norm_eps: float = 1e-6
    residual_gate_bias: float = 2.0
    mtp_depth: int = 2

    def __post_init__(self) -> None:
        if self.dim != 640 and self.d_model == 640:
            self.d_model = self.dim
        elif self.d_model != 640 and self.dim == 640:
            self.dim = self.d_model

        if self.n_heads != 10 and self.num_heads == 10:
            self.num_heads = self.n_heads
        elif self.num_heads != 10 and self.n_heads == 10:
            self.n_heads = self.num_heads

        if self.n_layers != 16 and self.num_layers == 16:
            self.num_layers = self.n_layers
        elif self.num_layers != 16 and self.n_layers == 16:
            self.n_layers = self.num_layers

        if self.kernel_size != 4 and self.conv_kernel_size == 4:
            self.conv_kernel_size = self.kernel_size
        elif self.conv_kernel_size != 4 and self.kernel_size == 4:
            self.kernel_size = self.conv_kernel_size

        if self.d_head == 64 and self.dim != 640:
            self.d_head = self.dim // self.n_heads
        self.d_k = self.d_head
        self.d_v = self.d_head

        valid_methods = {"adaptive", "neumann", "exact"}
        if not isinstance(self.inversion_method, str) or self.inversion_method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid inversion_method: '{self.inversion_method}'. "
                f"Supported options: ['adaptive', 'neumann', 'exact']."
            )
        self.inversion_method = self.inversion_method.lower()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MabaSparseConfig":
        return cls(**d)

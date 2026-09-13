import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGIndexer(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        d_idx: int = 64,
        block_size: int = 64,
        top_k: int = 32,
        dist_lambda: float = 0.5,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = getattr(config, "dim", getattr(config, "d_model", dim))
            d_idx = getattr(config, "d_idx", d_idx)
            block_size = getattr(config, "block_size", block_size)
            top_k = getattr(config, "top_k", top_k)
            dist_lambda = getattr(config, "dist_lambda", dist_lambda)

        self.dim = dim
        self.d_idx = d_idx
        self.block_size = block_size
        self.top_k = top_k
        self.dist_lambda = dist_lambda

        self.q_idx_proj = nn.Linear(dim, d_idx, bias=False)
        self.k_idx_proj = nn.Linear(dim, d_idx, bias=False)
        self.scale = 1.0 / math.sqrt(d_idx)

    def forward(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
        dist_lambda: Optional[float] = None,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        b, l, d = x.shape
        k = self.top_k if top_k is None else top_k
        lam = self.dist_lambda if dist_lambda is None else dist_lambda

        qi = self.q_idx_proj(x) * self.scale
        ki = self.k_idx_proj(x)

        nb = (l + self.block_size - 1) // self.block_size
        pad = nb * self.block_size - l
        kp = F.pad(ki, (0, 0, 0, pad), value=0.0) if pad > 0 else ki

        kb = kp.view(b, nb, self.block_size, self.d_idx)
        c = 0.5 * (kb.mean(dim=2) + kb.max(dim=2)[0])

        s = torch.einsum("bld,bnd->bln", qi, c)

        qi_idx = torch.arange(l, device=x.device).unsqueeze(1) // self.block_size
        ni_idx = torch.arange(nb, device=x.device).unsqueeze(0)
        dist = (qi_idx - ni_idx).abs().float()
        pen = lam * torch.log(1.0 + dist)
        sc = s - pen.unsqueeze(0)

        msk = ni_idx > qi_idx
        sc = sc.masked_fill(msk.unsqueeze(0), float("-inf"))

        ak = min(k, nb)
        _, idx = torch.topk(sc, k=ak, dim=-1)

        if return_scores:
            return idx, c, sc
        return idx, c


DeltaGuidedCentroidIndexer = DGIndexer

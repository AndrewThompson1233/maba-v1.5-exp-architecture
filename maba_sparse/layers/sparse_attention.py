import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.layers.indexer import DGIndexer


class MabaSparseAttention(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        dim: int = 640,
        n_heads: int = 10,
        d_head: int = 64,
        d_c: int = 128,
        window_size: int = 128,
        block_size: int = 64,
        top_k: int = 32,
        hca_pool_size: int = 64,
        dist_lambda: float = 0.5,
        d_idx: int = 64,
    ) -> None:
        super().__init__()
        if config is not None:
            self.dim = getattr(config, "dim", getattr(config, "d_model", dim))
            self.n_heads = getattr(config, "n_heads", getattr(config, "num_heads", n_heads))
            self.d_head = getattr(config, "d_head", d_head)
            self.d_c = getattr(config, "d_c", d_c)
            self.window_size = getattr(config, "window_size", window_size)
            self.block_size = getattr(config, "block_size", block_size)
            self.top_k = getattr(config, "top_k", top_k)
            self.hca_pool_size = getattr(config, "hca_pool_size", hca_pool_size)
            self.dist_lambda = getattr(config, "dist_lambda", dist_lambda)
            d_idx = getattr(config, "d_idx", d_idx)
        else:
            self.dim = dim
            self.n_heads = n_heads
            self.d_head = d_head
            self.d_c = d_c
            self.window_size = window_size
            self.block_size = block_size
            self.top_k = top_k
            self.hca_pool_size = hca_pool_size
            self.dist_lambda = dist_lambda

        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.d_head, bias=False)
        self.kv_down_proj = nn.Linear(self.dim, self.d_c, bias=False)
        self.k_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)
        self.v_up_proj = nn.Linear(self.d_c, self.n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.dim, bias=False)
        self.stream_gate = nn.Linear(self.dim, 3, bias=True)
        self.indexer = DGIndexer(
            dim=self.dim,
            d_idx=d_idx,
            block_size=self.block_size,
            top_k=self.top_k,
            dist_lambda=self.dist_lambda,
        )

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.stream_gate(x), dim=-1)

    def _compute_local_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L_full: int
    ) -> torch.Tensor:
        b, h, lq, _ = q.shape
        lkv = k.shape[2]
        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        qo = lkv - lq
        i = (qo + torch.arange(lq, device=q.device)).unsqueeze(1)
        j = torch.arange(lkv, device=q.device).unsqueeze(0)
        c = j <= i
        w = (i - j) < self.window_size
        s = j < min(4, lkv)
        m = c & (w | s)
        a = a.masked_fill(~m.unsqueeze(0).unsqueeze(0), float("-inf"))
        p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
        return torch.matmul(p, v)

    def _compute_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        top_indices: torch.Tensor,
        L_full: int,
    ) -> torch.Tensor:
        b, h, lq, _ = q.shape
        lkv = k.shape[2]
        nb = (lkv + self.block_size - 1) // self.block_size
        bm = torch.zeros(b, lq, nb, device=q.device, dtype=torch.bool)
        bm.scatter_(2, top_indices, True)
        tbi = torch.arange(lkv, device=q.device) // self.block_size
        tm = bm[:, :, tbi]
        qo = lkv - lq
        i = (qo + torch.arange(lq, device=q.device)).unsqueeze(1)
        j = torch.arange(lkv, device=q.device).unsqueeze(0)
        tm = tm & (j <= i).unsqueeze(0)
        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        a = a.masked_fill(~tm.unsqueeze(1), float("-inf"))
        a = torch.nan_to_num(a, nan=-1e9, neginf=-1e9)
        p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
        return torch.matmul(p, v)

    def _compute_hca_attention(
        self, q: torch.Tensor, c_kv: torch.Tensor, L_full: int
    ) -> torch.Tensor:
        b, h, lq, dh = q.shape
        lkv = c_kv.shape[1]
        r = self.hca_pool_size
        nt = (lkv + r - 1) // r
        if nt == 0:
            return torch.zeros(b, h, lq, dh, device=q.device, dtype=q.dtype)
        pad = nt * r - lkv
        cp = F.pad(c_kv, (0, 0, 0, pad)) if pad > 0 else c_kv
        c_pool = cp.view(b, nt, r, self.d_c).mean(dim=2)
        kh = self.k_up_proj(c_pool).view(b, nt, h, dh).transpose(1, 2)
        vh = self.v_up_proj(c_pool).view(b, nt, h, dh).transpose(1, 2)
        a = torch.matmul(q, kh.transpose(-1, -2)) * self.scale
        qo = lkv - lq
        qi = (qo + torch.arange(lq, device=q.device)).unsqueeze(1)
        bs = torch.arange(nt, device=q.device) + 1
        ei = (bs * r - 1).unsqueeze(0)
        c = ei <= qi
        a = a.masked_fill(~c.unsqueeze(0).unsqueeze(0), float("-inf"))
        p = torch.nan_to_num(F.softmax(a, dim=-1), nan=0.0)
        return torch.matmul(p, vh)

    def forward(
        self, x: torch.Tensor, past_c_kv: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        c_kv = self.kv_down_proj(x)
        c_full = torch.cat([past_c_kv, c_kv], dim=1) if past_c_kv is not None else c_kv
        lf = c_full.shape[1]

        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_up_proj(c_full).view(b, lf, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up_proj(c_full).view(b, lf, self.n_heads, self.d_head).transpose(1, 2)

        ol = self._compute_local_attention(q, k, v, lf)
        idx, _ = self.indexer(x)
        os = self._compute_sparse_attention(q, k, v, idx, lf)
        oh = self._compute_hca_attention(q, c_full, lf)

        g = F.softmax(self.stream_gate(x), dim=-1)
        gl = g[:, :, 0:1].unsqueeze(1)
        gs = g[:, :, 1:2].unsqueeze(1)
        gh = g[:, :, 2:3].unsqueeze(1)

        o = gl * ol + gs * os + gh * oh
        o = o.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head)
        return self.o_proj(o), c_full


MABASALayer = MabaSparseAttention

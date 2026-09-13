import math
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.model import FactorizedEmbeddings, MabaSparseOutput, RMSNorm, SwiGLUFFN


class DenseAttention(nn.Module):
    def __init__(self, dim: int = 640, n_heads: int = 10, d_head: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.q_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        nkv = (k, v)
        c = (kv_cache is None) and (l > 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=c)
        o = o.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head)
        return self.o_proj(o), nkv


class DenseTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        n_heads: int = 10,
        d_head: int = 64,
        intermediate_size: int = 1728,
        eps: float = 1e-6,
        residual_gate_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.mixer = DenseAttention(dim, n_heads, d_head)
        self.res_gate1 = nn.Parameter(torch.full((dim,), residual_gate_bias))

        self.norm2 = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLUFFN(dim, intermediate_size)
        self.res_gate2 = nn.Parameter(torch.full((dim,), residual_gate_bias))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.norm1(x)
        ao, nkv = self.mixer(h, kv_cache=kv_cache)
        x = x + torch.sigmoid(self.res_gate1) * ao
        x = x + torch.sigmoid(self.res_gate2) * self.ffn(self.norm2(x))
        return x, nkv


class DenseTransformerForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32768,
        d_emb: int = 128,
        dim: int = 640,
        n_layers: int = 20,
        n_heads: int = 10,
        d_head: int = 64,
        intermediate_size: int = 1728,
        eps: float = 1e-6,
        residual_gate_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.embeddings = FactorizedEmbeddings(vocab_size, d_emb, dim)
        self.layers = nn.ModuleList([
            DenseTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                d_head=d_head,
                intermediate_size=intermediate_size,
                eps=eps,
                residual_gate_bias=residual_gate_bias,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim, eps=eps)
        self.head_proj = nn.Linear(dim, d_emb, bias=False)
        self.lm_head = nn.Linear(d_emb, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.in_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_states: Optional[List[Any]] = None,
    ) -> MabaSparseOutput:
        if targets is None and labels is not None:
            targets = labels

        x = self.embeddings(input_ids)
        nps = []

        for i, layer in enumerate(self.layers):
            kv = past_states[i] if past_states is not None else None
            x, nkv = layer(x, kv_cache=kv)
            nps.append(nkv)

        xn = self.final_norm(x)
        logits = self.lm_head(self.head_proj(xn))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return MabaSparseOutput(
            logits=logits,
            loss=loss,
            past_states=nps,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        self.eval()
        gen = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self(gen)
            nl = out.logits[:, -1, :]
            if temperature > 0:
                nl = nl / temperature
                if top_k is not None:
                    v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                    nl[nl < v[:, [-1]]] = float("-inf")
                p = F.softmax(nl, dim=-1)
                tok = torch.multinomial(p, num_samples=1)
            else:
                tok = torch.argmax(nl, dim=-1, keepdim=True)
            gen = torch.cat([gen, tok], dim=1)
        return gen


DenseTransformerLM = DenseTransformerForCausalLM

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvState(tuple):
    @property
    def shape(self) -> torch.Size:
        if len(self) == 0:
            return torch.Size([0, 0, 0, 0])
        f = self[0]
        return torch.Size([f.shape[0], len(self), f.shape[1], f.shape[2]])

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(self, dim=1)


class DGDALayer(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        dim: int = 640,
        n_heads: int = 10,
        d_head: Optional[int] = None,
        kernel_size: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if config is not None:
            dim = getattr(config, "dim", getattr(config, "d_model", dim))
            n_heads = getattr(config, "n_heads", getattr(config, "num_heads", n_heads))
            d_head = getattr(config, "d_head", d_head)
            kernel_size = getattr(
                config, "kernel_size", getattr(config, "conv_kernel_size", kernel_size)
            )
            eps = getattr(config, "eps", getattr(config, "rms_norm_eps", eps))

        self.dim = dim
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else (dim // n_heads)
        self.d_k = self.d_head
        self.d_v = self.d_head
        self.kernel_size = kernel_size
        self.k_size = kernel_size
        self.eps = eps
        self.chunk_size = getattr(config, "chunk_size", 16)
        self.inversion_method = getattr(config, "inversion_method", "adaptive")
        self.adaptive_tol = getattr(config, "adaptive_tol", 7e-5)

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_heads * self.d_v, bias=False)

        self.conv_q = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, groups=self.dim, bias=False, padding=0
        )
        self.conv_k = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, groups=self.dim, bias=False, padding=0
        )
        self.conv_v = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, groups=self.dim, bias=False, padding=0
        )

        self.gate_alpha = nn.Linear(self.dim, self.n_heads * self.d_k, bias=False)
        self.gate_erase = nn.Linear(self.dim, self.n_heads * self.d_k, bias=False)
        self.gate_write = nn.Linear(self.dim, self.n_heads * self.d_v, bias=False)

        self.alpha_proj = self.gate_alpha
        self.b_proj = self.gate_erase
        self.w_proj = self.gate_write
        self.o_proj = nn.Linear(self.n_heads * self.d_v, self.dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.gate_alpha.weight)
        nn.init.xavier_uniform_(self.gate_erase.weight)
        nn.init.xavier_uniform_(self.gate_write.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.normal_(self.conv_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_v.weight, mean=0.0, std=0.02)

    def _apply_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.kernel_size
        xt = x.transpose(1, 2)
        p = torch.cat([conv_state, xt], dim=2) if conv_state is not None else F.pad(xt, (k - 1, 0))
        ns = p[:, :, -(k - 1):].contiguous()
        y = F.silu(conv(p)).transpose(1, 2)
        return y, ns

    def _unpack_conv_state(
        self,
        conv_state: Optional[Union[ConvState, Tuple[torch.Tensor, ...], torch.Tensor]],
        b_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if conv_state is None:
            return None, None, None
        if isinstance(conv_state, (tuple, list)):
            if len(conv_state) == 3:
                return conv_state[0], conv_state[1], conv_state[2]
            if len(conv_state) == 1 and isinstance(conv_state[0], (tuple, list)):
                return conv_state[0][0], conv_state[0][1], conv_state[0][2]
        if isinstance(conv_state, torch.Tensor):
            if conv_state.dim() == 4 and conv_state.shape[1] == 3:
                return conv_state[:, 0], conv_state[:, 1], conv_state[:, 2]
            if conv_state.dim() == 5 and conv_state.shape[1] == 3:
                cs = conv_state.view(b_size, 3, self.dim, self.kernel_size - 1)
                return cs[:, 0], cs[:, 1], cs[:, 2]
            if conv_state.dim() == 4:
                cs = conv_state.view(b_size, self.dim, self.kernel_size - 1)
                return cs, cs, cs
            if conv_state.dim() == 3:
                return conv_state, conv_state, conv_state
        raise ValueError(f"Unsupported conv_state shape or type: {type(conv_state)}")

    def _chunk_neumann(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        log_alpha: torch.Tensor,
        S0: torch.Tensor,
        inversion_method: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = q.shape[2]
        cla = torch.cumsum(log_alpha, dim=-2)
        diff = cla.unsqueeze(3) - cla.unsqueeze(2)
        dec = torch.exp(torch.clamp(diff, max=0.0))

        bk = (b * k).unsqueeze(3)
        ks = k.unsqueeze(2)
        l_mat = torch.tril((bk * dec * ks).sum(dim=-1), diagonal=-1)

        lam = torch.exp(cla)
        bh = (b * k) * lam
        ve = (w * v) - torch.matmul(bh, S0)

        eye = torch.eye(t, dtype=q.dtype, device=q.device).view(1, 1, t, t)
        l2 = torch.matmul(l_mat, l_mat)
        l3 = torch.matmul(l2, l_mat)
        inv_l = eye - l_mat + l2 - l3
        u_neu = torch.matmul(inv_l, ve)

        m = inversion_method if inversion_method is not None else getattr(self, "inversion_method", "adaptive")
        tol = getattr(self, "adaptive_tol", 7e-5)

        if m == "neumann":
            exact = False
        elif m == "exact":
            exact = True
        elif m == "adaptive":
            r = ve - u_neu - torch.matmul(l_mat, u_neu)
            exact = r.abs().max() > tol or torch.isnan(u_neu).any() or torch.isinf(u_neu).any()
        else:
            raise ValueError(f"Unknown inversion_method: '{m}'")

        if not exact:
            u = u_neu
            clae = cla[:, :, -1:, :]
            dte = torch.exp(clae - cla)
            kd = k * dte
            sc = torch.matmul(kd.transpose(-1, -2), u)
            ds0 = torch.exp(clae).transpose(-1, -2)
            sf = ds0 * S0 + sc

            oi = torch.matmul(q * lam, S0)
            qs = q.unsqueeze(3)
            a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)
            ot = torch.matmul(a_mat, u)
            return oi + ot, sf
        else:
            orig_dt = q.dtype
            if q.device.type == "cpu":
                qd, kd, vd, bd, wd, lad, s0d = [x.double() for x in (q, k, v, b, w, log_alpha, S0)]
                clad = torch.cumsum(lad, dim=-2)
                diffd = clad.unsqueeze(3) - clad.unsqueeze(2)
                decd = torch.exp(torch.clamp(diffd, max=0.0))
                bkd = (bd * kd).unsqueeze(3)
                ksd = kd.unsqueeze(2)
                ld = torch.tril((bkd * decd * ksd).sum(dim=-1), diagonal=-1)
                lamd = torch.exp(clad)
                bhd = (bd * kd) * lamd
                ved = (wd * vd) - torch.matmul(bhd, s0d)
                eyed = torch.eye(t, dtype=torch.float64, device=q.device).view(1, 1, t, t)
                ud = torch.linalg.solve_triangular(eyed + ld, ved, upper=False)

                claed = clad[:, :, -1:, :]
                dted = torch.exp(claed - clad)
                kdd = kd * dted
                scd = torch.matmul(kdd.transpose(-1, -2), ud)
                ds0d = torch.exp(claed).transpose(-1, -2)
                sfd = ds0d * s0d + scd

                oid = torch.matmul(qd * lamd, s0d)
                qsd = qd.unsqueeze(3)
                ad = torch.tril((qsd * decd * ksd).sum(dim=-1), diagonal=0)
                otd = torch.matmul(ad, ud)
                return (oid + otd).to(orig_dt), sfd.to(orig_dt)
            else:
                cdt = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
                u = torch.linalg.solve_triangular((eye + l_mat).to(cdt), ve.to(cdt), upper=False).to(orig_dt)
                clae = cla[:, :, -1:, :]
                dte = torch.exp(clae - cla)
                kd = k * dte
                sc = torch.matmul(kd.transpose(-1, -2), u)
                ds0 = torch.exp(clae).transpose(-1, -2)
                sf = ds0 * S0 + sc

                oi = torch.matmul(q * lam, S0)
                qs = q.unsqueeze(3)
                a_mat = torch.tril((qs * dec * ks).sum(dim=-1), diagonal=0)
                ot = torch.matmul(a_mat, u)
                return oi + ot, sf

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        chunk_size: int = 16,
        inversion_method: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConvState]:
        b, l, d = x.shape
        h, dk, dv = self.n_heads, self.d_k, self.d_v

        if l == 0:
            eo = torch.empty(b, 0, d, dtype=x.dtype, device=x.device)
            es = state if state is not None else torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device)
            zc = torch.zeros(b, d, self.kernel_size - 1, dtype=x.dtype, device=x.device)
            return eo, es, ConvState((zc, zc, zc))

        cq, ck, cv = self._unpack_conv_state(conv_state, b, x.device, x.dtype)
        q, nq = self._apply_conv(self.q_proj(x), self.conv_q, cq)
        k, nk = self._apply_conv(self.k_proj(x), self.conv_k, ck)
        v, nv = self._apply_conv(self.v_proj(x), self.conv_v, cv)
        ncs = ConvState((nq, nk, nv))

        eb = torch.sigmoid(self.gate_erase(x))
        ew = torch.sigmoid(self.gate_write(x))
        la = -F.softplus(self.gate_alpha(x))

        q = q.view(b, l, h, dk).transpose(1, 2)
        k = k.view(b, l, h, dk).transpose(1, 2)
        v = v.view(b, l, h, dv).transpose(1, 2)
        eb = eb.view(b, l, h, dk).transpose(1, 2)
        ew = ew.view(b, l, h, dv).transpose(1, 2)
        la = la.view(b, l, h, dk).transpose(1, 2)

        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.eps)

        cs = torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device) if state is None else state.clone()

        n_chunks = l // chunk_size
        rem = l % chunk_size
        outs = []

        for i in range(n_chunks):
            s = i * chunk_size
            e = s + chunk_size
            oc, cs = self._chunk_neumann(
                q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], eb[:, :, s:e], ew[:, :, s:e], la[:, :, s:e], cs, inversion_method=inversion_method
            )
            outs.append(oc)

        if rem > 0:
            s = n_chunks * chunk_size
            oc, cs = self._chunk_neumann(
                q[:, :, s:], k[:, :, s:], v[:, :, s:], eb[:, :, s:], ew[:, :, s:], la[:, :, s:], cs, inversion_method=inversion_method
            )
            outs.append(oc)

        o = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(b, l, h * dv)
        return self.o_proj(o), cs, ncs

    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        conv_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConvState]:
        b, l = x.shape[0], x.shape[1]
        assert l == 1
        h, dk, dv = self.n_heads, self.d_k, self.d_v

        cq, ck, cv = self._unpack_conv_state(conv_state, b, x.device, x.dtype)
        q, nq = self._apply_conv(self.q_proj(x), self.conv_q, cq)
        k, nk = self._apply_conv(self.k_proj(x), self.conv_k, ck)
        v, nv = self._apply_conv(self.v_proj(x), self.conv_v, cv)
        ncs = ConvState((nq, nk, nv))

        eb = torch.sigmoid(self.gate_erase(x)).view(b, 1, h, dk).transpose(1, 2)
        ew = torch.sigmoid(self.gate_write(x)).view(b, 1, h, dv).transpose(1, 2)
        la = -F.softplus(self.gate_alpha(x)).view(b, 1, h, dk).transpose(1, 2)
        alpha = torch.exp(la)

        q = q.view(b, 1, h, dk).transpose(1, 2)
        k = k.view(b, 1, h, dk).transpose(1, 2)
        v = v.view(b, 1, h, dv).transpose(1, 2)

        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + self.eps)

        sp = torch.zeros(b, h, dk, dv, dtype=x.dtype, device=x.device) if state is None else state
        sd = alpha.transpose(-1, -2) * sp
        beta = eb * k
        bs = torch.matmul(beta, sd)
        delta = (ew * v) - bs
        st = sd + torch.matmul(k.transpose(-1, -2), delta)

        oh = torch.matmul(q, st)
        o = self.o_proj(oh.transpose(1, 2).reshape(b, 1, h * dv))
        return o, st, ncs


DecoupledGatedDeltaAttention = DGDALayer

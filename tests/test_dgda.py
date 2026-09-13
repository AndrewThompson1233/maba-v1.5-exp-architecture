"""Automated unit and integration test suite for Decoupled Gated Delta Attention (DGDA).

Validates Milestone 1 acceptance criteria:
1. Chunkwise vs Sequential Recurrent State & Output Equivalence (< 1e-4 tolerance).
2. Autoregressive step() decode vs full forward() equivalence.
3. Analytical gradient continuity (finite, non-zero, zero-NaN gradients on all parameters).
4. Strict causal masking isolation (past output unchanged; future gradients identically 0.0).
5. Numerical stability across float32, bfloat16, and float16 (no autograd trap).
6. O(1) state memory footprint invariance across sequence lengths.
7. Robust boundary conditions (arbitrary lengths L, zero norm keys, uninitialized states).
"""

import math
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import ConvState, DGDALayer

# ==============================================================================
# Mathematical Sequential Reference Implementation (Gold Standard)
# ==============================================================================


def sequential_dgda_reference(
    x: torch.Tensor,
    layer: DGDALayer,
    state: Optional[torch.Tensor] = None,
    conv_state: Optional[Union[ConvState, Tuple[torch.Tensor, ...]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, ConvState]:
    """Pure sequential token-by-token recurrence reference implementation of DGDA.

    Executes a standard Python loop without chunking or Neumann series approximations.
    Used as the mathematical ground-truth for parity verification.
    """
    B, L, D = x.shape
    H, d_k, d_v = layer.n_heads, layer.d_head, layer.d_head
    k_size = layer.kernel_size

    cs_q, cs_k, cs_v = layer._unpack_conv_state(conv_state, B, x.device, x.dtype)

    def _conv(val: torch.Tensor, conv_mod: nn.Conv1d, cs: Optional[torch.Tensor]):
        xt = val.transpose(1, 2)
        if cs is not None:
            pad = torch.cat([cs, xt], dim=2)
        else:
            pad = F.pad(xt, (k_size - 1, 0))
        tail = k_size - 1
        new_cs = pad[:, :, -tail:]
        out = F.silu(conv_mod(pad)).transpose(1, 2)
        return out, new_cs

    q_conv, ncsq = _conv(layer.q_proj(x), layer.conv_q, cs_q)
    k_conv, ncsk = _conv(layer.k_proj(x), layer.conv_k, cs_k)
    v_conv, ncsv = _conv(layer.v_proj(x), layer.conv_v, cs_v)
    ref_conv_state = ConvState((ncsq, ncsk, ncsv))

    b = torch.sigmoid(layer.gate_erase(x)).view(B, L, H, d_k)
    w = torch.sigmoid(layer.gate_write(x)).view(B, L, H, d_v)
    log_alpha = -F.softplus(layer.gate_alpha(x).float())
    alpha = torch.exp(log_alpha).to(x.dtype).view(B, L, H, d_k)

    q = q_conv.view(B, L, H, d_k)
    k = F.normalize(k_conv.view(B, L, H, d_k), p=2, dim=-1, eps=layer.eps)
    v = v_conv.view(B, L, H, d_v)

    beta = b * k
    u = w * v

    if state is None:
        S = torch.zeros(B, H, d_k, d_v, dtype=x.dtype, device=x.device)
    else:
        S = state.clone()

    outs = []
    for t in range(L):
        # Scale rows of S by alpha_t
        S_decay = alpha[:, t, :, :, None] * S  # [B, H, d_k, d_v]
        # Erase projection: (b_t * k_t) @ S_decay
        pred = torch.matmul(beta[:, t].unsqueeze(-2), S_decay)  # [B, H, 1, d_v]
        # Delta innovation: u_t - pred
        delta = u[:, t].unsqueeze(-2) - pred  # [B, H, 1, d_v]
        # Rank-1 state update
        S = S_decay + torch.matmul(k[:, t].unsqueeze(-1), delta)  # [B, H, d_k, d_v]
        # Readout: q_t @ S_t
        ot = torch.matmul(q[:, t].unsqueeze(-2), S).squeeze(-2)  # [B, H, d_v]
        outs.append(ot)

    if outs:
        out_stacked = torch.stack(outs, dim=1).reshape(B, L, H * d_v)
        out_proj = layer.o_proj(out_stacked)
    else:
        out_proj = torch.empty(B, 0, D, dtype=x.dtype, device=x.device)

    return out_proj, S, ref_conv_state


# ==============================================================================
# Test 1: Layer Initialization & Weight Parameter Integrity
# ==============================================================================


def test_dgda_initialization(config):
    """Verify weight matrices, causal convolutions, aliases, and parameter dimensions."""
    assert isinstance(config, MabaSparseConfig)
    layer = DGDALayer(config)

    assert layer.dim == 640
    assert layer.n_heads == 10
    assert layer.d_head == 64
    assert layer.d_k == 64
    assert layer.d_v == 64
    assert layer.kernel_size == 4
    assert layer.k_size == 4

    # Verify projection shapes
    assert layer.q_proj.weight.shape == (640, 640)
    assert layer.k_proj.weight.shape == (640, 640)
    assert layer.v_proj.weight.shape == (640, 640)
    assert layer.gate_erase.weight.shape == (640, 640)
    assert layer.gate_write.weight.shape == (640, 640)
    assert layer.gate_alpha.weight.shape == (640, 640)
    assert layer.o_proj.weight.shape == (640, 640)

    # Verify compatibility aliases
    assert layer.b_proj is layer.gate_erase
    assert layer.w_proj is layer.gate_write
    assert layer.alpha_proj is layer.gate_alpha

    # Verify depthwise causal conv shapes (groups=dim)
    assert layer.conv_q.weight.shape == (640, 1, 4)
    assert layer.conv_k.weight.shape == (640, 1, 4)
    assert layer.conv_v.weight.shape == (640, 1, 4)


# ==============================================================================
# Test 2: Chunkwise vs Sequential Recurrent State & Output Equivalence (< 1e-4)
# ==============================================================================


class TestChunkwiseVsSequentialEquivalence:
    """Validates that chunkwise parallel prefill matches exact sequential recurrence within < 1e-4."""

    @pytest.mark.parametrize("seq_len", [16, 32, 48, 64])
    def test_chunkwise_vs_sequential_multi_length(self, dgda_layer, seq_len):
        """Verify max difference < 1e-4 across multiple sequence lengths (L=16, 32, 48, 64)."""
        torch.manual_seed(100 + seq_len)
        B, D = 2, dgda_layer.dim
        x = torch.randn(B, seq_len, D)

        out_seq, state_seq, _ = sequential_dgda_reference(x, dgda_layer)
        out_chunk, state_chunk, _ = dgda_layer(x, chunk_size=16)

        diff_state = (state_seq - state_chunk).abs().max().item()
        diff_out = (out_seq - out_chunk).abs().max().item()

        assert diff_state < 1e-4, f"State diff {diff_state:.6e} >= 1e-4 at L={seq_len}"
        assert diff_out < 1e-4, f"Output diff {diff_out:.6e} >= 1e-4 at L={seq_len}"

    @pytest.mark.parametrize("chunk_size", [16, 32, 64])
    def test_chunkwise_vs_sequential_various_chunk_sizes(self, dgda_layer, chunk_size):
        """Verify that configuring chunk sizes C=16, 32, 64 maintains parity < 1e-4 on L=64."""
        torch.manual_seed(200 + chunk_size)
        B, L, D = 2, 64, dgda_layer.dim
        x = torch.randn(B, L, D)

        out_seq, state_seq, _ = sequential_dgda_reference(x, dgda_layer)
        out_chunk, state_chunk, _ = dgda_layer(x, chunk_size=chunk_size)

        diff_state = (state_seq - state_chunk).abs().max().item()
        diff_out = (out_seq - out_chunk).abs().max().item()

        assert (
            diff_state < 1e-4
        ), f"State diff {diff_state:.6e} >= 1e-4 at C={chunk_size}"
        assert diff_out < 1e-4, f"Output diff {diff_out:.6e} >= 1e-4 at C={chunk_size}"

    def test_chunkwise_with_non_zero_initial_state(self, dgda_layer):
        """Verify chunkwise prefill correctly integrates non-zero initial recurrent state S0."""
        torch.manual_seed(300)
        B, L, D = 2, 32, dgda_layer.dim
        H, d_k, d_v = dgda_layer.n_heads, dgda_layer.d_head, dgda_layer.d_head

        initial_state = torch.randn(B, H, d_k, d_v) * 0.5
        x = torch.randn(B, L, D)

        out_seq, state_seq, _ = sequential_dgda_reference(
            x, dgda_layer, state=initial_state
        )
        out_chunk, state_chunk, _ = dgda_layer(x, state=initial_state, chunk_size=16)

        diff_state = (state_seq - state_chunk).abs().max().item()
        diff_out = (out_seq - out_chunk).abs().max().item()

        assert diff_state < 1e-4, f"State diff with S0 {diff_state:.6e} >= 1e-4"
        assert diff_out < 1e-4, f"Output diff with S0 {diff_out:.6e} >= 1e-4"


# ==============================================================================
# Test 3: Autoregressive Single-Token step() Decode vs forward() Equivalence
# ==============================================================================


class TestStepVsForwardEquivalence:
    """Validates that token-by-token decode via step() matches forward() execution."""

    def test_step_loop_vs_sequential_forward_exact(self, dgda_layer):
        """Verify that looping through step() matches sequential reference within < 1e-6."""
        torch.manual_seed(400)
        B, L, D = 2, 16, dgda_layer.dim
        x = torch.randn(B, L, D)

        out_fwd, state_fwd, _ = sequential_dgda_reference(x, dgda_layer)

        state = None
        conv_state = None
        step_outputs = []
        for t in range(L):
            t_next = t + 1
            xt = x[:, t:t_next, :]
            ot, state, conv_state = dgda_layer.step(
                xt, state=state, conv_state=conv_state
            )
            step_outputs.append(ot)

        out_step = torch.cat(step_outputs, dim=1)

        diff_out = (out_fwd - out_step).abs().max().item()
        diff_state = (state_fwd - state).abs().max().item()

        assert (
            diff_out < 1e-6
        ), f"step() vs forward() output diff {diff_out:.6e} >= 1e-6"
        assert (
            diff_state < 1e-6
        ), f"step() vs forward() state diff {diff_state:.6e} >= 1e-6"

    def test_step_loop_vs_chunkwise_forward(self, dgda_layer):
        """Verify that looping through step() matches chunkwise prefill within < 1e-4."""
        torch.manual_seed(401)
        B, L, D = 2, 48, dgda_layer.dim
        x = torch.randn(B, L, D)

        out_chunk, state_chunk, _ = dgda_layer(x, chunk_size=16)

        state = None
        conv_state = None
        step_outputs = []
        for t in range(L):
            t_next = t + 1
            xt = x[:, t:t_next, :]
            ot, state, conv_state = dgda_layer.step(
                xt, state=state, conv_state=conv_state
            )
            step_outputs.append(ot)

        out_step = torch.cat(step_outputs, dim=1)

        diff_out = (out_chunk - out_step).abs().max().item()
        diff_state = (state_chunk - state).abs().max().item()

        assert (
            diff_out < 1e-4
        ), f"step() vs chunkwise output diff {diff_out:.6e} >= 1e-4"
        assert (
            diff_state < 1e-4
        ), f"step() vs chunkwise state diff {diff_state:.6e} >= 1e-4"

    def test_conv_state_caching_invariance(self, dgda_layer):
        """Verify conv_state preserves causal left-padding across successive calls."""
        torch.manual_seed(402)
        B, D = 2, dgda_layer.dim
        k = dgda_layer.kernel_size

        x1 = torch.randn(B, 1, D)
        x2 = torch.randn(B, 1, D)

        # Step 1
        _, _, conv_state1 = dgda_layer.step(x1)
        assert conv_state1 is not None
        assert conv_state1.shape == (B, 3, D, k - 1)

        for cs in conv_state1:
            assert cs.shape == (B, D, k - 1)

        # Step 2
        _, _, conv_state2 = dgda_layer.step(x2, conv_state=conv_state1)
        assert conv_state2.shape == (B, 3, D, k - 1)

        for cs in conv_state2:
            assert cs.shape == (B, D, k - 1)


# ==============================================================================
# Test 4: Analytical Gradient Continuity (Finite, Non-Zero, Zero-NaNs)
# ==============================================================================


class TestAnalyticalGradientContinuity:
    """Validates that backward computes non-zero, finite, NaN-free gradients across all parameters."""

    @pytest.mark.parametrize("chunk_size", [16, 32])
    def test_backward_all_parameters_finite_and_nonzero(self, small_config, chunk_size):
        """Verify that dLoss/dParam is non-zero, finite, and free of NaNs for all weights."""
        torch.manual_seed(500)
        layer = DGDALayer(small_config)
        layer.train()

        B, L, D = 2, 32, layer.dim
        x = torch.randn(B, L, D, requires_grad=True)

        out, final_state, _ = layer(x, chunk_size=chunk_size)
        loss = out.sum() + final_state.sum()
        loss.backward()

        assert x.grad is not None, "Input x.grad is None"
        assert not torch.isnan(x.grad).any(), "Input x.grad contains NaNs"
        assert not torch.isinf(x.grad).any(), "Input x.grad contains Infs"
        assert torch.count_nonzero(x.grad) > 0, "Input x.grad is entirely zero"

        for name, param in layer.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"Param {name} grad is None"
            assert not torch.isnan(param.grad).any(), f"Param {name} grad has NaNs"
            assert not torch.isinf(param.grad).any(), f"Param {name} grad has Infs"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0.0, f"Param {name} grad norm is zero ({grad_norm})"
            assert math.isfinite(
                grad_norm
            ), f"Param {name} grad norm is not finite: {grad_norm}"

    def test_backward_step_decode_graph(self, small_config):
        """Verify that gradients flow cleanly through autoregressive step() unrolling."""
        torch.manual_seed(501)
        layer = DGDALayer(small_config)
        layer.train()

        B, D = 2, layer.dim
        state = None
        conv_state = None
        loss = 0.0

        for _ in range(4):
            xt = torch.randn(B, 1, D, requires_grad=True)
            ot, state, conv_state = layer.step(xt, state=state, conv_state=conv_state)
            loss = loss + ot.sum()

        loss.backward()

        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Step param {name} grad is None"
                assert not torch.isnan(
                    param.grad
                ).any(), f"Step param {name} grad has NaNs"
                assert param.grad.norm().item() > 0.0, f"Step param {name} grad is zero"


# ==============================================================================
# Test 5: Causal Masking Isolation
# ==============================================================================


class TestCausalMaskingIsolation:
    """Validates that no future information leaks backwards into past representations."""

    @pytest.mark.parametrize("t_perturb", [0, 5, 10, 15, 20])
    def test_causal_output_perturbation_isolation(self, dgda_layer, t_perturb):
        """Perturbing token t must result in exactly 0.0 change on tokens < t."""
        torch.manual_seed(600 + t_perturb)
        B, L, D = 1, 32, dgda_layer.dim

        x1 = torch.randn(B, L, D)
        x2 = x1.clone()
        x2[:, t_perturb, :] += 10.0

        out1, _, _ = dgda_layer(x1, chunk_size=16)
        out2, _, _ = dgda_layer(x2, chunk_size=16)

        if t_perturb > 0:
            past_diff = (
                (out1[:, :t_perturb, :] - out2[:, :t_perturb, :]).abs().max().item()
            )
            assert past_diff == 0.0, (
                f"Causal violation: perturbing token {t_perturb} altered past outputs (< {t_perturb}) "
                f"by max diff {past_diff:.6e}"
            )

        future_diff = (
            (out1[:, t_perturb:, :] - out2[:, t_perturb:, :]).abs().max().item()
        )
        assert (
            future_diff > 1e-3
        ), f"Perturbation had no effect on future tokens: {future_diff}"

    @pytest.mark.parametrize("target_t", [0, 7, 15, 23])
    def test_causal_gradient_jacobian_isolation(self, small_config, target_t):
        """Computing loss on token target_t must produce exactly 0.0 gradient on future input tokens (> target_t)."""
        torch.manual_seed(700 + target_t)
        layer = DGDALayer(small_config)
        layer.train()

        B, L, D = 1, 32, layer.dim
        x = torch.randn(B, L, D, requires_grad=True)

        out, _, _ = layer(x, chunk_size=16)
        loss = out[:, target_t, :].sum()
        loss.backward()

        if target_t < L - 1:
            t_next = target_t + 1
            future_grad_max = x.grad[:, t_next:, :].abs().max().item()
            assert future_grad_max == 0.0, (
                f"Causal gradient leakage: loss at token {target_t} produced gradient "
                f"on future tokens (> {target_t}) with max magnitude {future_grad_max:.6e}"
            )

        target_grad = x.grad[:, target_t, :].abs().max().item()
        assert target_grad > 0.0, "Gradient at target token is zero"


# ==============================================================================
# Test 6: Numerical Stability under Mixed Precision (FP32, BF16, FP16)
# ==============================================================================


class TestNumericalStabilityMixedPrecision:
    """Validates execution across float32, bfloat16, and float16 without NaNs or Infs."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_precision_forward_backward_clean(self, small_config, dtype):
        """Verify forward and backward execution under FP32, BF16, and FP16 produces zero NaNs and zero Infs."""
        torch.manual_seed(800)
        layer = DGDALayer(small_config).to(dtype)
        layer.train()

        B, L, D = 2, 32, layer.dim
        x = torch.randn(B, L, D, dtype=dtype, requires_grad=True)

        out, final_state, _ = layer(x, chunk_size=16)

        assert not torch.isnan(out).any(), f"Forward output has NaNs in {dtype}"
        assert not torch.isinf(out).any(), f"Forward output has Infs in {dtype}"
        assert not torch.isnan(final_state).any(), f"Final state has NaNs in {dtype}"
        assert not torch.isinf(final_state).any(), f"Final state has Infs in {dtype}"

        loss = out.sum() + final_state.sum()
        loss.backward()

        assert not torch.isnan(x.grad).any(), f"Input x.grad has NaNs in {dtype}"
        assert not torch.isinf(x.grad).any(), f"Input x.grad has Infs in {dtype}"

        for name, param in layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(
                    param.grad
                ).any(), f"Param {name} grad has NaNs in {dtype}"
                assert not torch.isinf(
                    param.grad
                ).any(), f"Param {name} grad has Infs in {dtype}"

    def test_extreme_decay_gate_logits_stability(self, small_config):
        """Verify extreme decay logits (+/- 50.0) do not cause overflow or underflow NaNs."""
        torch.manual_seed(801)
        layer = DGDALayer(small_config)
        layer.eval()

        B, L, D = 2, 16, layer.dim
        x_extreme = torch.zeros(B, L, D)
        x_extreme[:, :8, :] = 50.0
        x_extreme[:, 8:, :] = -50.0

        out, state, _ = layer(x_extreme, chunk_size=16)

        assert not torch.isnan(out).any(), "Extreme decay logits caused NaNs in output"
        assert not torch.isinf(out).any(), "Extreme decay logits caused Infs in output"
        assert not torch.isnan(state).any(), "Extreme decay logits caused NaNs in state"


# ==============================================================================
# Test 7: O(1) Memory Invariance
# ==============================================================================


class TestMemoryInvariance:
    """Validates that recurrent state tensor shape and memory footprint are strictly invariant to L."""

    @pytest.mark.parametrize("seq_len", [1, 16, 64, 256, 1024])
    def test_recurrent_state_shape_and_memory_invariance(self, dgda_layer, seq_len):
        """Verify state shape is strictly (B, H, d_k, d_v) and bytes allocated do not change with seq_len."""
        torch.manual_seed(900)
        B, D = 2, dgda_layer.dim
        H, d_k, d_v = dgda_layer.n_heads, dgda_layer.d_head, dgda_layer.d_head

        x = torch.randn(B, seq_len, D)
        _, state, conv_state = dgda_layer(x, chunk_size=16)

        # 1. State shape check
        assert state.shape == (
            B,
            H,
            d_k,
            d_v,
        ), f"Recurrent state shape {state.shape} violates O(1) invariant (B, H, d_k, d_v)"

        # 2. Element count and memory check
        expected_elements = B * H * d_k * d_v
        expected_bytes = expected_elements * state.element_size()
        assert (
            state.nelement() == expected_elements
        ), "State element count grew with seq_len"
        assert (
            state.nelement() * state.element_size()
        ) == expected_bytes, "State memory grew"

        # 3. Conv state shape check
        assert conv_state.shape == (B, 3, D, dgda_layer.kernel_size - 1)
        for cs in conv_state:
            assert cs.shape == (B, D, dgda_layer.kernel_size - 1)

    def test_autoregressive_step_memory_leak_free(self, dgda_layer):
        """Verify running 20 consecutive step() calls maintains identical state object dimensions."""
        torch.manual_seed(901)
        B, D = 2, dgda_layer.dim
        state = None
        conv_state = None

        states_tracked = []
        for _ in range(20):
            xt = torch.randn(B, 1, D)
            _, state, conv_state = dgda_layer.step(
                xt, state=state, conv_state=conv_state
            )
            states_tracked.append(state.shape)

        assert all(
            s == states_tracked[0] for s in states_tracked
        ), "State shape mutated across steps"


# ==============================================================================
# Test 8: Boundary Conditions & Non-Divisible Sequence Length Handling
# ==============================================================================


class TestBoundaryConditions:
    """Validates boundary conditions: arbitrary lengths L, zero norm keys, uninitialized states."""

    @pytest.mark.parametrize(
        "seq_len", [0, 1, 5, 7, 15, 16, 17, 25, 31, 32, 37, 48, 65, 70, 128]
    )
    def test_arbitrary_sequence_lengths(self, dgda_layer, seq_len):
        """Verify arbitrary sequence lengths execute cleanly without shape mismatches or index errors."""
        torch.manual_seed(1000 + seq_len)
        B, D = 2, dgda_layer.dim
        x = torch.randn(B, seq_len, D)

        out, state, conv_state = dgda_layer(x, chunk_size=16)

        assert out.shape == (
            B,
            seq_len,
            D,
        ), f"Output shape {out.shape} does not match (B, {seq_len}, D)"
        assert not torch.isnan(out).any(), f"Sequence length {seq_len} produced NaNs"
        assert state.shape == (
            B,
            dgda_layer.n_heads,
            dgda_layer.d_head,
            dgda_layer.d_head,
        )
        assert conv_state.shape == (B, 3, D, dgda_layer.kernel_size - 1)

    def test_zero_key_norm_stability(self, dgda_layer):
        """Verify all-zero input projection resulting in ||k_t|| = 0 is stabilized by epsilon without NaNs."""
        B, L, D = 2, 16, dgda_layer.dim
        x_zero = torch.zeros(B, L, D)

        out, state, _ = dgda_layer(x_zero, chunk_size=16)

        assert not torch.isnan(out).any(), "Zero key norm resulted in NaNs"
        assert not torch.isinf(out).any(), "Zero key norm resulted in Infs"

    def test_uninitialized_state_defaults(self, dgda_layer):
        """Verify passing state=None and conv_state=None initializes cleanly to zeros."""
        B, L, D = 2, 16, dgda_layer.dim
        x = torch.randn(B, L, D)

        out, state, conv_state = dgda_layer(x, state=None, conv_state=None)

        assert state is not None
        assert not torch.isnan(state).any()
        assert conv_state is not None

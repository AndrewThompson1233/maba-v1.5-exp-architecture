"""Empirical Challenger Verification & Stress-Test Suite for DGDALayer.

Conducted by: teamwork_preview_challenger_m1_2
Target: maba_sparse.layers.dgda.DGDALayer

Verifies:
1. Causal Masking Isolation:
   - Output perturbation: Perturbing token t produces strictly 0.0 difference on tokens < t.
   - Autograd Jacobian: d(output_{<t}) / d(input_t) == 0.0 identically.
2. Strict O(1) Memory Footprint:
   - Measures recurrent state tensor byte size across sequence lengths L = 16, 64, 256, 1024, 4096.
   - Asserts memory is strictly invariant (zero byte variance).
3. Step Decode Loop vs Forward Pass:
   - Asserts max output and state discrepancy < 1e-5 across multiple sequence lengths and initial conditions.
"""

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer


class TestChallengerCausalIsolation:
    """Stress tests strict causal isolation in forward pass and backward autograd."""

    @pytest.mark.parametrize("chunk_size", [16, 32])
    @pytest.mark.parametrize("L", [32, 64])
    def test_forward_output_perturbation_isolation(self, chunk_size: int, L: int):
        """Verify that perturbing token t has strictly 0.0 impact on tokens < t."""
        torch.manual_seed(42 + L + chunk_size)
        config = MabaSparseConfig(dim=128, n_heads=4, d_head=32)
        layer = DGDALayer(config)
        layer.eval()

        B = 2
        x = torch.randn(B, L, config.dim)

        with torch.no_grad():
            base_out, _, _ = layer(x, chunk_size=chunk_size)

        # Test perturbation across multiple critical token locations
        perturb_indices = [0, 1, chunk_size - 1, chunk_size, chunk_size + 1, L - 2, L - 1]
        perturb_indices = sorted(list(set([idx for idx in perturb_indices if 0 <= idx < L])))

        for t in perturb_indices:
            x_pert = x.clone()
            x_pert[:, t, :] += torch.randn(B, config.dim) * 5.0 + 10.0

            with torch.no_grad():
                pert_out, _, _ = layer(x_pert, chunk_size=chunk_size)

            if t > 0:
                past_diff = (base_out[:, :t, :] - pert_out[:, :t, :]).abs().max().item()
                assert past_diff <= 1e-7, (
                    f"Causal violation at L={L}, chunk_size={chunk_size}, t={t}: "
                    f"tokens < {t} differed by max {past_diff:.6e}"
                )

            # Future tokens (>= t) must reflect the perturbation
            future_diff = (base_out[:, t:, :] - pert_out[:, t:, :]).abs().max().item()
            assert future_diff > 1e-4, f"Perturbation at t={t} had no effect on future tokens"

    def test_autograd_jacobian_causal_triangularity(self):
        """Verify full autograd Jacobian d(output_{<t}) / d(input_t) == 0.0 identically."""
        torch.manual_seed(1337)
        config = MabaSparseConfig(dim=64, n_heads=2, d_head=32)
        layer = DGDALayer(config)
        layer.eval()

        for L in [16, 32]:
            x = torch.randn(1, L, config.dim)

            def fwd_fn(inp: torch.Tensor) -> torch.Tensor:
                out, _, _ = layer(inp, chunk_size=16)
                return out

            # Jacobian J shape: [1, L, D, 1, L, D]
            # J[0, s, :, 0, t, :] = d(output_s) / d(input_t)
            J = torch.autograd.functional.jacobian(fwd_fn, x)

            max_past_grad = 0.0
            violation_count = 0

            for t in range(L):
                for s in range(t):
                    d_s_t = J[0, s, :, 0, t, :]
                    val = d_s_t.abs().max().item()
                    if val > 0.0:
                        violation_count += 1
                        if val > max_past_grad:
                            max_past_grad = val

            assert violation_count == 0 and max_past_grad == 0.0, (
                f"Jacobian causal violation for L={L}: {violation_count} non-zero entries "
                f"for s < t, max magnitude {max_past_grad:.6e}"
            )

    def test_autograd_slice_gradient_isolation_long_sequence(self):
        """Verify autograd loss on tokens < t produces exactly 0.0 gradient on tokens >= t."""
        torch.manual_seed(2026)
        config = MabaSparseConfig(dim=128, n_heads=4, d_head=32)
        layer = DGDALayer(config)
        layer.train()

        L = 64
        x = torch.randn(1, L, config.dim, requires_grad=True)
        out, _, _ = layer(x, chunk_size=16)

        # Loss computed strictly on tokens 0..15 (chunk 0)
        target_t = 16
        loss = out[:, :target_t, :].sum()
        loss.backward()

        assert x.grad is not None
        future_grad_max = x.grad[:, target_t:, :].abs().max().item()
        assert future_grad_max == 0.0, (
            f"Gradient leaked into future tokens [16:64]: max grad = {future_grad_max:.6e}"
        )
        past_grad_max = x.grad[:, :target_t, :].abs().max().item()
        assert past_grad_max > 0.0, "Past tokens should have non-zero gradient"


class TestChallengerMemoryInvariance:
    """Stress tests strict O(1) recurrent memory footprint invariance."""

    def test_o1_state_byte_size_invariance(self):
        """Measure recurrent state tensor byte size across L = 16, 64, 256, 1024, 4096."""
        config = MabaSparseConfig()  # 101M MABA-Tiny: dim=640, n_heads=10, d_head=64
        layer = DGDALayer(config)
        layer.eval()

        test_lengths = [16, 64, 256, 1024, 4096]
        measured_bytes: List[int] = []
        measured_conv_bytes: List[int] = []
        measured_shapes = []

        for L in test_lengths:
            torch.manual_seed(42 + L)
            x = torch.randn(1, L, config.dim)

            with torch.no_grad():
                _, state, conv_state = layer(x, chunk_size=16)

            # Recurrent state bytes
            sb = state.nelement() * state.element_size()
            storage_bytes = state.untyped_storage().nbytes()
            assert sb == storage_bytes, f"Storage mismatch: {sb} != {storage_bytes}"

            measured_bytes.append(sb)
            measured_shapes.append(tuple(state.shape))

            # Conv state bytes
            cb = sum(c.nelement() * c.element_size() for c in conv_state)
            measured_conv_bytes.append(cb)

        # Invariance assertions
        unique_bytes = set(measured_bytes)
        unique_shapes = set(measured_shapes)
        unique_conv_bytes = set(measured_conv_bytes)

        expected_elements = 1 * config.n_heads * config.d_head * config.d_head
        expected_bytes = expected_elements * 4  # float32 = 4 bytes

        assert len(unique_bytes) == 1, f"State byte size varied across lengths: {measured_bytes}"
        assert measured_bytes[0] == expected_bytes, (
            f"Expected {expected_bytes} bytes (1, {config.n_heads}, {config.d_head}, {config.d_head}), "
            f"got {measured_bytes[0]}"
        )
        assert len(unique_shapes) == 1, f"State shape varied: {measured_shapes}"
        assert len(unique_conv_bytes) == 1, f"Conv state byte size varied: {measured_conv_bytes}"

    def test_o1_step_memory_stability_50_steps(self):
        """Verify 50 unrolled decode steps maintain identical state byte size without leak."""
        config = MabaSparseConfig(dim=128, n_heads=4, d_head=32)
        layer = DGDALayer(config)
        layer.eval()

        torch.manual_seed(99)
        state = None
        conv_state = None
        state_sizes = []

        for _ in range(50):
            xt = torch.randn(1, 1, config.dim)
            with torch.no_grad():
                _, state, conv_state = layer.step(xt, state=state, conv_state=conv_state)
            state_sizes.append(state.nelement() * state.element_size())

        assert len(set(state_sizes)) == 1, "State size mutated during decode steps"


class TestChallengerStepVsForwardEquivalence:
    """Stress tests step() decode loop vs forward() pass (< 1e-5 discrepancy)."""

    @pytest.mark.parametrize("L", [16, 32, 48, 64, 128])
    def test_step_loop_vs_forward_standard_input(self, L: int):
        """Verify max discrepancy < 1e-5 on sequence lengths up to 128."""
        torch.manual_seed(1000 + L)
        config = MabaSparseConfig(dim=256, n_heads=4, d_head=64)
        layer = DGDALayer(config)
        layer.eval()

        B = 2
        x = torch.randn(B, L, config.dim)

        with torch.no_grad():
            out_fwd, state_fwd, _ = layer(x, chunk_size=16)

            # Unroll step()
            state = None
            conv_state = None
            step_outs = []
            for t in range(L):
                xt = x[:, t : t + 1, :]
                ot, state, conv_state = layer.step(xt, state=state, conv_state=conv_state)
                step_outs.append(ot)

            out_step = torch.cat(step_outs, dim=1)

            diff_out = (out_fwd - out_step).abs().max().item()
            diff_state = (state_fwd - state).abs().max().item()

        assert diff_out < 1e-5, f"Output discrepancy {diff_out:.6e} >= 1e-5 at L={L}"
        assert diff_state < 1e-5, f"State discrepancy {diff_state:.6e} >= 1e-5 at L={L}"

    @pytest.mark.parametrize("L", [7, 15, 17, 33, 50])
    def test_step_loop_vs_forward_non_divisible_lengths(self, L: int):
        """Verify max discrepancy < 1e-5 for sequence lengths non-divisible by chunk_size=16."""
        torch.manual_seed(2000 + L)
        config = MabaSparseConfig(dim=256, n_heads=4, d_head=64)
        layer = DGDALayer(config)
        layer.eval()

        B = 2
        x = torch.randn(B, L, config.dim)

        with torch.no_grad():
            out_fwd, state_fwd, _ = layer(x, chunk_size=16)

            state = None
            conv_state = None
            step_outs = []
            for t in range(L):
                xt = x[:, t : t + 1, :]
                ot, state, conv_state = layer.step(xt, state=state, conv_state=conv_state)
                step_outs.append(ot)

            out_step = torch.cat(step_outs, dim=1)

            diff_out = (out_fwd - out_step).abs().max().item()
            diff_state = (state_fwd - state).abs().max().item()

        assert diff_out < 1e-5, f"Non-divisible output diff {diff_out:.6e} >= 1e-5 at L={L}"
        assert diff_state < 1e-5, f"Non-divisible state diff {diff_state:.6e} >= 1e-5 at L={L}"

    def test_step_loop_vs_forward_with_prior_state(self):
        """Verify max discrepancy < 1e-5 when starting from non-zero initial recurrent state."""
        torch.manual_seed(3000)
        config = MabaSparseConfig(dim=256, n_heads=4, d_head=64)
        layer = DGDALayer(config)
        layer.eval()

        B, L = 2, 48
        initial_state = torch.randn(B, config.n_heads, config.d_head, config.d_head) * 0.5
        x = torch.randn(B, L, config.dim)

        with torch.no_grad():
            out_fwd, state_fwd, _ = layer(x, state=initial_state, chunk_size=16)

            state = initial_state.clone()
            conv_state = None
            step_outs = []
            for t in range(L):
                xt = x[:, t : t + 1, :]
                ot, state, conv_state = layer.step(xt, state=state, conv_state=conv_state)
                step_outs.append(ot)

            out_step = torch.cat(step_outs, dim=1)

            diff_out = (out_fwd - out_step).abs().max().item()
            diff_state = (state_fwd - state).abs().max().item()

        assert diff_out < 1e-5, f"Output discrepancy with prior state {diff_out:.6e} >= 1e-5"
        assert diff_state < 1e-5, f"State discrepancy with prior state {diff_state:.6e} >= 1e-5"


def run_standalone_measurements():
    """Run empirical benchmark and print table of results."""
    print("=" * 80)
    print("EMPIRICAL CHALLENGER STRESS HARNESS EXECUTION")
    print("=" * 80)

    # 1. Causal Masking Isolation
    print("\n--- 1. Causal Masking Isolation Verification ---")
    torch.manual_seed(42)
    config_med = MabaSparseConfig(dim=128, n_heads=4, d_head=32)
    layer_med = DGDALayer(config_med)
    layer_med.eval()

    L = 64
    x = torch.randn(2, L, config_med.dim)
    with torch.no_grad():
        base_out, _, _ = layer_med(x, chunk_size=16)

    max_causal_past_diff = 0.0
    for t in [1, 15, 16, 31, 32, 47, 48, 63]:
        x_p = x.clone()
        x_p[:, t, :] += 50.0
        with torch.no_grad():
            p_out, _, _ = layer_med(x_p, chunk_size=16)
        past_diff = (base_out[:, :t, :] - p_out[:, :t, :]).abs().max().item()
        if past_diff > max_causal_past_diff:
            max_causal_past_diff = past_diff
        print(f"Token t={t:2d} perturbed (+50.0) -> max diff on tokens < {t:2d}: {past_diff:.10f}")

    print(f"Overall Forward Output Perturbation Max Past Leakage: {max_causal_past_diff:.10f} (Expected: 0.0)")
    assert max_causal_past_diff == 0.0, "Forward causal isolation failed!"

    # Jacobian check
    config_small = MabaSparseConfig(dim=64, n_heads=2, d_head=32)
    layer_small = DGDALayer(config_small)
    layer_small.eval()

    L_jac = 32
    x_jac = torch.randn(1, L_jac, config_small.dim)
    J = torch.autograd.functional.jacobian(lambda inp: layer_small(inp, chunk_size=16)[0], x_jac)
    max_jac_past = 0.0
    for t in range(L_jac):
        for s in range(t):
            val = J[0, s, :, 0, t, :].abs().max().item()
            if val > max_jac_past:
                max_jac_past = val

    print(f"Full Autograd Jacobian d(output_<t)/d(input_t) Max Value: {max_jac_past:.10f} (Expected: 0.0)")
    assert max_jac_past == 0.0, "Autograd Jacobian causal isolation failed!"

    # 2. Strict O(1) Memory Footprint
    print("\n--- 2. Strict O(1) Memory Footprint Invariance Verification ---")
    config_101m = MabaSparseConfig()  # 101M MABA-Tiny: dim=640, n_heads=10, d_head=64
    layer_101m = DGDALayer(config_101m)
    layer_101m.eval()

    lengths = [16, 64, 256, 1024, 4096]
    memory_table: Dict[int, Dict[str, int]] = {}

    for seq_len in lengths:
        torch.manual_seed(100 + seq_len)
        x_mem = torch.randn(1, seq_len, config_101m.dim)
        with torch.no_grad():
            _, state, conv_state = layer_101m(x_mem, chunk_size=16)

        state_bytes = state.nelement() * state.element_size()
        conv_bytes = sum(c.nelement() * c.element_size() for c in conv_state)
        total_recurrent_bytes = state_bytes + conv_bytes

        memory_table[seq_len] = {
            "state_bytes": state_bytes,
            "conv_bytes": conv_bytes,
            "total_bytes": total_recurrent_bytes,
        }
        print(
            f"L={seq_len:5d}: State Bytes = {state_bytes:7d} ({state_bytes/1024:.2f} KB), "
            f"Conv Bytes = {conv_bytes:6d} ({conv_bytes/1024:.2f} KB), "
            f"Total = {total_recurrent_bytes:7d} ({total_recurrent_bytes/1024:.2f} KB)"
        )

    all_state_bytes = [v["state_bytes"] for v in memory_table.values()]
    all_conv_bytes = [v["conv_bytes"] for v in memory_table.values()]
    assert len(set(all_state_bytes)) == 1, "Recurrent state memory varied across sequence lengths!"
    assert len(set(all_conv_bytes)) == 1, "Conv state memory varied across sequence lengths!"
    print(f"Memory Invariance Confirmed: Exactly {all_state_bytes[0]} bytes across all sequence lengths.")

    # 3. Step Decode Loop vs Forward Pass
    print("\n--- 3. Step Decode Loop vs Forward Pass Equivalence Verification ---")
    config_eval = MabaSparseConfig(dim=256, n_heads=4, d_head=64)
    layer_eval = DGDALayer(config_eval)
    layer_eval.eval()

    eval_lengths = [16, 32, 48, 64, 128]
    for seq_len in eval_lengths:
        torch.manual_seed(500 + seq_len)
        x_eval = torch.randn(2, seq_len, config_eval.dim)

        with torch.no_grad():
            out_fwd, state_fwd, _ = layer_eval(x_eval, chunk_size=16)

            state = None
            conv_state = None
            step_outs = []
            for t in range(seq_len):
                xt = x_eval[:, t : t + 1, :]
                ot, state, conv_state = layer_eval.step(xt, state=state, conv_state=conv_state)
                step_outs.append(ot)

            out_step = torch.cat(step_outs, dim=1)

            diff_out = (out_fwd - out_step).abs().max().item()
            diff_state = (state_fwd - state).abs().max().item()

        print(
            f"L={seq_len:3d}: Max Output Discrepancy = {diff_out:.6e} (Threshold: < 1e-5) | "
            f"Max State Discrepancy = {diff_state:.6e}"
        )
        assert diff_out < 1e-5, f"Discrepancy {diff_out:.6e} exceeded 1e-5 threshold at L={seq_len}"
        assert diff_state < 1e-5, f"State discrepancy {diff_state:.6e} exceeded 1e-5 threshold at L={seq_len}"

    print("\nAll empirical assertions passed successfully with zero violations.")


if __name__ == "__main__":
    run_standalone_measurements()

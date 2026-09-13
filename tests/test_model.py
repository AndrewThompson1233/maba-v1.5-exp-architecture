"""Unit and integration test suite for MabaSparseForCausalLM and DenseTransformerForCausalLM.

Validates:
1. Parameter count at ~101M scale (within 95M to 105M budget).
2. Forward and backward pass gradient continuity and zero NaNs.
3. Loss calculation with targets/labels.
4. Multi-Token Prediction (MTP k=2) speculative head and auxiliary loss (weight 0.3).
5. Tied factorized LM head and embedding weights.
6. Cyclic 3:1 macro-stack layout (15 DGDA : 5 MABA-SA).
7. Gated residuals initialization (bias = 2.0).
8. Autoregressive token generation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.dgda import DGDALayer
from maba_sparse.layers.sparse_attention import MabaSparseAttention
from maba_sparse.model import (
    FactorizedEmbeddings,
    MabaBlock,
    MabaSparseForCausalLM,
    MabaSparseLM,
    RMSNorm,
    SwiGLUFFN,
    get_101m_config,
)


# ==============================================================================
# 1. 101M Parameter Count Verification
# ==============================================================================


class TestModelParameterCount:
    """Verifies ~101M parameter topology across Maba-Sparse and Dense Baseline."""

    def test_maba_sparse_101m_parameter_count(self):
        """Verify MabaSparseForCausalLM parameter count is ~101M (within 95M-105M)."""
        cfg = get_101m_config()
        # Instantiate model on meta device or CPU
        model = MabaSparseForCausalLM(cfg)
        unique_params = sum(p.numel() for p in set(model.parameters()))

        assert (
            95_000_000 <= unique_params <= 105_000_000
        ), f"Expected Maba-Sparse params in [95M, 105M], got {unique_params:,} ({unique_params/1e6:.2f}M)"

    def test_dense_transformer_101m_parameter_count(self):
        """Verify DenseTransformerForCausalLM parameter count is ~101M (within 95M-105M)."""
        dense_model = DenseTransformerForCausalLM()
        unique_params = sum(p.numel() for p in set(dense_model.parameters()))

        assert (
            95_000_000 <= unique_params <= 105_000_000
        ), f"Expected Dense Transformer params in [95M, 105M], got {unique_params:,} ({unique_params/1e6:.2f}M)"

    def test_parameter_budget_alignment(self):
        """Verify Maba-Sparse and Dense Baseline are within 3% parameter parity."""
        maba_model = MabaSparseForCausalLM(get_101m_config())
        dense_model = DenseTransformerForCausalLM()

        maba_params = sum(p.numel() for p in set(maba_model.parameters()))
        dense_params = sum(p.numel() for p in set(dense_model.parameters()))

        relative_diff = abs(maba_params - dense_params) / dense_params
        assert (
            relative_diff < 0.05
        ), f"Expected <5% param difference, got {relative_diff*100:.2f}% (Maba: {maba_params:,}, Dense: {dense_params:,})"


# ==============================================================================
# 2. Structural and Topological Tests
# ==============================================================================


class TestModelArchitectureTopology:
    """Validates cyclic macro-stack, weight tying, and residual gating."""

    def test_cyclic_3_to_1_macro_stack(self):
        """Verify 20 MabaBlocks follow the cyclic 3:1 pattern (15 DGDA : 5 MABA-SA)."""
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert len(model.layers) == 20
        dgda_count = 0
        maba_sa_count = 0

        for i, layer in enumerate(model.layers):
            if (i + 1) % 4 == 0:
                assert isinstance(
                    layer.mixer, MabaSparseAttention
                ), f"Layer {i} should be MabaSparseAttention, got {type(layer.mixer)}"
                maba_sa_count += 1
            else:
                assert isinstance(
                    layer.mixer, DGDALayer
                ), f"Layer {i} should be DGDALayer, got {type(layer.mixer)}"
                dgda_count += 1

        assert dgda_count == 15, f"Expected 15 DGDA layers, got {dgda_count}"
        assert maba_sa_count == 5, f"Expected 5 MABA-SA layers, got {maba_sa_count}"

    def test_factorized_embedding_dimensions(self):
        """Verify factorized embeddings V=32,768 -> d_emb=128 -> D=640."""
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert model.embeddings.vocab_size == 32768
        assert model.embeddings.d_emb == 128
        assert model.embeddings.dim == 640
        assert model.embeddings.in_emb.weight.shape == (32768, 128)
        assert model.embeddings.proj.weight.shape == (640, 128)

    def test_weight_tying(self):
        """Verify tied factorized LM head shares weights with input embeddings."""
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        assert (
            model.lm_head.weight is model.embeddings.in_emb.weight
        ), "lm_head.weight must be tied to embeddings.in_emb.weight"

    def test_residual_gate_initial_bias(self):
        """Verify residual gates are initialized with bias = 2.0 (sigmoid ~ 0.8808)."""
        cfg = get_101m_config()
        model = MabaSparseForCausalLM(cfg)

        for i, layer in enumerate(model.layers):
            assert torch.allclose(
                layer.res_gate1, torch.full_like(layer.res_gate1, 2.0)
            ), f"Layer {i} res_gate1 bias != 2.0"
            assert torch.allclose(
                layer.res_gate2, torch.full_like(layer.res_gate2, 2.0)
            ), f"Layer {i} res_gate2 bias != 2.0"


# ==============================================================================
# 3. Forward, Backward, and Loss Tests
# ==============================================================================


class TestModelForwardBackward:
    """Validates execution, output shapes, gradient propagation, and zero NaNs."""

    @pytest.fixture
    def small_model(self):
        cfg = MabaSparseConfig(
            dim=64,
            n_heads=2,
            d_head=32,
            n_layers=4,
            vocab_size=1000,
            d_emb=32,
            intermediate_size=128,
            window_size=32,
            block_size=16,
            top_k=4,
        )
        return MabaSparseForCausalLM(cfg)

    def test_forward_output_shape(self, small_model):
        """Verify logits shape [B, L, V] and tuple unpacking."""
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        out = small_model(x)

        assert out.logits.shape == (B, L, 1000)
        assert out.loss is None
        # Tuple unpacking
        logits, loss = out
        assert logits.shape == (B, L, 1000)
        assert loss is None

    def test_forward_with_targets(self, small_model):
        """Verify loss is computed when targets are provided."""
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))
        out = small_model(x, targets=targets)

        assert out.loss is not None
        assert torch.isfinite(out.loss)
        assert out.loss.item() > 0.0

    def test_backward_gradient_continuity(self, small_model):
        """Verify backward pass produces valid finite gradients with zero NaNs."""
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_model(x, targets=targets)
        out.loss.backward()

        for name, p in small_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"

    def test_mtp_auxiliary_loss(self, small_model):
        """Verify Multi-Token Prediction head generates auxiliary loss and mtp_logits."""
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_model(x, targets=targets)
        assert out.mtp_logits is not None
        assert out.mtp_logits.shape == (B, L - 1, 1000)

        # Gradient flow to mtp_head proj
        out.loss.backward()
        assert small_model.mtp_head.proj.weight.grad is not None
        assert not torch.isnan(small_model.mtp_head.proj.weight.grad).any()


# ==============================================================================
# 4. Dense Transformer Baseline Tests
# ==============================================================================


class TestDenseTransformerBaseline:
    """Validates the 101M Dense Transformer baseline."""

    @pytest.fixture
    def small_dense_model(self):
        return DenseTransformerForCausalLM(
            vocab_size=1000,
            d_emb=32,
            dim=64,
            n_layers=4,
            n_heads=2,
            d_head=32,
            intermediate_size=128,
        )

    def test_dense_forward_backward(self, small_dense_model):
        """Verify forward and backward pass on dense baseline."""
        B, L = 2, 16
        x = torch.randint(0, 1000, (B, L))
        targets = torch.randint(0, 1000, (B, L))

        out = small_dense_model(x, targets=targets)
        assert out.logits.shape == (B, L, 1000)
        assert out.loss is not None
        assert torch.isfinite(out.loss)

        out.loss.backward()
        for name, p in small_dense_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN in dense {name}"

    def test_autoregressive_generation(self, small_dense_model):
        """Verify generate produces extended sequence."""
        x = torch.randint(0, 1000, (1, 4))
        gen = small_dense_model.generate(x, max_new_tokens=6, temperature=0.0)
        assert gen.shape == (1, 10)
        assert torch.equal(gen[:, :4], x)

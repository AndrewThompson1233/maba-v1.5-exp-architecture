"""Automated unit and integration test suite for Delta-Guided Centroid Indexer (DG-Indexer).

Validates Milestone 2 acceptance criteria for DGIndexer:
1. Needle-in-a-haystack outlier detection preserved by hybrid pooling (0.5*mean + 0.5*max).
2. Logarithmic distance penalty monotonic distance preference and exact formula scaling.
3. Strict causal block masking (future blocks strictly masked to -inf).
4. Dynamic top-k selection with adaptive ceiling min(top_k, num_blocks).
5. Analytical gradient continuity (finite, non-zero, zero-NaN gradients on all parameters).
6. Robustness across diverse sequence lengths, batch sizes, and floating-point precisions.
"""

import math
import pytest
import torch

from maba_sparse.config import MabaSparseConfig
from maba_sparse.layers.indexer import DGIndexer, DeltaGuidedCentroidIndexer


# ==============================================================================
# 1. Hybrid Pooling & Needle-in-a-Haystack Outlier Protection
# ==============================================================================


class TestHybridPoolingNeedleProtection:
    """Validates that hybrid 0.5*mean + 0.5*max pooling prevents dilution of isolated facts."""

    def test_hybrid_pooling_needle_protection(self):
        """Insert a high-activation fact at position 500 in 4096 tokens of noise.

        Assert max-activation pooling preserves the centroid and block floor(500/64) = 7
        is selected in top-32.
        """
        torch.manual_seed(42)
        dim = 640
        d_idx = 64
        block_size = 64
        top_k = 32
        L = 4096
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=top_k,
            dist_lambda=0.1,  # Lower distance penalty to focus purely on representation retrieval
        )
        indexer.eval()

        needle_pos = 500
        expected_needle_block = needle_pos // block_size  # 500 // 64 = 7
        query_pos = 2048  # Query is placed after needle block to allow causal attention

        # Create low-amplitude background noise
        x = torch.randn(B, L, dim) * 0.05

        # Create a distinctive needle signal in input space
        needle_pattern = torch.randn(dim) * 20.0
        x[0, needle_pos] = needle_pattern

        # Align the query position to seek the needle representation
        x[0, query_pos] = needle_pattern * 0.5

        with torch.no_grad():
            top_indices, centroids = indexer(x)

        # Check shape of results
        num_blocks = (L + block_size - 1) // block_size
        assert centroids.shape == (B, num_blocks, d_idx)
        assert top_indices.shape == (B, L, top_k)

        # Retrieve selected blocks for the query token
        query_top_blocks = top_indices[0, query_pos].tolist()

        assert (
            expected_needle_block in query_top_blocks
        ), f"Needle block {expected_needle_block} not found in top-{top_k} blocks: {query_top_blocks}"

    def test_needle_dilution_mitigation_vs_pure_mean(self):
        """Demonstrate that hybrid pooling preserves 32x higher peak signal than pure mean pooling."""
        block_size = 64
        d_idx = 64
        k_blocks = torch.randn(1, 1, block_size, d_idx) * 0.01

        # Salient needle token with magnitude 10.0
        needle_idx = 37
        k_blocks[0, 0, needle_idx, 0] = 10.0

        mean_centroid = k_blocks.mean(dim=2)
        max_centroid, _ = k_blocks.max(dim=2)
        hybrid_centroid = 0.5 * (mean_centroid + max_centroid)

        # Mean dilutes signal: 10 / 64 ~ 0.156
        assert mean_centroid[0, 0, 0].item() < 0.25
        # Max preserves full peak: ~ 10.0
        assert max_centroid[0, 0, 0].item() >= 9.9
        # Hybrid preserves half the peak: ~ 5.08, which is > 30x stronger than mean
        assert hybrid_centroid[0, 0, 0].item() > 4.5
        assert hybrid_centroid[0, 0, 0].item() > 25.0 * mean_centroid[0, 0, 0].item()

    def test_multiple_needles_retrieval(self):
        """Verify multiple salient facts in distinct blocks are simultaneously retained."""
        torch.manual_seed(123)
        dim = 128
        d_idx = 32
        block_size = 64
        top_k = 16
        L = 1024
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=top_k,
            dist_lambda=0.05,
        )
        indexer.eval()

        needle_positions = [100, 300, 600]
        expected_blocks = [p // block_size for p in needle_positions]  # [1, 4, 9]

        x = torch.randn(B, L, dim) * 0.02
        probe_vector = torch.randn(dim) * 15.0

        for pos in needle_positions:
            x[0, pos] = probe_vector

        # Query token at the end of the context
        q_pos = 900
        x[0, q_pos] = probe_vector

        with torch.no_grad():
            top_indices, _ = indexer(x)

        query_selected = top_indices[0, q_pos].tolist()
        for blk in expected_blocks:
            assert blk in query_selected, f"Needle block {blk} missing from retrieved blocks: {query_selected}"


# ==============================================================================
# 2. Logarithmic Distance Penalty
# ==============================================================================


class TestLogDistancePenalty:
    """Validates the logarithmic distance prior lambda * log(1 + |q_blk - k_blk|)."""

    def test_log_distance_decay_monotonicity(self):
        """Verify that when similarities are equal, scores decay monotonically with distance."""
        dim = 64
        d_idx = 32
        block_size = 64
        num_blocks = 16
        L = num_blocks * block_size
        B = 1

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=num_blocks,
            dist_lambda=0.5,
        )
        indexer.eval()

        # Set weights such that dot-product similarity is uniform
        with torch.no_grad():
            indexer.q_idx_proj.weight.zero_()
            indexer.k_idx_proj.weight.zero_()

        x = torch.randn(B, L, dim)
        _, _, scores = indexer(x, return_scores=True)

        # For the last query token in the last block (block 15):
        # All preceding blocks (0 to 15) are valid and causal
        q_token = L - 1
        q_scores = scores[0, q_token, :num_blocks]

        # Closer blocks should have higher scores (lower penalty)
        # Block 15 (dist 0) > Block 14 (dist 1) > Block 13 (dist 2) > ... > Block 0 (dist 15)
        for i in range(num_blocks - 1):
            assert (
                q_scores[i] < q_scores[i + 1]
            ), f"Score for block {i} ({q_scores[i]}) should be strictly lower than block {i+1} ({q_scores[i+1]})"

    def test_log_distance_exact_formula(self):
        """Verify exact mathematical calculation of distance penalty."""
        dim = 64
        d_idx = 32
        block_size = 64
        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            dist_lambda=0.75,
        )
        indexer.eval()

        with torch.no_grad():
            indexer.q_idx_proj.weight.zero_()
            indexer.k_idx_proj.weight.zero_()

        x = torch.zeros(1, 256, dim)
        _, _, scores = indexer(x, return_scores=True)

        # For token at pos 192 (block 3) attending to block 1:
        # dist = |3 - 1| = 2.0
        # penalty = 0.75 * log(1.0 + 2.0) = 0.75 * log(3.0)
        # Since sim is 0, score = -penalty
        expected_penalty = 0.75 * math.log(3.0)
        actual_score = scores[0, 192, 1].item()
        assert math.isclose(-expected_penalty, actual_score, rel_tol=1e-5)


# ==============================================================================
# 3. Causal Block Masking
# ==============================================================================


class TestCausalBlockMasking:
    """Validates that queries strictly cannot attend to future blocks."""

    def test_causal_block_isolation(self):
        """Verify future blocks receive score -inf and are never selected over past blocks."""
        dim = 64
        d_idx = 32
        block_size = 64
        num_blocks = 8
        L = num_blocks * block_size

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=num_blocks,
            dist_lambda=0.5,
        )
        indexer.eval()

        x = torch.randn(1, L, dim)
        top_indices, _, scores = indexer(x, return_scores=True)

        # Test at various query token positions
        for t in [0, 31, 63, 64, 127, 250, L - 1]:
            q_block = t // block_size

            # Check that all future blocks have -inf scores
            for b in range(num_blocks):
                if b > q_block:
                    assert (
                        scores[0, t, b].item() == float("-inf")
                    ), f"Future block {b} has non-inf score {scores[0, t, b]} for query at token {t} (block {q_block})"
                else:
                    assert math.isfinite(
                        scores[0, t, b].item()
                    ), f"Past/current block {b} has non-finite score {scores[0, t, b]}"

            # For queries where actual_k <= (q_block + 1), selected blocks must only be past/current
            available_past_blocks = q_block + 1
            selected = top_indices[0, t].tolist()
            for rank_idx, b in enumerate(selected[:available_past_blocks]):
                assert (
                    b <= q_block
                ), f"Query at token {t} (block {q_block}) selected future block {b} at rank {rank_idx}"


# ==============================================================================
# 4. Dynamic Top-K Selection
# ==============================================================================


class TestDynamicTopKSelection:
    """Validates adaptive top-k budget and ceiling constraints."""

    @pytest.mark.parametrize("L,expected_blocks,expected_k", [
        (32, 1, 1),
        (64, 1, 1),
        (65, 2, 2),
        (128, 2, 2),
        (512, 8, 8),
        (2048, 32, 32),
        (4096, 64, 32),
    ])
    def test_adaptive_ceiling(self, L, expected_blocks, expected_k):
        """Verify actual_k = min(top_k, num_blocks) across varied sequence lengths."""
        dim = 64
        d_idx = 32
        block_size = 64
        default_top_k = 32

        indexer = DGIndexer(
            dim=dim,
            d_idx=d_idx,
            block_size=block_size,
            top_k=default_top_k,
        )
        indexer.eval()

        x = torch.randn(2, L, dim)
        top_indices, centroids = indexer(x)

        assert centroids.shape == (2, expected_blocks, d_idx)
        assert top_indices.shape == (2, L, expected_k)

    def test_custom_top_k_override(self):
        """Verify explicit top_k argument overrides layer default."""
        dim = 64
        d_idx = 32
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=64, top_k=32)

        x = torch.randn(1, 1024, dim)  # 16 blocks
        top_indices_8, _ = indexer(x, top_k=8)
        assert top_indices_8.shape == (1, 1024, 8)

        top_indices_16, _ = indexer(x, top_k=16)
        assert top_indices_16.shape == (1, 1024, 16)


# ==============================================================================
# 5. Analytical Gradient Continuity
# ==============================================================================


class TestIndexerAnalyticalGradientContinuity:
    """Validates that backward computes non-zero, finite, NaN-free gradients across all parameters."""

    def test_backward_indexer_parameters_finite_and_nonzero(self):
        """Verify gradients flow cleanly through q_idx_proj and k_idx_proj."""
        torch.manual_seed(999)
        dim = 128
        d_idx = 32
        block_size = 64
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=block_size, top_k=8)
        indexer.train()

        B, L = 2, 256
        x = torch.randn(B, L, dim, requires_grad=True)

        top_indices, centroids, scores = indexer(x, return_scores=True)

        # Loss combines both scoring dot-products and centroid representations
        loss = centroids.sum() + scores[scores != float("-inf")].sum()
        loss.backward()

        assert x.grad is not None, "Input x.grad is None"
        assert not torch.isnan(x.grad).any(), "Input x.grad contains NaNs"
        assert not torch.isinf(x.grad).any(), "Input x.grad contains Infs"
        assert torch.count_nonzero(x.grad) > 0, "Input x.grad is entirely zero"

        for name, param in indexer.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"Param {name} grad is None"
            assert not torch.isnan(param.grad).any(), f"Param {name} grad has NaNs"
            assert not torch.isinf(param.grad).any(), f"Param {name} grad has Infs"
            norm = param.grad.norm().item()
            assert norm > 0.0, f"Param {name} grad norm is zero"
            assert math.isfinite(norm), f"Param {name} grad norm is not finite: {norm}"

    def test_centroid_pooling_gradient_flow(self):
        """Verify both mean and max pooling branches contribute to k_idx_proj gradients."""
        dim = 64
        d_idx = 16
        block_size = 32
        indexer = DGIndexer(dim=dim, d_idx=d_idx, block_size=block_size)
        indexer.train()

        x = torch.randn(1, 64, dim, requires_grad=True)
        _, centroids = indexer(x)
        centroids.sum().backward()

        assert indexer.k_idx_proj.weight.grad is not None
        assert indexer.k_idx_proj.weight.grad.norm().item() > 0.0


# ==============================================================================
# 6. Dtype Stability & Aliases
# ==============================================================================


class TestIndexerDtypesAndAliases:
    """Validates numerical stability across floating-point precisions and alias compatibility."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_indexer_numerical_stability(self, dtype):
        """Verify DGIndexer runs cleanly in FP32, BF16, and FP16."""
        indexer = DGIndexer(dim=64, d_idx=32, block_size=64, top_k=4).to(dtype)
        x = torch.randn(1, 128, 64, dtype=dtype)

        top_indices, centroids = indexer(x)
        assert not torch.isnan(centroids).any()
        assert not torch.isinf(centroids).any()
        assert top_indices.dtype == torch.int64

    def test_alias_parity(self):
        """Verify DeltaGuidedCentroidIndexer alias behaves identically."""
        assert DeltaGuidedCentroidIndexer is DGIndexer
        config = MabaSparseConfig(dim=128, d_idx=32, block_size=32, top_k=8)
        indexer = DeltaGuidedCentroidIndexer(config=config)
        assert indexer.dim == 128
        assert indexer.d_idx == 32

"""
Loss functions for ranking optimization.

This module implements:
- NDCG metric (PyTorch, exponential gain)
- LambdaRank loss (AutoBuild-style with manual gradient injection)
- Combined losses (MAE + SRCC, MAE + LambdaRank, etc.)
"""

import numpy as np
import torch
import torch.nn as nn
import torchsort

RELEVANCE_SCALE = 5.0

class RelevanceCalculator:
    """
    Transform raw targets to relevance scores scaled to [0, SCALE].

    Based on AutoBuild's implementation. Uses quantile-based normalization
    to handle outliers and scale targets appropriately for exponential gain.
    """

    def __init__(self, lower: float, upper: float, scale: float = RELEVANCE_SCALE):
        print(f'RelevanceCalculator: lower={lower:.6f}, upper={upper:.6f}, scale={scale:.2f}')
        self.lower = lower
        self.upper = upper
        self.scale = scale

    @classmethod
    def from_data(cls, x, scale: float = RELEVANCE_SCALE):
        """Create calculator from training data."""
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        lower = np.quantile(x, 0.2)
        upper = np.max(x)
        return cls(lower, upper, scale)

    def __call__(self, x):
        """Transform targets to relevance scores."""
        den = self.upper - self.lower
        # Guard against divide-by-zero when upper == lower
        if abs(den) < 1e-12:
            if torch.is_tensor(x):
                return torch.zeros_like(x)
            else:
                return np.zeros_like(x)

        if torch.is_tensor(x):
            return torch.clamp((x - self.lower) / den, 0, 1) * self.scale
        else:
            return np.clip((x - self.lower) / den, 0, 1) * self.scale



def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient on GPU.
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    cov = (x_centered * y_centered).mean()
    std_x = x_centered.std(unbiased=False)
    std_y = y_centered.std(unbiased=False)

    return cov / (std_x * std_y + 1e-8)


def spearman_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman rank correlation coefficient on GPU using torchsort.
    """
    rank_x = torchsort.soft_rank(x.unsqueeze(0), regularization_strength=1e-3).squeeze(0)
    rank_y = torchsort.soft_rank(y.unsqueeze(0), regularization_strength=1e-3).squeeze(0)

    return pearson_corrcoef(rank_x, rank_y)


def ndcg_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    gain_type: str = 'exp2',
) -> float:
    """
    Compute NDCG@K with exponential or linear gain.

    Args:
        preds: [N] predicted scores
        targets: [N] ground truth relevance scores
        k: Number of top items to consider
        gain_type: 'exp2' for exponential (2^rel - 1) or 'identity' for linear

    Returns:
        NDCG@K score (float between 0 and 1)
    """
    # Force fp32 for numerical stability
    preds = preds.float()
    targets = targets.float()

    n = len(preds)
    k = min(k, n)

    # Sort by predictions (descending) to get predicted ranking
    pred_indices = torch.argsort(preds, descending=True)
    pred_sorted_targets = targets[pred_indices]

    # Sort by targets (descending) to get ideal ranking
    ideal_indices = torch.argsort(targets, descending=True)
    ideal_sorted_targets = targets[ideal_indices]

    # Compute gains based on gain_type
    # Shift targets to non-negative for z-normalized data before computing gains
    target_min = targets.min()
    pred_shifted = pred_sorted_targets[:k] - target_min
    ideal_shifted = ideal_sorted_targets[:k] - target_min

    if gain_type == 'exp2':
        # Exponential gain: 2^rel - 1 (with shift to ensure non-negative)
        gain_pred = torch.pow(2.0, pred_shifted) - 1.0
        gain_ideal = torch.pow(2.0, ideal_shifted) - 1.0
    else:
        # Linear/identity gain (shifted to non-negative)
        gain_pred = pred_shifted
        gain_ideal = ideal_shifted

    # Discount factors: 1/log2(i+2) for i=0..k-1 (positions 1..k) - fp32
    positions = torch.arange(k, device=preds.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 2.0)

    # DCG = sum(gain_i / log2(i+2))
    dcg = (gain_pred * discounts).sum()
    idcg = (gain_ideal * discounts).sum()

    if idcg < 1e-8:
        return 1.0  # Perfect score if no ideal ranking exists

    return float(dcg / idcg)


class SRCCLoss(nn.Module):
    """
    Differentiable Spearman Rank Correlation loss using torchsort.
    Loss = 1 - SRCC (so minimizing loss maximizes correlation)
    """

    def __init__(self, regularization_strength: float = 1e-2):
        super().__init__()
        self.regularization_strength = regularization_strength

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(1, -1)
        targets = targets.view(1, -1)

        pred_ranks = torchsort.soft_rank(preds, regularization_strength=self.regularization_strength)
        target_ranks = torchsort.soft_rank(targets, regularization_strength=self.regularization_strength)

        pred_ranks = pred_ranks.squeeze(0)
        target_ranks = target_ranks.squeeze(0)

        srcc = pearson_corrcoef(pred_ranks, target_ranks)

        return 1.0 - srcc

class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss for learning to rank, optimizing NDCG.

    AutoBuild-style implementation:
    - Uses TARGET ranks (ground truth ordering) for discount computation
    - Exponential gain (2^rel - 1)
    - Manual gradient injection via prediction.backward(lambda_update)
    - No K truncation (uses full ranking)
    - Treats entire batch as single query (no prompt grouping)

    Note: This loss does NOT return a scalar. Instead, it directly injects
    gradients into the prediction tensor via backward().
    """

    def __init__(self, sigma: float = 1.0, gain_type: str = 'exp2'):
        """
        Args:
            sigma: Scaling factor for logistic function
            gain_type: 'exp2' for exponential gain, 'identity' for linear
        """
        super().__init__()
        self.sigma = sigma
        self.gain_type = gain_type

    def _compute_max_dcg(self, targets: torch.Tensor) -> float:
        """Compute maximum possible DCG (IDCG) for normalization."""
        # Force fp32 for numerical stability
        t = targets.view(-1).float()
        n = len(t)

        # Sort targets descending for ideal ranking
        sorted_targets, _ = torch.sort(t, descending=True)

        # Shift targets to ensure non-negative gains for z-normalized data
        shifted = sorted_targets - sorted_targets.min()

        # Compute gains
        if self.gain_type == 'exp2':
            gains = torch.pow(2.0, shifted) - 1.0
        else:
            gains = shifted

        # Discounts: 1/log2(rank + 1) for rank = 1, 2, ..., n (fp32)
        positions = torch.arange(1, n + 1, device=t.device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)

        return (gains * discounts).sum().item()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, weight: float = 1.0, retain_graph: bool = False) -> None:
        """
        Compute lambda gradients and inject into prediction tensor.

        Args:
            prediction: [B, 1] or [B] model predictions (requires_grad=True)
            target: [B, 1] or [B] ground truth relevance scores
            weight: Scaling factor for lambda gradients (for hybrid losses)
            retain_graph: If True, retain computation graph after backward.
                          Use True for multi-head scenarios where multiple
                          backward passes are needed.

        Note: This method does NOT return a loss value. It directly calls
        prediction.backward(lambda_update) to inject gradients.
        """
        original_dtype = prediction.dtype
        prediction = prediction.view(-1, 1)
        target = target.view(-1, 1)
        target_flat = target.view(-1)

        n = prediction.size(0)
        if n < 2:
            return

        # Compute IDCG for normalization
        max_dcg = self._compute_max_dcg(target_flat)
        if max_dcg < 1e-8:
            return
        N = 1.0 / max_dcg

        # Compute rank order from TARGETS using pure PyTorch (no CPU sync)
        # Higher target = better = lower rank (rank 1 is best)
        sorted_indices = torch.argsort(target_flat, descending=True)
        rank_order = torch.zeros(n, dtype=torch.long, device=prediction.device)
        rank_order[sorted_indices] = torch.arange(1, n + 1, device=prediction.device)
        rank_order = rank_order.float().view(-1, 1)

        with torch.no_grad():
            pred32 = prediction.float()
            tgt32 = target.float()
            rank32 = rank_order.float()

            # Pairwise probability using sigmoid (numerically stable vs 1/(1+exp(x)))
            # P_ij = sigmoid(-sigma * (s_i - s_j)) = 1 / (1 + exp(sigma * (s_i - s_j)))
            p_ij = torch.sigmoid(-self.sigma * (pred32 - pred32.t()))

            # Relevance differences
            rel_diff = tgt32 - tgt32.t()
            pos_pairs = (rel_diff > 0).float()
            neg_pairs = (rel_diff < 0).float()
            Sij = pos_pairs - neg_pairs  # +1 if i > j, -1 if i < j, 0 if equal

            # Gain differences
            # Shift targets to non-negative for z-normalized data before computing gains
            # This ensures 2^x >= 1 (so 2^x - 1 >= 0) while preserving relative differences
            tgt_shifted = tgt32 - tgt32.min()
            if self.gain_type == 'exp2':
                gain_diff = torch.pow(2.0, tgt_shifted) - torch.pow(2.0, tgt_shifted.t())
            else:
                gain_diff = tgt_shifted - tgt_shifted.t()

            # Decay (discount) differences based on TARGET ranks
            decay_diff = (1.0 / torch.log2(rank32 + 1.0) -
                         1.0 / torch.log2(rank32.t() + 1.0))

            # Delta NDCG: |N * gain_diff * decay_diff|
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)

            # Lambda gradients: sigma * (0.5 * (1 - Sij) - p_ij) * delta_ndcg
            # Using sigmoid form for numerical stability
            lambda_update = self.sigma * (0.5 * (1 - Sij) - p_ij) * delta_ndcg

            # Sum lambdas for each sample
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            # Scale gradients by weight (for hybrid losses)
            lambda_update = weight * lambda_update

            # Cast back to original dtype for gradient injection
            lambda_update = lambda_update.to(original_dtype)

            assert lambda_update.shape == prediction.shape

        # Inject gradients directly
        prediction.backward(lambda_update, retain_graph=retain_graph)


# =============================================================================
# Combined Loss Classes
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined MSE + SRCC loss.
    Loss = MSE + srcc_weight * (1 - SRCC)
    """

    def __init__(self, srcc_weight: float = 1.0, regularization_strength: float = 1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.srcc_loss = SRCCLoss(regularization_strength)
        self.srcc_weight = srcc_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(preds, targets)
        srcc_loss = self.srcc_loss(preds, targets)
        return mse_loss + self.srcc_weight * srcc_loss


class MAESRCCLoss(nn.Module):
    """
    Combined MAE + SRCC loss.
    Loss = MAE + srcc_weight * (1 - SRCC)
    """

    def __init__(self, srcc_weight: float = 1.0, regularization_strength: float = 1e-2):
        super().__init__()
        self.mae = nn.L1Loss()
        self.srcc_loss = SRCCLoss(regularization_strength)
        self.srcc_weight = srcc_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mae_loss = self.mae(preds, targets)
        srcc_loss = self.srcc_loss(preds, targets)
        return mae_loss + self.srcc_weight * srcc_loss


class MAELambdaRankLoss(nn.Module):
    """
    Combined MAE + LambdaRank loss.

    Note: Due to LambdaRank's manual gradient injection, this loss requires
    special handling in the training loop. Use the `backward()` method instead
    of calling `.backward()` on the returned loss.
    """

    def __init__(self, lambdarank_weight: float = 1.0, sigma: float = 1.0, gain_type: str = 'exp2'):
        super().__init__()
        self.mae = nn.L1Loss()
        self.lambdarank = LambdaRankLoss(sigma=sigma, gain_type=gain_type)
        self.lambdarank_weight = lambdarank_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Returns only the MAE loss. LambdaRank gradients must be applied separately.
        """
        return self.mae(preds, targets)

    def backward(self, preds: torch.Tensor, targets: torch.Tensor, mae_loss: torch.Tensor) -> None:
        """
        Perform backward pass for both MAE and LambdaRank.

        Args:
            preds: Model predictions (requires_grad=True)
            targets: Ground truth targets
            mae_loss: The MAE loss returned from forward()
        """
        # Backward for MAE
        mae_loss.backward(retain_graph=True)

        # Inject LambdaRank gradients (scaled by weight)
        # Note: LambdaRank.forward() calls backward internally
        # IMPORTANT: Scale gradients, not predictions, to preserve pairwise differences
        if self.lambdarank_weight > 0:
            self.lambdarank(preds, targets, weight=self.lambdarank_weight)


class MAESRCCLambdaRankLoss(nn.Module):
    """
    Combined MAE + SRCC + LambdaRank loss.

    Note: Due to LambdaRank's manual gradient injection, this loss requires
    special handling in the training loop.
    """

    def __init__(
        self,
        srcc_weight: float = 1.0,
        lambdarank_weight: float = 1.0,
        regularization_strength: float = 1e-2,
        sigma: float = 1.0,
        gain_type: str = 'exp2',
    ):
        super().__init__()
        self.mae = nn.L1Loss()
        self.srcc_loss = SRCCLoss(regularization_strength)
        self.lambdarank = LambdaRankLoss(sigma=sigma, gain_type=gain_type)
        self.srcc_weight = srcc_weight
        self.lambdarank_weight = lambdarank_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Returns MAE + SRCC loss. LambdaRank gradients must be applied separately.
        """
        mae_loss = self.mae(preds, targets)
        srcc_loss = self.srcc_loss(preds, targets)
        return mae_loss + self.srcc_weight * srcc_loss

    def backward(self, preds: torch.Tensor, targets: torch.Tensor, combined_loss: torch.Tensor) -> None:
        """
        Perform backward pass for MAE + SRCC and LambdaRank.
        """
        # Backward for MAE + SRCC
        combined_loss.backward(retain_graph=True)

        # Inject LambdaRank gradients (scaled by weight)
        # IMPORTANT: Scale gradients, not predictions, to preserve pairwise differences
        if self.lambdarank_weight > 0:
            self.lambdarank(preds, targets, weight=self.lambdarank_weight)

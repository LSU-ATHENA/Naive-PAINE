import numpy as np
import torch
import torch.nn as nn
import torchsort
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score

RELEVANCE_SCALE = 5.0


def _pearson_corrcoef_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    cov = (x_centered * y_centered).mean()
    std_x = x_centered.std(unbiased=False)
    std_y = y_centered.std(unbiased=False)

    return cov / (std_x * std_y + 1e-8)


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    corr, _ = pearsonr(x_np, y_np)
    return torch.tensor(corr, dtype=x.dtype)


def spearman_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    corr, _ = spearmanr(x_np, y_np)
    return torch.tensor(corr, dtype=x.dtype)


def spearman_per_prompt(preds: torch.Tensor, targets: torch.Tensor, prompt_ids: torch.Tensor) -> float:
    """Within-prompt Spearman, averaged over prompts (the selection signal; global srcc is the prior)."""
    p = preds.detach().cpu().float().numpy()
    t = targets.detach().cpu().float().numpy()
    pid = prompt_ids.detach().cpu().numpy()
    out = []
    for u in np.unique(pid):
        m = pid == u
        if m.sum() < 2 or p[m].std() < 1e-9 or t[m].std() < 1e-9:
            continue
        c, _ = spearmanr(p[m], t[m])
        if not np.isnan(c):
            out.append(c)
    return float(np.mean(out)) if out else 0.0


def ndcg_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    gain_type: str = 'exp2',
) -> float:

    preds_np = preds.detach().cpu().float().numpy()
    targets_np = targets.detach().cpu().float().numpy()

    n = len(preds_np)
    k = min(k, n)

    target_range = targets_np.max() - targets_np.min()
    if target_range < 1e-8:
        return 1.0

    targets_scaled = (targets_np - targets_np.min()) / target_range * RELEVANCE_SCALE

    if gain_type == 'exp2':
        return float(ndcg_score(
            y_true=targets_scaled.reshape(1, -1),
            y_score=preds_np.reshape(1, -1),
            k=k,
        ))
    else:
        sorted_by_pred = targets_scaled[np.argsort(-preds_np)][:k]
        sorted_ideal = np.sort(targets_scaled)[::-1][:k]
        positions = np.arange(k) + 2.0
        discounts = 1.0 / np.log2(positions)
        dcg = (sorted_by_pred * discounts).sum()
        idcg = (sorted_ideal * discounts).sum()
        if idcg < 1e-8:
            return 1.0
        return float(dcg / idcg)


def ndcg_at_k_per_prompt(
    preds: torch.Tensor,
    targets: torch.Tensor,
    prompt_ids: torch.Tensor,
    k: int = 5,
    gain_type: str = 'exp2',
) -> float:
    preds_np = preds.detach().cpu().float().numpy()
    targets_np = targets.detach().cpu().float().numpy()
    pids_np = prompt_ids.detach().cpu().numpy()

    unique_pids = np.unique(pids_np)
    ndcg_scores = []

    for pid in unique_pids:
        mask = (pids_np == pid)
        p = preds_np[mask]
        t = targets_np[mask]

        n = len(p)
        if n < 2:
            continue

        kk = min(k, n)

        t_range = t.max() - t.min()
        if t_range < 1e-8:
            continue

        t_scaled = (t - t.min()) / t_range * RELEVANCE_SCALE

        if gain_type == 'exp2':
            score = float(ndcg_score(
                y_true=t_scaled.reshape(1, -1),
                y_score=p.reshape(1, -1),
                k=kk,
            ))
        else:
            sorted_by_pred = t_scaled[np.argsort(-p)][:kk]
            sorted_ideal = np.sort(t_scaled)[::-1][:kk]
            positions = np.arange(kk) + 2.0
            discounts = 1.0 / np.log2(positions)
            dcg = (sorted_by_pred * discounts).sum()
            idcg = (sorted_ideal * discounts).sum()
            score = float(dcg / idcg) if idcg > 1e-8 else 1.0

        ndcg_scores.append(score)

    if len(ndcg_scores) == 0:
        return 0.0

    return float(np.mean(ndcg_scores))


class SRCCLoss(nn.Module):

    def __init__(self, regularization_strength: float = 1e-2):
        super().__init__()
        self.regularization_strength = regularization_strength

    def _srcc_for_group(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = preds.view(1, -1)
        t = targets.view(1, -1)
        pred_ranks = torchsort.soft_rank(p, regularization_strength=self.regularization_strength)
        target_ranks = torchsort.soft_rank(t, regularization_strength=self.regularization_strength)
        srcc = _pearson_corrcoef_torch(pred_ranks.squeeze(0), target_ranks.squeeze(0))
        return 1.0 - srcc

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        if group_ids is None:
            return self._srcc_for_group(preds, targets)

        srcc_losses = []
        for gid in group_ids.unique():
            mask = (group_ids == gid)
            if mask.sum() < 2:
                continue
            srcc_losses.append(self._srcc_for_group(preds[mask], targets[mask]))

        if len(srcc_losses) == 0:
            return self._srcc_for_group(preds, targets)

        return torch.stack(srcc_losses).mean()


class LambdaRankLoss(nn.Module):

    def __init__(self, sigma: float = 1.0, gain_type: str = 'exp2', k: int = 0):
        super().__init__()
        self.sigma = sigma
        self.gain_type = gain_type
        self.k = k  # NDCG@k truncation; 0 = full-rank (LambdaRank), >0 = LambdaLoss@k

    def _discount(self, rank: torch.Tensor) -> torch.Tensor:
        d = 1.0 / torch.log2(rank + 1.0)
        if self.k > 0:
            d = torch.where(rank <= self.k, d, torch.zeros_like(d))
        return d

    def _compute_max_dcg(self, targets: torch.Tensor) -> float:
        t = targets.view(-1).float()
        n = len(t)

        sorted_targets, _ = torch.sort(t, descending=True)
        shifted = sorted_targets - sorted_targets.min()

        if self.gain_type == 'exp2':
            gains = torch.pow(2.0, shifted) - 1.0
        else:
            gains = shifted

        positions = torch.arange(1, n + 1, device=t.device, dtype=torch.float32)
        discounts = self._discount(positions)   # IDCG@k: positions are ideal-ranked, _discount zeros rank>k

        return (gains * discounts).sum().item()

    def _compute_lambdas_for_group(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.view(-1, 1)
        target = target.view(-1, 1)
        target_flat = target.view(-1)

        n = prediction.size(0)
        if n < 2:
            return torch.zeros_like(prediction)

        max_dcg = self._compute_max_dcg(target_flat)
        if max_dcg < 1e-8:
            return torch.zeros_like(prediction)
        N = 1.0 / max_dcg

        sorted_indices = torch.argsort(target_flat, descending=True)
        rank_order = torch.zeros(n, dtype=torch.long, device=prediction.device)
        rank_order[sorted_indices] = torch.arange(1, n + 1, device=prediction.device)
        rank_order = rank_order.float().view(-1, 1)

        pred32 = prediction.float()
        tgt32 = target.float()
        rank32 = rank_order.float()

        p_ij = torch.sigmoid(-self.sigma * (pred32 - pred32.t()))

        rel_diff = tgt32 - tgt32.t()
        pos_pairs = (rel_diff > 0).float()
        neg_pairs = (rel_diff < 0).float()
        Sij = pos_pairs - neg_pairs

        tgt_shifted = tgt32 - tgt32.min()
        if self.gain_type == 'exp2':
            gain_diff = torch.pow(2.0, tgt_shifted) - torch.pow(2.0, tgt_shifted.t())
        else:
            gain_diff = tgt_shifted - tgt_shifted.t()

        decay_diff = self._discount(rank32) - self._discount(rank32.t())

        delta_ndcg = torch.abs(N * gain_diff * decay_diff)

        lambda_update = self.sigma * (0.5 * (1 - Sij) - p_ij) * delta_ndcg
        lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

        return lambda_update

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: float = 1.0,
        retain_graph: bool = False,
        group_ids: torch.Tensor = None,
    ) -> None:
        original_dtype = prediction.dtype
        pred_flat = prediction.view(-1, 1)
        tgt_flat = target.view(-1, 1)

        n = pred_flat.size(0)
        if n < 2:
            return

        with torch.no_grad():
            if group_ids is None:
                lambda_update = self._compute_lambdas_for_group(pred_flat, tgt_flat)
            else:
                lambda_update = torch.zeros_like(pred_flat)
                for gid in group_ids.unique():
                    mask = (group_ids == gid)
                    if mask.sum() < 2:
                        continue
                    group_lambdas = self._compute_lambdas_for_group(
                        pred_flat[mask], tgt_flat[mask],
                    )
                    lambda_update[mask] = group_lambdas

            lambda_update = weight * lambda_update
            lambda_update = lambda_update.to(original_dtype)

        prediction.view(-1, 1).backward(lambda_update, retain_graph=retain_graph)


class MAESRCCLoss(nn.Module):

    def __init__(self, srcc_weight: float = 1.0, regularization_strength: float = 1e-2):
        super().__init__()
        self.mae = nn.L1Loss()
        self.srcc_loss = SRCCLoss(regularization_strength)
        self.srcc_weight = srcc_weight

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        mae_loss = self.mae(preds, targets)
        srcc_loss = self.srcc_loss(preds, targets, group_ids=group_ids)
        return mae_loss + self.srcc_weight * srcc_loss


class MAELambdaRankLoss(nn.Module):

    def __init__(self, lambdarank_weight: float = 1.0, sigma: float = 1.0, gain_type: str = 'exp2', k: int = 0):
        super().__init__()
        self.mae = nn.L1Loss()
        self.lambdarank = LambdaRankLoss(sigma=sigma, gain_type=gain_type, k=k)
        self.lambdarank_weight = lambdarank_weight

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.mae(preds, targets)

    def backward(self, preds: torch.Tensor, targets: torch.Tensor, mae_loss: torch.Tensor, group_ids: torch.Tensor = None) -> None:
        mae_loss.backward(retain_graph=True)
        if self.lambdarank_weight > 0:
            self.lambdarank(preds, targets, weight=self.lambdarank_weight, group_ids=group_ids)

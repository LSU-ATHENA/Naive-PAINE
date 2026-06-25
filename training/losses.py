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

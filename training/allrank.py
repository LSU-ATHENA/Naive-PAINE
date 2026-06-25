"""Listwise ranking losses (LambdaRank / LambdaLoss / NeuralNDCG) for prompt-grouped batches.

The ranking functions are vendored from allegro/allRank (Apache-2.0):
    https://github.com/allegro/allRank  (Pobrotyn et al., NeuralNDCG: arXiv:2102.07831)
We vendor instead of `pip install allRank` because allRank pins numpy<=1.21.6 / gcsfs==0.6.2 and
imports gcsfs at module top-level, which is uninstallable in a modern CUDA-torch env. Only torch +
numpy are needed; the compute device is taken from the input tensors.

`AllRankLoss` turns each prompt's noises into one allRank slate, min-max scales the continuous
z-targets per slate to graded relevance in [0, REL], and optionally adds an MAE prior anchor.
"""
import numpy as np
import torch
import torch.nn as nn

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1.0


def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)
    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)
        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
    return mat


def deterministic_neural_sort(s, tau, mask):
    dev = s.device
    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)
    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))
    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    P_hat = torch.nn.Softmax(-1)(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10):
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10):
    dev = s.device
    batch_size, n = s.size()[0], s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)
    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)
    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    return P_hat.view(n_samples, batch_size, n, n)


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0
    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    y_true = y_true.clone()
    y_pred = y_pred.clone()
    actual_length = y_true.shape[1]
    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]
    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)
    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)
    gains = gain_function(true_sorted_by_preds)
    discounted_gains = (gains * discounts)[:, :np.max(ats)]
    cum_dcg = torch.cumsum(discounted_gains, dim=1)
    ats_tensor = (torch.tensor(ats, dtype=torch.long) - 1).to(cum_dcg.device)
    return cum_dcg[:, ats_tensor]


def lambdaLoss(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None,
               sigma=1., mu=10., reduction="sum", reduction_log="binary"):
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)

    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")
    return loss


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()
    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def neuralNDCG(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    dev = y_pred.device
    if k is None:
        k = y_true.shape[1]

    mask = (y_true == padded_value_indicator)
    if stochastic:
        P_hat = stochastic_neural_sort(y_pred.unsqueeze(-1), n_samples=n_samples, tau=temperature, mask=mask,
                                       beta=beta, log_scores=log_scores)
    else:
        P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)

    P_hat = sinkhorn_scaling(P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
                             mask.repeat_interleave(P_hat.shape[0], dim=0), tol=1e-6, max_iter=50)
    P_hat = P_hat.view(int(P_hat.shape[0] / y_pred.shape[0]), y_pred.shape[0], P_hat.shape[1], P_hat.shape[2])

    P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1).unsqueeze(0)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.

    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (torch.tensor(1.) / torch.log2(torch.arange(y_true.shape[-1], dtype=torch.float) + 2.)).to(dev)
    discounted_gains = ground_truth * discounts

    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)

    discounted_gains = discounted_gains[:, :, :k]
    ndcg = discounted_gains.sum(dim=-1) / (idcg + DEFAULT_EPS)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.)
    if idcg_mask.all():
        return y_pred.sum() * 0.0
    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])
    return -1. * mean_ndcg


REL = 5.0
PAD = PADDED_Y_VALUE
_KIND_TO_SCHEME = {'lambdarank': 'lambdaRank_scheme', 'lambdaloss': 'ndcgLoss2_scheme'}


def _build_slates(preds, targets, group_ids):
    p = preds.view(-1)
    t = targets.view(-1)
    if group_ids is None:
        groups = [torch.arange(p.numel(), device=p.device)]
    else:
        g = group_ids.view(-1)
        groups = [(g == u).nonzero(as_tuple=True)[0] for u in g.unique()]
    groups = [idx for idx in groups if idx.numel() >= 2]
    if not groups:
        return None, None
    n_max = max(idx.numel() for idx in groups)
    pred_mat = p.new_zeros(len(groups), n_max)
    rel_mat = p.new_full((len(groups), n_max), PAD)
    for i, idx in enumerate(groups):
        pi, ti = p[idx], t[idx]
        rng = ti.max() - ti.min()
        rel = (ti - ti.min()) / rng * REL if rng > 1e-8 else torch.zeros_like(ti)
        n = idx.numel()
        pred_mat[i, :n] = pi
        rel_mat[i, :n] = rel
    return pred_mat, rel_mat


class AllRankLoss(nn.Module):
    def __init__(self, kind: str, k: int = 5, sigma: float = 1.0, temperature: float = 1.0,
                 use_mae: bool = True, mae_weight: float = 1.0, rank_weight: float = 1.0):
        super().__init__()
        if kind not in ('lambdarank', 'lambdaloss', 'neuralndcg'):
            raise ValueError(f"unknown allRank loss kind: {kind}")
        self.kind = kind
        self.k = k if (k and k > 0) else None
        self.sigma = sigma
        self.temperature = temperature
        self.use_mae = use_mae
        self.mae_weight = mae_weight
        self.rank_weight = rank_weight
        self.mae = nn.L1Loss()

    def _rank_term(self, pred_mat, rel_mat):
        if self.kind == 'neuralndcg':
            return neuralNDCG(pred_mat, rel_mat, temperature=self.temperature, k=self.k, padded_value_indicator=PAD)
        return lambdaLoss(pred_mat, rel_mat, weighing_scheme=_KIND_TO_SCHEME[self.kind], k=self.k,
                          sigma=self.sigma, padded_value_indicator=PAD, reduction="mean")

    def terms(self, preds, targets, group_ids=None):
        pred_mat, rel_mat = _build_slates(preds, targets, group_ids)
        rank = self._rank_term(pred_mat, rel_mat) if pred_mat is not None else preds.sum() * 0.0
        rank = self.rank_weight * rank
        mae = self.mae(preds, targets) if self.use_mae else preds.new_zeros(())
        return {'total': self.mae_weight * mae + rank, 'mae': mae, 'rank': rank}

    def forward(self, preds, targets, group_ids=None):
        return self.terms(preds, targets, group_ids)['total']

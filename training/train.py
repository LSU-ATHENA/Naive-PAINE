import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import prep_dataloaders, denormalize, AVAILABLE_TARGETS
from model import get_model, get_dims, MODEL_DIMS, NOISE_ENCODERS, TEXT_ENCODERS
from losses import (
    ndcg_at_k_per_prompt,
    spearman_corrcoef,
    pearson_corrcoef,
    spearman_per_prompt,
    MAESRCCLoss,
    MAELambdaRankLoss,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, use_grouped=False) -> Dict[str, float]:
    model.train()
    running_total_loss = 0.0
    uses_lambdarank = isinstance(criterion, MAELambdaRankLoss)

    for batch in loader:
        noise = batch['noise'].to(device)
        prompt_embeds = batch['prompt_embeds'].to(device)
        prompt_mask = batch['prompt_mask'].to(device)
        targets = batch['y'].to(device).unsqueeze(1)
        group_ids = batch['prompt_id'].to(device) if use_grouped else None

        optimizer.zero_grad()
        preds = model(noise, prompt_embeds, prompt_mask)

        if uses_lambdarank:
            loss = criterion(preds, targets, group_ids=group_ids)
            criterion.backward(preds, targets, loss, group_ids=group_ids)
        else:
            loss = criterion(preds, targets, group_ids=group_ids) if group_ids is not None \
                else criterion(preds, targets)
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_total_loss += loss.item() * noise.size(0)

    return {'loss': running_total_loss / len(loader.dataset)}


@torch.no_grad()
def evaluate(model, loader, device, y_mean=0.0, y_std=1.0, gain_type='exp2') -> Dict[str, float]:
    model.eval()
    P, T, I = [], [], []
    for batch in loader:
        preds = model(batch['noise'].to(device), batch['prompt_embeds'].to(device),
                      batch['prompt_mask'].to(device)).squeeze(1)
        P.append(denormalize(preds, y_mean, y_std))
        T.append(batch['raw_y'].to(device))
        I.append(batch['prompt_id'].to(device))
    preds, targets, pids = torch.cat(P), torch.cat(T), torch.cat(I)

    n = len(preds)
    mae = (preds - targets).abs().mean().item()
    if n > 1 and preds.std() > 1e-9:
        srcc = spearman_corrcoef(preds, targets).item()
        pearson = pearson_corrcoef(preds, targets).item()
        pp_srcc = spearman_per_prompt(preds, targets, pids)
        pp_ndcg5 = ndcg_at_k_per_prompt(preds, targets, pids, k=5, gain_type=gain_type)
        pp_ndcg10 = ndcg_at_k_per_prompt(preds, targets, pids, k=10, gain_type=gain_type)
    else:
        srcc = pearson = pp_srcc = pp_ndcg5 = pp_ndcg10 = 0.0

    return {
        'n_samples': n, 'mae_raw': mae,
        'srcc': srcc, 'pearson': pearson,
        'pp_srcc': pp_srcc, 'pp_ndcg5': pp_ndcg5, 'pp_ndcg10': pp_ndcg10,
        'target_mean': targets.mean().item(), 'target_std': targets.std().item(),
        'pred_mean': preds.mean().item(), 'pred_std': preds.std().item(),
    }


@torch.no_grad()
def evaluate_prompt_alone(model, loader, device, y_mean, y_std) -> float:
    """Text Prompt Alone: mask the noise encoder, one text-only score per prompt vs mu_label."""
    model.eval()
    pred_by_pid, raw_by_pid = {}, {}
    for batch in loader:
        preds_raw = denormalize(
            model(batch['noise'].to(device), batch['prompt_embeds'].to(device),
                  batch['prompt_mask'].to(device), mask_noise=True).squeeze(1),
            y_mean, y_std).cpu()
        raw = batch['raw_y'].cpu()
        for i, pid in enumerate(batch['prompt_id'].tolist()):
            pred_by_pid.setdefault(pid, preds_raw[i].item())
            raw_by_pid.setdefault(pid, []).append(raw[i].item())
    pids = sorted(pred_by_pid)
    y_hat = torch.tensor([pred_by_pid[p] for p in pids])
    mu_label = torch.tensor([float(np.mean(raw_by_pid[p])) for p in pids])
    srcc = spearman_corrcoef(y_hat, mu_label).item()
    mae = (y_hat - mu_label).abs().mean().item()
    mape = ((y_hat - mu_label).abs() / mu_label.abs().clamp_min(1e-8)).mean().item() * 100
    print(f"[Text Prompt Alone] prompts={len(pids)}  SRCC={srcc:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%  "
          f"dist=N({mu_label.mean():.2f},{mu_label.std():.2f})  "
          f"range=[{mu_label.min():.2f},{mu_label.max():.2f}]")
    return srcc


def load_predictor_from_ckpt(path, device):
    """Build the model from the checkpoint's own model_config (encoders/dims may differ per DM)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck['model_config']
    dims = get_dims(cfg['model_type'], cfg.get('text_embed_type', 'default')) \
        if cfg.get('model_type') in MODEL_DIMS else {}
    model = get_model(
        noise_enc=cfg.get('noise_enc', 'custom'),
        text_enc=cfg['text_enc'],
        dropout=cfg.get('dropout', 0.1),
        num_heads=cfg.get('num_heads', 1),
        spatial_size=cfg.get('spatial_size', dims.get('spatial_size', 128)),
        in_channels=cfg.get('in_channels', dims.get('latent_shape', (4,))[0]),
        embed_dim=cfg.get('embed_dim', dims.get('embed_dim', 2048)),
        seq_len=cfg.get('seq_len', dims.get('seq_len', 77)),
        pos_encoding=cfg.get('pos_encoding', 'none'),
    ).to(device)
    model.load_state_dict({k: v.float() for k, v in ck['model_state_dict'].items()})
    model.eval()
    return model, ck.get('normalization', {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_type', required=True, choices=list(MODEL_DIMS))
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--noise_enc', default='custom', choices=NOISE_ENCODERS)
    ap.add_argument('--text_enc', default='summarytoken', choices=TEXT_ENCODERS)
    ap.add_argument('--text_embed_type', default='default')
    ap.add_argument('--target', default='pick_score', choices=AVAILABLE_TARGETS)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-8)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--loss', default='mae+srcc', choices=['mae+srcc', 'mae+lambdarank'])
    ap.add_argument('--lambdarank_k', type=int, default=5)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--k_prompts', type=int, default=6)
    ap.add_argument('--primary_metric', default='srcc', choices=['srcc', 'pp_ndcg5', 'pp_srcc'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_prompts', type=int, default=-1)
    ap.add_argument('--exp_name', default='baseline')
    ap.add_argument('--output_dir', default='./experiments')
    ap.add_argument('--eval_only', default=None, help='path to a .pth: load it, evaluate, exit (no training)')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    dims = get_dims(args.model_type, text_embed_type=args.text_embed_type)
    use_grouped = args.k_prompts > 0

    train_loader, val_loader, test_loader, stats = prep_dataloaders(
        data_dir=args.data_dir, model_type=args.model_type, target=args.target, split_by='prompt',
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        k_prompts_per_batch=args.k_prompts, text_embed_type=args.text_embed_type, max_prompts=args.max_prompts)
    y_mean, y_std = stats['y_mean'], stats['y_std']

    if args.eval_only:
        model, nrm = load_predictor_from_ckpt(args.eval_only, device)
        ym, ys = nrm.get('y_mean', y_mean), nrm.get('y_std', y_std)
        val = evaluate(model, val_loader, device, ym, ys)
        test = evaluate(model, test_loader, device, ym, ys)
        print('VAL ', {k: round(v, 4) for k, v in val.items()})
        print('TEST', {k: round(v, 4) for k, v in test.items()})
        pa_val = evaluate_prompt_alone(model, val_loader, device, ym, ys)
        pa_test = evaluate_prompt_alone(model, test_loader, device, ym, ys)
        out = {'checkpoint': args.eval_only, 'val': val, 'test': test,
               'prompt_alone_srcc': {'val': pa_val, 'test': pa_test}}
        Path(args.eval_only).with_suffix('.eval.json').write_text(json.dumps(out, indent=2))
        print(f"prompt-alone SRCC  val={pa_val:.4f}  test={pa_test:.4f}")
        return

    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'config.json').write_text(json.dumps(vars(args), indent=4))

    model = get_model(
        noise_enc=args.noise_enc, text_enc=args.text_enc, dropout=args.dropout, num_heads=1,
        spatial_size=dims['spatial_size'], in_channels=dims['latent_shape'][0],
        embed_dim=dims['embed_dim'], seq_len=dims['seq_len'], pos_encoding='sinusoidal').to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == 'mae+srcc':
        criterion = MAESRCCLoss(srcc_weight=1.0, regularization_strength=1e-2)
    else:
        criterion = MAELambdaRankLoss(lambdarank_weight=1.0, sigma=1.0, gain_type='exp2', k=args.lambdarank_k)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    def save_ckpt(path):
        torch.save({
            'model_state_dict': {k: v.half() for k, v in model.state_dict().items()},
            'model_config': {
                'noise_enc': args.noise_enc, 'text_enc': args.text_enc, 'dropout': args.dropout,
                'num_heads': 1, 'model_type': args.model_type, 'spatial_size': dims['spatial_size'],
                'in_channels': dims['latent_shape'][0], 'embed_dim': dims['embed_dim'],
                'seq_len': dims['seq_len'], 'pos_encoding': 'sinusoidal',
                'text_embed_type': args.text_embed_type,
            },
            'normalization': {'target': args.target, 'y_mean': y_mean, 'y_std': y_std},
        }, path)

    best, best_epoch, history = float('-inf'), 0, []
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, use_grouped=use_grouped)
        val = evaluate(model, val_loader, device, y_mean, y_std)
        primary = val[args.primary_metric]
        print(f"Epoch {epoch+1}/{args.epochs}  srcc={val['srcc']:.4f}  ppNDCG5={val['pp_ndcg5']:.4f}  "
              f"ppSRCC={val['pp_srcc']:.4f}  MAE={val['mae_raw']:.4f}", flush=True)
        scheduler.step(primary)
        history.append({'epoch': epoch + 1, 'val': val})
        if primary > best:
            best, best_epoch = primary, epoch + 1
            save_ckpt(exp_dir / 'best_model.pth')

    best_ck = torch.load(exp_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict({k: v.float() for k, v in best_ck['model_state_dict'].items()})
    test = evaluate(model, test_loader, device, y_mean, y_std)
    pa_val = evaluate_prompt_alone(model, val_loader, device, y_mean, y_std)
    pa_test = evaluate_prompt_alone(model, test_loader, device, y_mean, y_std)
    summary = {'best_epoch': best_epoch, 'primary_metric': args.primary_metric,
               'best_val': history[best_epoch - 1]['val'], 'test': test,
               'prompt_alone_srcc': {'val': pa_val, 'test': pa_test}}
    (exp_dir / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(f"best epoch {best_epoch}  val {args.primary_metric}={best:.4f}")
    print(f"  test ppNDCG5={test['pp_ndcg5']:.4f} ppSRCC={test['pp_srcc']:.4f} "
          f"ppNDCG10={test['pp_ndcg10']:.4f} srcc={test['srcc']:.4f} MAE={test['mae_raw']:.4f}")
    print(f"  prompt-alone SRCC val={pa_val:.4f} test={pa_test:.4f}")


if __name__ == '__main__':
    main()

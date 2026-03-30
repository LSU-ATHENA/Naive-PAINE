import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from predictor.training.dataloader import prep_dataloaders, denormalize, AVAILABLE_TARGETS
from predictor.models import get_model, NOISE_ENCODERS, TEXT_ENCODERS
from predictor.configs.model_dims import MODEL_DIMS, get_dims

from predictor.training.losses import (
    ndcg_at_k,
    ndcg_at_k_per_prompt,
    spearman_corrcoef,
    pearson_corrcoef,
    MAESRCCLoss,
    MAELambdaRankLoss,
    LambdaRankLoss,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    loss_type: str = 'mae+srcc',
    use_grouped: bool = False,
) -> Dict[str, float]:
    model.train()
    running_display_loss = 0.0
    running_total_loss = 0.0

    targetlist = []
    predictionlist = []

    uses_lambdarank = isinstance(criterion, MAELambdaRankLoss)

    for batch_idx, batch in enumerate(loader):
        noise = batch['noise'].to(device)
        prompt_embeds = batch['prompt_embeds'].to(device)
        prompt_mask = batch['prompt_mask'].to(device)

        optimizer.zero_grad()
        preds = model(noise, prompt_embeds, prompt_mask)

        targets = batch['y'].to(device).unsqueeze(1)

        group_ids = batch['prompt_id'].to(device) if use_grouped else None

        if uses_lambdarank:
            loss = criterion(preds, targets, group_ids=group_ids)
            criterion.backward(preds, targets, loss, group_ids=group_ids)
            batch_display_loss = loss.item()
            batch_total_loss = loss.item()
        else:
            if group_ids is not None:
                loss = criterion(preds, targets, group_ids=group_ids)
            else:
                loss = criterion(preds, targets)
            loss.backward()
            batch_display_loss = loss.item()
            batch_total_loss = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_display_loss += batch_display_loss * noise.size(0)
        running_total_loss += batch_total_loss * noise.size(0)
        targetlist.extend(targets.squeeze(1).cpu().numpy())
        predictionlist.extend(preds.squeeze(1).detach().cpu().numpy())

    n_samples = len(loader.dataset)
    result = {
        'display_loss': running_display_loss / n_samples,
        'total_loss': running_total_loss / n_samples,
        'loss': running_display_loss / n_samples,
        'target_mean': float(np.mean(targetlist)),
        'target_std': float(np.std(targetlist)),
        'pred_mean': float(np.mean(predictionlist)),
        'pred_std': float(np.std(predictionlist)),
    }

    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ndcg_k: int = 5,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    gain_type: str = 'exp2',
) -> Dict[str, float]:
    model.eval()

    all_preds_raw = []
    all_targets_raw = []
    all_prompt_ids = []

    for batch in loader:
        noise = batch['noise'].to(device)
        prompt_embeds = batch['prompt_embeds'].to(device)
        prompt_mask = batch['prompt_mask'].to(device)
        targets_raw = batch['raw_y'].to(device)
        prompt_ids = batch['prompt_id'].to(device)

        preds_norm = model(noise, prompt_embeds, prompt_mask).squeeze(1)
        preds_raw = denormalize(preds_norm, y_mean, y_std)

        all_preds_raw.append(preds_raw)
        all_targets_raw.append(targets_raw)
        all_prompt_ids.append(prompt_ids)

    all_preds_raw = torch.cat(all_preds_raw, dim=0)
    all_targets_raw = torch.cat(all_targets_raw, dim=0)
    all_prompt_ids = torch.cat(all_prompt_ids, dim=0)

    n_samples = len(all_preds_raw)
    mae_raw = (all_preds_raw - all_targets_raw).abs().mean().item()

    if n_samples > 1 and all_preds_raw.std() > 1e-9:
        srcc = spearman_corrcoef(all_preds_raw, all_targets_raw).item()
        pearson = pearson_corrcoef(all_preds_raw, all_targets_raw).item()
        ndcg = ndcg_at_k_per_prompt(
            all_preds_raw, all_targets_raw, all_prompt_ids,
            k=ndcg_k, gain_type=gain_type,
        )
    else:
        srcc = 0.0
        pearson = 0.0
        ndcg = 0.0

    return {
        'n_samples': n_samples,
        'mae_raw': mae_raw,
        'srcc': srcc,
        'pearson': pearson,
        f'ndcg_{ndcg_k}': ndcg,
        'target_mean': all_targets_raw.mean().item(),
        'target_std': all_targets_raw.std().item(),
        'pred_mean': all_preds_raw.mean().item(),
        'pred_std': all_preds_raw.std().item(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, required=True,
                        choices=list(MODEL_DIMS.keys()))
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--noise_enc', type=str, default='custom', choices=NOISE_ENCODERS)
    parser.add_argument('--text_enc', type=str, default='summarytoken', choices=TEXT_ENCODERS)

    parser.add_argument('--target', type=str, default='hpsv2', choices=AVAILABLE_TARGETS)
    parser.add_argument('--split_by', type=str, default='prompt', choices=['prompt', 'seed'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--loss', type=str, default='mae+srcc',
                        choices=['mae+srcc', 'mae+lambdarank'])
    parser.add_argument('--gain_type', type=str, default='exp2', choices=['exp2', 'identity'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--pos_encoding', type=str, default='none',
                        choices=['none', 'sinusoidal'])
    parser.add_argument('--text_embed_type', type=str, default='default',
                        choices=['default', 't5', 't5+clip'])
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_prompts', type=int, default=-1)

    parser.add_argument('--k_prompts', type=int, default=2)

    parser.add_argument('--ndcg_k', type=int, default=3)

    parser.add_argument('--patience', type=int, default=-1)

    parser.add_argument('--primary_metric', type=str, default='ndcg',
                        choices=['ndcg', 'srcc', 'mae', 'pearson'])

    args = parser.parse_args()

    dims = get_dims(args.model_type, text_embed_type=args.text_embed_type)
    spatial_size = dims['spatial_size']
    in_channels = dims['latent_shape'][0]
    embed_dim = dims['embed_dim']
    seq_len = dims['seq_len']

    if args.epochs == -1 or args.patience == -1:
        default_epochs, default_patience = 30, 12

        if args.epochs == -1:
            args.epochs = default_epochs
        if args.patience == -1:
            args.patience = default_patience

    set_seed(args.seed)
    exp_dir = Path(args.output_dir) / f"{args.exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    use_grouped = args.k_prompts > 0

    train_loader, val_loader, test_loader, stats = prep_dataloaders(
        data_dir=args.data_dir,
        model_type=args.model_type,
        target=args.target,
        split_by=args.split_by,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        k_prompts_per_batch=args.k_prompts,
        text_embed_type=args.text_embed_type,
        max_prompts=args.max_prompts,
    )

    y_mean, y_std = stats['y_mean'], stats['y_std']

    model = get_model(
        noise_enc=args.noise_enc,
        text_enc=args.text_enc,
        dropout=args.dropout,
        num_heads=1,
        spatial_size=spatial_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        seq_len=seq_len,
        pos_encoding=args.pos_encoding,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss == 'mae+srcc':
        criterion = MAESRCCLoss(srcc_weight=1.0, regularization_strength=1e-2)
    elif args.loss == 'mae+lambdarank':
        criterion = MAELambdaRankLoss(lambdarank_weight=1.0, sigma=1.0, gain_type=args.gain_type)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    primary_higher_better = (args.primary_metric != 'mae')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max' if primary_higher_better else 'min', factor=0.5, patience=5
    )

    best_primary_value = float('-inf') if primary_higher_better else float('inf')
    best_epoch = -1
    best_val_metrics = {}

    epochs_without_improvement = 0

    ndcg_key = f'ndcg_{args.ndcg_k}'

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch,
            loss_type=args.loss,
            use_grouped=use_grouped,
        )
        val_metrics = evaluate(
            model, val_loader, device, args.ndcg_k,
            y_mean=y_mean, y_std=y_std,
            gain_type=args.gain_type,
        )

        if args.primary_metric == 'ndcg':
            current_primary = val_metrics[ndcg_key]
        elif args.primary_metric == 'srcc':
            current_primary = val_metrics['srcc']
        elif args.primary_metric == 'mae':
            current_primary = val_metrics['mae_raw']
        elif args.primary_metric == 'pearson':
            current_primary = val_metrics['pearson']

        scheduler.step(current_primary)

        checkpoint = {
            'model_state_dict': {k: v.half() for k, v in model.state_dict().items()},
            'model_config': {
                'noise_enc': args.noise_enc,
                'text_enc': args.text_enc,
                'dropout': args.dropout,
                'num_heads': 1,
                'model_type': args.model_type,
                'spatial_size': spatial_size,
                'in_channels': in_channels,
                'embed_dim': embed_dim,
                'seq_len': seq_len,
                'pos_encoding': args.pos_encoding,
                'text_embed_type': args.text_embed_type,
            },
            'normalization': {
                'target': args.target,
                'y_mean': y_mean,
                'y_std': y_std,
            },
        }

        improved = (primary_higher_better and current_primary > best_primary_value) or \
                   (not primary_higher_better and current_primary < best_primary_value)

        if improved:
            best_primary_value = current_primary
            best_epoch = epoch + 1
            best_val_metrics = {
                'mae_raw': val_metrics['mae_raw'],
                'srcc': val_metrics['srcc'],
                'pearson': val_metrics['pearson'],
                ndcg_key: val_metrics[ndcg_key],
            }
            torch.save(checkpoint, exp_dir / "best_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(exp_dir / "best_model.pth", weights_only=False)
    state_dict = {k: v.float() for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)

    test_metrics = evaluate(
        model, test_loader, device, args.ndcg_k,
        y_mean=y_mean, y_std=y_std,
        gain_type=args.gain_type,
    )


if __name__ == "__main__":
    main()

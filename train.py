import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader import prep_dataloaders, denormalize, AVAILABLE_TARGETS
from models import get_model, NOISE_ENCODERS, TEXT_ENCODERS

from losses import (
    # Metrics
    ndcg_at_k,
    spearman_corrcoef,
    pearson_corrcoef,
    # Loss classes
    SRCCLoss,
    CombinedLoss,
    MAESRCCLoss,
    MAELambdaRankLoss,
    MAESRCCLambdaRankLoss,
    LambdaRankLoss,
)

# Target order for multi-head model (must match AVAILABLE_TARGETS order)
TARGET_ORDER = AVAILABLE_TARGETS  # ['hpsv2', 'image_reward', 'clip_score']



def get_loss_column_name(loss_type: str) -> str:
    column_names = {
        'mse': 'MSE Loss',
        'mae': 'MAE Loss',
        'mse+srcc': 'MSE+SRCC Loss',
        'lambdarank': '—',
        'mae+srcc': 'MAE+SRCC Loss',
        'mae+lambdarank': 'MAE+LambdaRank Loss',
        'mae+srcc+lambdarank': 'MAE+SRCC+LambdaRank Loss',
    }
    return column_names.get(loss_type, 'Loss')

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
    multi_target: bool = False,
    epoch: int = 0,
    loss_type: str = 'mse',
) -> Dict[str, float]:
    """
    Returns:
        Dict containing:
        - 'display_loss': Loss value to display (primary target only for multi-target,
                          visible components only for LambdaRank hybrids)
        - 'total_loss': Total loss across all targets (for reference)
        - prediction/target statistics
    """
    model.train()
    running_display_loss = 0.0
    running_total_loss = 0.0

    if multi_target:
        targetlists = {tgt: [] for tgt in TARGET_ORDER}
        predictionlists = {tgt: [] for tgt in TARGET_ORDER}
    else:
        targetlist = []
        predictionlist = []

    # Check if criterion uses LambdaRank (requires special backward handling)
    uses_lambdarank = isinstance(criterion, (LambdaRankLoss, MAELambdaRankLoss, MAESRCCLambdaRankLoss))
    is_pure_lambdarank = isinstance(criterion, LambdaRankLoss)

    # Debug: print info for first 2 batches of epoch 1
    debug_batches = (epoch == 0) and uses_lambdarank

    for batch_idx, batch in enumerate(loader):
        noise = batch['noise'].to(device)
        prompt_embeds = batch['prompt_embeds'].to(device)
        prompt_mask = batch['prompt_mask'].to(device)

        optimizer.zero_grad()
        preds = model(noise, prompt_embeds, prompt_mask)  # [B, 1] or [B, 3]

        if multi_target:
            # Extract targets for each head
            targets = torch.stack([
                batch[f'y_{tgt}'].to(device) for tgt in TARGET_ORDER
            ], dim=1)  # [B, 3]

            if is_pure_lambdarank:
                # Pure LambdaRank: manual gradient injection for each head
                # Must apply LambdaRank separately per head to avoid mixing tasks
                # Use retain_graph=True for all but the last head
                n_heads = len(TARGET_ORDER)
                for i, tgt in enumerate(TARGET_ORDER):
                    retain = (i < n_heads - 1)  # retain graph for all except last
                    criterion(preds[:, i:i+1], targets[:, i:i+1], retain_graph=retain)
                # No loss value to track for pure LambdaRank
                batch_display_loss = 0.0
                batch_total_loss = 0.0
            elif uses_lambdarank:
                # Combined loss with LambdaRank (MAE+LambdaRank or MAE+SRCC+LambdaRank)
                # The returned loss is only the visible component (MAE or MAE+SRCC)
                # IMPORTANT: Apply LambdaRank separately per head to avoid mixing tasks
                total_loss = torch.tensor(0.0, device=device)
                primary_loss = None
                for i, tgt in enumerate(TARGET_ORDER):
                    head_loss = criterion(preds[:, i:i+1], targets[:, i:i+1])
                    total_loss = total_loss + head_loss
                    if i == 0:  # Primary target (hpsv2)
                        primary_loss = head_loss

                # Backward for visible loss (MAE or MAE+SRCC) - retain graph for LambdaRank
                total_loss.backward(retain_graph=True)

                # Inject LambdaRank gradients separately for each head
                # Use retain_graph=True for all but the last head
                n_heads = len(TARGET_ORDER)
                if criterion.lambdarank_weight > 0:
                    for i, tgt in enumerate(TARGET_ORDER):
                        retain = (i < n_heads - 1)
                        criterion.lambdarank(
                            preds[:, i:i+1], targets[:, i:i+1],
                            weight=criterion.lambdarank_weight,
                            retain_graph=retain
                        )

                batch_display_loss = primary_loss.item()
                batch_total_loss = total_loss.item()
            else:
                # Standard loss (MSE, MAE, SRCC, etc.)
                total_loss = torch.tensor(0.0, device=device)
                primary_loss = torch.tensor(0.0, device=device)
                for i, tgt in enumerate(TARGET_ORDER):
                    head_loss = criterion(preds[:, i:i+1], targets[:, i:i+1])
                    total_loss = total_loss + head_loss
                    if i == 0:  # Primary target (hpsv2)
                        primary_loss = head_loss
                total_loss.backward()
                batch_display_loss = primary_loss.item()
                batch_total_loss = total_loss.item()

            optimizer.step()
            running_display_loss += batch_display_loss * noise.size(0)
            running_total_loss += batch_total_loss * noise.size(0)

            # Track per-head stats
            for i, tgt in enumerate(TARGET_ORDER):
                targetlists[tgt].extend(targets[:, i].cpu().numpy())
                predictionlists[tgt].extend(preds[:, i].detach().cpu().numpy())
        else:
            targets = batch['y'].to(device).unsqueeze(1)  # [B, 1]

            if is_pure_lambdarank:
                # Pure LambdaRank: manual gradient injection (no loss value returned)
                criterion(preds, targets)
                batch_display_loss = 0.0
                batch_total_loss = 0.0

            elif uses_lambdarank:
                # Combined loss with LambdaRank
                # The returned loss is only the visible component (MAE or MAE+SRCC)
                loss = criterion(preds, targets)
                criterion.backward(preds, targets, loss)
                batch_display_loss = loss.item()
                batch_total_loss = loss.item()
            else:
                # Standard loss
                loss = criterion(preds, targets)
                loss.backward()
                batch_display_loss = loss.item()
                batch_total_loss = loss.item()

            optimizer.step()
            running_display_loss += batch_display_loss * noise.size(0)
            running_total_loss += batch_total_loss * noise.size(0)
            targetlist.extend(targets.squeeze(1).cpu().numpy())
            predictionlist.extend(preds.squeeze(1).detach().cpu().numpy())

    n_samples = len(loader.dataset)
    result = {
        'display_loss': running_display_loss / n_samples,
        'total_loss': running_total_loss / n_samples,
        'loss': running_display_loss / n_samples,  # Backward compatibility
    }

    if multi_target:
        result['target_mean'] = float(np.mean(targetlists['hpsv2']))
        result['target_std'] = float(np.std(targetlists['hpsv2']))
        result['pred_mean'] = float(np.mean(predictionlists['hpsv2']))
        result['pred_std'] = float(np.std(predictionlists['hpsv2']))
    else:
        result['target_mean'] = float(np.mean(targetlist))
        result['target_std'] = float(np.std(targetlist))
        result['pred_mean'] = float(np.mean(predictionlist))
        result['pred_std'] = float(np.std(predictionlist))

    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ndcg_k: int = 10,
    multi_target: bool = False,
    target_stats: Dict = None,  # {target: {mean, std}} for multi-target
    y_mean: float = 0.0,        # For single-target
    y_std: float = 1.0,         # For single-target
    gain_type: str = 'exp2',    # Gain type for NDCG: 'exp2' or 'identity'
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.

    For single-target: returns metrics for that target
    For multi-target: returns metrics for each target with prefix (e.g., hpsv2_mae)

    Includes both global NDCG and per-prompt (query-level) NDCG.
    Uses exponential gain (2^rel - 1) by default for NDCG computation.
    """
    model.eval()

    if multi_target:
        # Track predictions/targets/prompt_ids per head
        all_preds_raw = {tgt: [] for tgt in TARGET_ORDER}
        all_targets_raw = {tgt: [] for tgt in TARGET_ORDER}
        all_prompt_ids = []

        for batch in loader:
            noise = batch['noise'].to(device)
            prompt_embeds = batch['prompt_embeds'].to(device)
            prompt_mask = batch['prompt_mask'].to(device)
            prompt_ids = torch.tensor(batch['prompt_id'], device=device)

            preds_norm = model(noise, prompt_embeds, prompt_mask)  # [B, 3]

            all_prompt_ids.append(prompt_ids)

            for i, tgt in enumerate(TARGET_ORDER):
                targets_raw = batch[f'raw_y_{tgt}'].to(device)
                mean = target_stats[tgt]['mean']
                std = target_stats[tgt]['std']
                preds_raw = denormalize(preds_norm[:, i], mean, std)

                all_preds_raw[tgt].append(preds_raw)
                all_targets_raw[tgt].append(targets_raw)

        # Concatenate prompt_ids
        all_prompt_ids = torch.cat(all_prompt_ids, dim=0)

        # Compute metrics for each target
        results = {}
        for tgt in TARGET_ORDER:
            preds = torch.cat(all_preds_raw[tgt], dim=0)
            targets = torch.cat(all_targets_raw[tgt], dim=0)

            mae = (preds - targets).abs().mean().item()

            if len(preds) > 1 and preds.std() > 1e-9:
                srcc = spearman_corrcoef(preds, targets).item()
                pearson = pearson_corrcoef(preds, targets).item()
                ndcg_global = ndcg_at_k(preds, targets, k=ndcg_k, gain_type=gain_type)
            else:
                srcc = 0.0
                pearson = 0.0
                ndcg_global = 0.0

            results[f'{tgt}_mae_raw'] = mae
            results[f'{tgt}_srcc'] = srcc
            results[f'{tgt}_pearson'] = pearson
            results[f'{tgt}_ndcg_{ndcg_k}'] = ndcg_global
            results[f'{tgt}_pred_mean'] = preds.mean().item()
            results[f'{tgt}_pred_std'] = preds.std().item()

        # Use hpsv2 as primary for backward compatibility in logging
        results['mae_raw'] = results['hpsv2_mae_raw']
        results['srcc'] = results['hpsv2_srcc']
        results['pearson'] = results['hpsv2_pearson']
        results[f'ndcg_{ndcg_k}'] = results[f'hpsv2_ndcg_{ndcg_k}']
        results['pred_mean'] = results['hpsv2_pred_mean']
        results['pred_std'] = results['hpsv2_pred_std']
        results['n_samples'] = len(all_prompt_ids)

        return results
    else:
        # Single-target mode
        all_preds_raw = []
        all_targets_raw = []

        for batch in loader:
            noise = batch['noise'].to(device)
            prompt_embeds = batch['prompt_embeds'].to(device)
            prompt_mask = batch['prompt_mask'].to(device)
            targets_raw = batch['raw_y'].to(device)

            preds_norm = model(noise, prompt_embeds, prompt_mask).squeeze(1)
            preds_raw = denormalize(preds_norm, y_mean, y_std)

            all_preds_raw.append(preds_raw)
            all_targets_raw.append(targets_raw)

        # Concatenate on GPU
        all_preds_raw = torch.cat(all_preds_raw, dim=0)
        all_targets_raw = torch.cat(all_targets_raw, dim=0)

        # Compute global metrics
        n_samples = len(all_preds_raw)
        mae_raw = (all_preds_raw - all_targets_raw).abs().mean().item()

        if n_samples > 1 and all_preds_raw.std() > 1e-9:
            srcc = spearman_corrcoef(all_preds_raw, all_targets_raw).item()
            pearson = pearson_corrcoef(all_preds_raw, all_targets_raw).item()
            ndcg = ndcg_at_k(all_preds_raw, all_targets_raw, k=ndcg_k, gain_type=gain_type)
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
    parser = argparse.ArgumentParser(description="Train PixArt Score Predictor")

    # Model architecture
    parser.add_argument('--noise_enc', type=str, default='basic_cnn_avg', choices=NOISE_ENCODERS,
                        help='Noise encoder type')
    parser.add_argument('--text_enc', type=str, default='compression_first', choices=TEXT_ENCODERS,
                        help='Text encoder type')

    # Training
    parser.add_argument('--target', type=str, default='hpsv2', choices=AVAILABLE_TARGETS)
    parser.add_argument('--split_by', type=str, default='prompt', choices=['prompt', 'seed'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=-1,
                        help='Training epochs (-1 for loss-specific defaults)')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mae', 'mse', 'mse+srcc', 'lambdarank', 'mae+srcc', 'mae+lambdarank', 'mae+srcc+lambdarank'],
                        help='Loss function: mae, mse, mse+srcc, lambdarank, mae+srcc, mae+lambdarank, mae+srcc+lambdarank')
    parser.add_argument('--gain_type', type=str, default='exp2', choices=['exp2', 'identity'],
                        help='Gain type for NDCG/LambdaRank: exp2 (2^rel-1) or identity (linear)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    # Evaluation
    parser.add_argument('--ndcg_k', type=int, default=10,
                        help='K for NDCG@K metric (top-K ranking quality)')

    # Multi-target mode
    parser.add_argument('--multi_target', action='store_true',
                        help='Train multi-head model on all 3 targets (hpsv2, image_reward, clip_score)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=-1,
                        help='Early stopping patience (-1 for loss-specific defaults)')

    args = parser.parse_args()

    # Set loss-specific epochs and patience defaults
    if args.epochs == -1 or args.patience == -1:
        if args.loss == 'lambdarank':
            default_epochs, default_patience = 125, 20
        elif args.loss in ['mse', 'mae']:
            default_epochs, default_patience = 100, 20
        else:  # hybrid losses: mae+srcc, mae+lambdarank, mae+srcc+lambdarank, mse+srcc
            default_epochs, default_patience = 150, 30

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

    # Enable TensorFloat32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx, A100, etc.)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"Experiment: {args.exp_name}")
    print(f"Noise encoder: {args.noise_enc}")
    print(f"Text encoder: {args.text_enc}")
    if args.multi_target:
        print(f"Targets: hpsv2, image_reward, clip_score (multi-head)")
    else:
        print(f"Target: {args.target}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    loss_info = args.loss
    if 'lambdarank' in args.loss:
        loss_info += f" (gain_type={args.gain_type})"
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Loss: {loss_info}")
    print(f"NDCG@K: {args.ndcg_k}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, stats = prep_dataloaders(
        target=args.target,
        split_by=args.split_by,
        batch_size=args.batch_size,
        load_attn=False,
        num_workers=args.num_workers,
        seed=args.seed,
        multi_target=args.multi_target,
    )

    # Get normalization stats
    if args.multi_target:
        target_stats = stats['target_stats']  # {target: {mean, std}}
    else:
        y_mean, y_std = stats['y_mean'], stats['y_std']

    # Create model
    num_heads = 3 if args.multi_target else 1
    model = get_model(
        noise_enc=args.noise_enc,
        text_enc=args.text_enc,
        dropout=args.dropout,
        num_heads=num_heads,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Select loss function
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mse+srcc':
        criterion = CombinedLoss(srcc_weight=1.0, regularization_strength=1e-2)
    elif args.loss == 'lambdarank':
        criterion = LambdaRankLoss(sigma=1.0, gain_type=args.gain_type)
    elif args.loss == 'mae+srcc':
        criterion = MAESRCCLoss(srcc_weight=1.0, regularization_strength=1e-2)
    elif args.loss == 'mae+lambdarank':
        criterion = MAELambdaRankLoss(lambdarank_weight=1.0, sigma=1.0, gain_type=args.gain_type)
    elif args.loss == 'mae+srcc+lambdarank':
        criterion = MAESRCCLambdaRankLoss(
            srcc_weight=1.0, lambdarank_weight=1.0,
            regularization_strength=1e-2, sigma=1.0, gain_type=args.gain_type
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # ---------------------------------------------------------
    # Checkpoint tracking: Constrained (SRCC >= threshold) + Fallback
    # ---------------------------------------------------------
    srcc_threshold = 0.60
    min_save_epoch = 5  # Skip noisy early epochs for constrained best

    # Constrained best: best NDCG where SRCC >= threshold
    best_constrained_ndcg = -1.0
    best_constrained_srcc = -1.0
    best_constrained_epoch = -1

    # Overall best: absolute best NDCG (fallback)
    best_overall_ndcg = -1.0
    best_overall_srcc = -1.0
    best_overall_epoch = -1

    # Legacy variables for summary (will be set to constrained best if available)
    best_val_ndcg = -1.0
    best_val_srcc = -1.0

    epochs_without_improvement = 0
    history = []

    ndcg_key = f'ndcg_{args.ndcg_k}'
    loss_column_name = get_loss_column_name(args.loss)
    is_lambdarank_only = (args.loss == 'lambdarank')

    # Calculate loss column width based on the loss name
    loss_col_width = max(10, len(loss_column_name))
    total_width = 5 + 3 + loss_col_width + 3 + 10 + 3 + 10 + 3 + 10 + 3 + 12 + 3 + 20 + 3 + 20  # columns + separators

    print(f"\n{'='*total_width}")
    print(f"{'Epoch':>5} | {loss_column_name:>{loss_col_width}} | {'Val MAE':>10} | {'Val SRCC':>10} | {'Val Pear':>10} | {f'Val NDCG@{args.ndcg_k}':>12} | "
          f"{'Train Pred':>20} | {'Val Pred':>20}")
    print(f"{'':>5} | {'':>{loss_col_width}} | {'':>10} | {'':>10} | {'':>10} | {'':>12} | "
          f"{'mean':>9} {'std':>9} | {'mean':>9} {'std':>9}")
    print(f"{'='*total_width}")

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            multi_target=args.multi_target,
            epoch=epoch,
            loss_type=args.loss,
        )
        if args.multi_target:
            val_metrics = evaluate(
                model, val_loader, device, args.ndcg_k,
                multi_target=True, target_stats=target_stats,
                gain_type=args.gain_type,
            )
        else:
            val_metrics = evaluate(
                model, val_loader, device, args.ndcg_k,
                multi_target=False, y_mean=y_mean, y_std=y_std,
                gain_type=args.gain_type,
            )

        current_ndcg = val_metrics[ndcg_key]
        current_srcc = val_metrics['srcc']

        # Schedule based on NDCG (primary metric)
        scheduler.step(current_ndcg)

        # ---------------------------------------------------------
        # Checkpoint saving logic
        # ---------------------------------------------------------
        is_best_overall = False
        is_best_constrained = False

        # Prepare checkpoint data (reused for both saves)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'noise_enc': args.noise_enc,
                'text_enc': args.text_enc,
                'dropout': args.dropout,
                'num_heads': num_heads,
            },
            'normalization': {
                'multi_target': args.multi_target,
                'target': args.target,
            },
        }
        if args.multi_target:
            checkpoint['normalization']['target_stats'] = target_stats
        else:
            checkpoint['normalization']['y_mean'] = y_mean
            checkpoint['normalization']['y_std'] = y_std

        # 1. Track and save global best NDCG (fallback checkpoint)
        if current_ndcg > best_overall_ndcg:
            best_overall_ndcg = current_ndcg
            best_overall_srcc = current_srcc
            best_overall_epoch = epoch + 1
            is_best_overall = True
            torch.save(checkpoint, exp_dir / "best_model_fallback.pth")

        # 2. Track and save constrained best (SRCC >= threshold, skip early epochs)
        if (epoch + 1) >= min_save_epoch and current_srcc >= srcc_threshold:
            if current_ndcg > best_constrained_ndcg:
                best_constrained_ndcg = current_ndcg
                best_constrained_srcc = current_srcc
                best_constrained_epoch = epoch + 1
                is_best_constrained = True
                torch.save(checkpoint, exp_dir / "best_model.pth")
                epochs_without_improvement = 0

                # Update legacy variables for summary
                best_val_ndcg = best_constrained_ndcg
                best_val_srcc = best_constrained_srcc

        # Patience logic: hybrid approach
        if is_best_constrained:
            # Found new constrained best, reset patience
            pass  # already reset above
        elif is_best_overall and best_constrained_epoch == -1:
            # Still exploring, no valid constrained checkpoint yet - be patient
            epochs_without_improvement = max(0, epochs_without_improvement - 1)
        else:
            epochs_without_improvement += 1

        # Build marker for logging
        if is_best_constrained:
            best_marker = " **"  # double star for constrained best
        elif is_best_overall:
            best_marker = " *"   # single star for overall best
        else:
            best_marker = ""

        # Format loss display: "—" for pure LambdaRank, number for others
        if is_lambdarank_only:
            loss_display = "—".center(loss_col_width)
        else:
            loss_display = f"{train_stats['display_loss']:>{loss_col_width}.6f}"

        print(f"{epoch+1:>5} | {loss_display} | {val_metrics['mae_raw']:>10.6f} | "
              f"{current_srcc:>10.4f} | {val_metrics['pearson']:>10.4f} | {current_ndcg:>12.4f} | "
              f"{train_stats['pred_mean']:>9.4f} {train_stats['pred_std']:>9.4f} | "
              f"{val_metrics['pred_mean']:>9.4f} {val_metrics['pred_std']:>9.4f}{best_marker}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_stats['loss']),
            'train_pred_mean': train_stats['pred_mean'],
            'train_pred_std': train_stats['pred_std'],
            'val_mae': float(val_metrics['mae_raw']),
            'val_srcc': float(current_srcc),
            'val_pearson': float(val_metrics['pearson']),
            f'val_{ndcg_key}': float(current_ndcg),
            'val_pred_mean': val_metrics['pred_mean'],
            'val_pred_std': val_metrics['pred_std'],
        })

        # Early stopping check
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
            break

    print(f"{'='*122}")

    # ---------------------------------------------------------
    # Checkpoint Summary
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print("CHECKPOINT SUMMARY")
    print(f"{'='*70}")
    print(f"Best Overall (fallback):  Epoch {best_overall_epoch:3d} | "
          f"NDCG={best_overall_ndcg:.4f} | SRCC={best_overall_srcc:.4f}")

    if best_constrained_epoch > 0:
        ndcg_drop = best_overall_ndcg - best_constrained_ndcg
        ndcg_drop_pct = 100 * ndcg_drop / best_overall_ndcg if best_overall_ndcg > 0 else 0
        print(f"Best Constrained:         Epoch {best_constrained_epoch:3d} | "
              f"NDCG={best_constrained_ndcg:.4f} | SRCC={best_constrained_srcc:.4f}")
        print(f"NDCG sacrifice for SRCC >= {srcc_threshold}: {ndcg_drop:.4f} ({ndcg_drop_pct:.1f}%)")
        print(f"\nUsing: best_model.pth (constrained)")
        use_fallback = False
    else:
        print(f"WARNING: No checkpoint achieved SRCC >= {srcc_threshold} after epoch {min_save_epoch}")
        print(f"\nUsing: best_model_fallback.pth (unconstrained)")
        use_fallback = True
        # Copy fallback to best_model.pth for consistency
        shutil.copy(exp_dir / "best_model_fallback.pth", exp_dir / "best_model.pth")
        best_val_ndcg = best_overall_ndcg
        best_val_srcc = best_overall_srcc
    print(f"{'='*70}")

    # Final evaluation on test set
    print(f"\nLoading best model (Val NDCG@{args.ndcg_k}: {best_val_ndcg:.4f}, Val SRCC: {best_val_srcc:.4f})...")
    checkpoint = torch.load(exp_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.multi_target:
        test_metrics = evaluate(
            model, test_loader, device, args.ndcg_k,
            multi_target=True, target_stats=target_stats,
            gain_type=args.gain_type,
        )
    else:
        test_metrics = evaluate(
            model, test_loader, device, args.ndcg_k,
            multi_target=False, y_mean=y_mean, y_std=y_std,
            gain_type=args.gain_type,
        )

    print(f"\n{'='*70}")
    print(f"TEST RESULTS (N={test_metrics.get('n_samples', 'N/A')} samples)")
    print(f"{'='*70}")

    if args.multi_target:
        # Print results for all targets
        for tgt in TARGET_ORDER:
            print(f"\n  [{tgt.upper()}]")
            print(f"    MAE:      {test_metrics[f'{tgt}_mae_raw']:.6f}")
            print(f"    SRCC:     {test_metrics[f'{tgt}_srcc']:.4f}")
            print(f"    Pearson:  {test_metrics[f'{tgt}_pearson']:.4f}")
            print(f"    NDCG@{args.ndcg_k}:  {test_metrics[f'{tgt}_ndcg_{args.ndcg_k}']:.4f}")
    else:
        print(f"\n  MAE:      {test_metrics['mae_raw']:.6f}")
        print(f"  SRCC:     {test_metrics['srcc']:.4f}")
        print(f"  Pearson:  {test_metrics['pearson']:.4f}")
        print(f"  NDCG@{args.ndcg_k}:  {test_metrics[ndcg_key]:.4f}")
        print(f"\n  Pred:   mean={test_metrics['pred_mean']:.4f}, std={test_metrics['pred_std']:.4f}")
        print(f"  Target: mean={test_metrics['target_mean']:.4f}, std={test_metrics['target_std']:.4f}")

    print(f"{'='*70}")

    summary = {
        'experiment': args.exp_name,
        'noise_enc': args.noise_enc,
        'text_enc': args.text_enc,
        'target': args.target if not args.multi_target else 'multi',
        'multi_target': args.multi_target,
        'loss': args.loss,
        'gain_type': args.gain_type,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'params': param_count,
        # Checkpoint info
        'srcc_threshold': srcc_threshold,
        'min_save_epoch': min_save_epoch,
        'best_overall_epoch': best_overall_epoch,
        'best_overall_ndcg': float(best_overall_ndcg),
        'best_overall_srcc': float(best_overall_srcc),
        'best_constrained_epoch': best_constrained_epoch,
        'best_constrained_ndcg': float(best_constrained_ndcg) if best_constrained_epoch > 0 else None,
        'best_constrained_srcc': float(best_constrained_srcc) if best_constrained_epoch > 0 else None,
        'used_fallback': use_fallback,
        # Legacy fields (point to the selected checkpoint)
        f'best_val_{ndcg_key}': float(best_val_ndcg),
        'best_val_srcc': float(best_val_srcc),
    }

    if args.multi_target:
        # Add metrics for each target
        for tgt in TARGET_ORDER:
            summary[f'{tgt}_test_mae'] = round(float(test_metrics[f'{tgt}_mae_raw']), 6)
            summary[f'{tgt}_test_srcc'] = round(float(test_metrics[f'{tgt}_srcc']), 6)
            summary[f'{tgt}_test_pearson'] = round(float(test_metrics[f'{tgt}_pearson']), 6)
            summary[f'{tgt}_test_{ndcg_key}'] = round(float(test_metrics[f'{tgt}_ndcg_{args.ndcg_k}']), 6)
        # Primary target metrics (hpsv2 for backward compatibility)
        summary['test_mae'] = summary['hpsv2_test_mae']
        summary['test_srcc'] = summary['hpsv2_test_srcc']
        summary['test_pearson'] = summary['hpsv2_test_pearson']
        summary[f'test_{ndcg_key}'] = summary[f'hpsv2_test_{ndcg_key}']
    else:
        summary['test_mae'] = round(float(test_metrics['mae_raw']), 6)
        summary['test_srcc'] = round(float(test_metrics['srcc']), 6)
        summary['test_pearson'] = round(float(test_metrics['pearson']), 6)
        summary[f'test_{ndcg_key}'] = round(float(test_metrics[ndcg_key]), 6)

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()

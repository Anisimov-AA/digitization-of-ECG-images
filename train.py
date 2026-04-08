# === train.py ===
#
# Training loop for ECG digitization model.
#
# What it does:
#   1. Loads data and splits into train/val by fold
#   2. Creates model, optimizer, scheduler
#   3. Trains for N epochs with mixed precision (fp16)
#   4. Validates every epoch by computing competition SNR
#   5. Saves best checkpoint to Google Drive
#
# Usage from Colab:
#   from train import run_training
#   run_training()
#
# Or from command line:
#   python train.py
#
# To resume training from a checkpoint:
#   run_training(resume='/path/to/checkpoint.pth')

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm.auto import tqdm

from config import CFG
from dataset import ECGRowDataset
from model import ECGRowNet
from loss import combined_loss
from metrics import compute_snr_for_record


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scaler):
    """Train for one epoch, return average loss."""
    model.train()
    losses = []

    pbar = tqdm(loader, desc='Train')
    for step, batch in enumerate(pbar):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred_mv, pred_prob = model(batch)
            gt_mv = batch['gt_mv'].to(CFG.device)
            gt_mask = batch['mask'].to(CFG.device)
            loss, l_jsd, l_snr = combined_loss(pred_mv, pred_prob, gt_mv, gt_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{np.mean(losses[-20:]):.4f}'})

    return np.mean(losses)


def validate(model, ds, fold_df_val):
    """
    Validate by computing competition SNR.

    Collects predictions for all 4 rows of each image,
    then computes SNR per record and returns the mean.
    """
    model.eval()
    snrs = []

    # Group samples by image
    num_images = len(fold_df_val)
    for img_idx in range(num_images):
        sid = str(fold_df_val.iloc[img_idx]['id'])
        ecg_path = f'{CFG.data_dir}/ecg_csv/{sid}.csv'

        pred_rows = {}
        for row_idx in range(4):
            sample_idx = img_idx * 4 + row_idx
            sample = ds[sample_idx]
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                     for k, v in sample.items()}

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred_mv, _ = model(batch)

            pred_rows[row_idx] = pred_mv[0, 0].float().cpu().numpy()

        snr = compute_snr_for_record(pred_rows, ecg_path)
        snrs.append(snr)

    return np.mean(snrs)


def run_training(resume=None):
    """Main training function."""
    set_seed(CFG.seed)

    # --- Data ---
    fold_df = pd.read_csv(
        f'{CFG.data_dir}/train_fold.csv',
        dtype={'id': str, 'type_id': str}
    )

    train_df = fold_df[fold_df['fold'] != CFG.fold].reset_index(drop=True)
    val_df = fold_df[fold_df['fold'] == CFG.fold].reset_index(drop=True)
    print(f'Train: {len(train_df)} images ({len(train_df)*4} rows)')
    print(f'Val:   {len(val_df)} images ({len(val_df)*4} rows)')

    train_ds = ECGRowDataset(train_df, CFG, is_train=True)
    val_ds = ECGRowDataset(val_df, CFG, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
    )

    # --- Model ---
    model = ECGRowNet(CFG).to(CFG.device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=1e-6
    )
    scaler = GradScaler('cuda')

    # --- Resume ---
    start_epoch = 0
    best_snr = -999
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=CFG.device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        best_snr = ckpt.get('snr', -999)
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}, SNR={best_snr:.2f} dB')

    # --- Training loop ---
    os.makedirs(CFG.save_dir, exist_ok=True)

    for epoch in range(start_epoch, CFG.epochs):
        print(f'\n--- Epoch {epoch+1}/{CFG.epochs} ---')

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        scheduler.step()

        val_snr = validate(model, val_ds, val_df)
        lr = optimizer.param_groups[0]['lr']
        print(f'Loss={train_loss:.4f}, Val SNR={val_snr:.2f} dB, LR={lr:.2e}')

        # Save best
        if val_snr > best_snr:
            best_snr = val_snr
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'snr': best_snr,
            }, f'{CFG.save_dir}/best_fold{CFG.fold}.pth')
            print(f'  -> New best! {best_snr:.2f} dB')

        # Save latest
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'snr': val_snr,
        }, f'{CFG.save_dir}/last_fold{CFG.fold}.pth')

    print(f'\nDone! Best Val SNR: {best_snr:.2f} dB')
    return best_snr


if __name__ == '__main__':
    run_training()
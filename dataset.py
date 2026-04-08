# === dataset.py ===
#
# Handles loading and preparing data for training.
#
# Each rectified ECG image (1700 x 5600) contains 4 signal rows.
# Instead of feeding the whole image to the model, we crop each
# row separately into a 480 x 5000 strip centered on its baseline.
# This makes the input 4x smaller and the task much simpler —
# the model only needs to find one trace per crop.
#
# For each crop we also extract the ground-truth y-position
# of the signal from the precomputed mask, and convert it to
# millivolts so we can compute the competition metric.
#
# Augmentations:
#   - Horizontal flip (the signal is valid in both directions)
#   - More can be added here later without touching other files

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CFG


def load_sparse_mask(filepath):
    """Load a precomputed signal mask stored in sparse COO format."""
    d = np.load(filepath)
    shape = tuple(d['shape'])
    mask = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        mask[i, d[f'ch{i}_y'], d[f'ch{i}_x']] = d[f'ch{i}_v']
    return mask


def crop_row(img, mask_channel, row_idx, cfg):
    """
    Crop a strip around one signal row.

    Takes the full rectified image and one channel of the mask,
    cuts out a 480px tall region centered on the row's baseline,
    and extracts only the signal columns (301–5300).

    Returns:
        row_img: (crop_h, signal_w, 3) uint8 image crop
        row_mask: (crop_h, signal_w) float32 mask crop
    """
    half = cfg.crop_h // 2
    baseline = int(cfg.zero_mv[row_idx])
    h0 = baseline - half
    h1 = baseline + half

    # Handle edge cases where crop goes outside image
    H = img.shape[0]
    src_h0, src_h1 = max(0, h0), min(H, h1)
    dst_h0 = src_h0 - h0
    dst_h1 = dst_h0 + (src_h1 - src_h0)

    # Create padded crop (white background)
    row_img = np.full((cfg.crop_h, cfg.signal_w, 3), 255, dtype=np.uint8)
    row_mask = np.zeros((cfg.crop_h, cfg.signal_w), dtype=np.float32)

    # Fill with actual data from signal region columns
    row_img[dst_h0:dst_h1] = img[src_h0:src_h1, cfg.signal_t0:cfg.signal_t1]
    row_mask[dst_h0:dst_h1] = mask_channel[src_h0:src_h1, cfg.signal_t0:cfg.signal_t1]

    return row_img, row_mask


class ECGRowDataset(Dataset):
    """
    Dataset that yields individual row crops.

    Each ECG image produces 4 samples (one per signal row).
    So 977 images become 3908 training samples.
    Each sample is a (480 x 5000) image crop paired with
    the ground-truth signal mask for that row.
    """

    def __init__(self, fold_df, cfg, is_train=True):
        self.df = fold_df.reset_index(drop=True)
        self.cfg = cfg
        self.is_train = is_train

        # Build list of (image_index, row_index) pairs
        self.samples = []
        for idx in range(len(self.df)):
            for row_idx in range(4):
                self.samples.append((idx, row_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, row_idx = self.samples[idx]
        row = self.df.iloc[img_idx]
        sid = str(row['id'])
        type_id = str(row['type_id'])
        cfg = self.cfg

        # Load full rectified image
        img_path = f'{cfg.data_dir}/rectified/{sid}-{type_id}.rect.png'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)

        # Load mask
        mask_path = f'{cfg.data_dir}/masks/{sid}.mask-coo.npz'
        full_mask = load_sparse_mask(mask_path)

        # Crop this row
        row_img, row_mask = crop_row(img, full_mask[row_idx], row_idx, cfg)

        # --- Augmentation ---
        if self.is_train and np.random.random() > 0.5:
            row_img = np.fliplr(row_img).copy()
            row_mask = np.fliplr(row_mask).copy()

        # --- Extract ground-truth y-position from mask ---
        # For each column, compute weighted mean of y-coordinates
        y_coords = np.arange(cfg.crop_h, dtype=np.float32).reshape(-1, 1)
        col_sum = row_mask.sum(axis=0) + 1e-8
        gt_y = (row_mask * y_coords).sum(axis=0) / col_sum  # (signal_w,)

        # Convert y-position to millivolts
        # In the crop, baseline is at crop_h // 2
        half = cfg.crop_h // 2
        gt_mv = (half - gt_y) / cfg.mv_to_pixel  # (signal_w,)

        # Convert to tensors
        row_img = torch.from_numpy(row_img.transpose(2, 0, 1)).float() / 255.0
        row_mask = torch.from_numpy(row_mask).unsqueeze(0).float()
        gt_mv = torch.from_numpy(gt_mv).unsqueeze(0).float()

        return {
            'image': row_img,           # (3, crop_h, signal_w)
            'mask': row_mask,           # (1, crop_h, signal_w)
            'gt_mv': gt_mv,            # (1, signal_w)
            'sid': sid,
            'row_idx': row_idx,
            'ecg_path': f'{cfg.data_dir}/ecg_csv/{sid}.csv',
        }
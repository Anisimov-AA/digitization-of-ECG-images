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


def to_grayscale_3ch(img):
    """
    Convert RGB to grayscale, then copy to 3 channels.
    Removes color as noise source (grid colors vary across image types)
    while keeping 3-channel input for pretrained encoder compatibility.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


def augment_image(img):
    """
    Apply random augmentations to simulate image degradation.
    Each augmentation is applied independently with 50% probability.
    """
    img = img.astype(np.float32)

    # Random brightness/gamma (like 1st place: gamma 0.9/1.0/1.1)
    if np.random.random() > 0.5:
        gamma = np.random.uniform(0.8, 1.2)
        img = np.power(img / 255.0, gamma) * 255.0

    # Random Gaussian blur (simulates camera out of focus)
    if np.random.random() > 0.5:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Random noise (simulates scanning/photo noise)
    if np.random.random() > 0.5:
        noise = np.random.normal(0, np.random.uniform(2, 8), img.shape)
        img = img + noise

    # Random contrast
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        mean = img.mean()
        img = (img - mean) * factor + mean

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def make_adaptive_mask(row_mask, gt_y, cfg):
    """
    Create mask with adaptive sigma for sharp peaks.
    Where the signal changes rapidly (sharp peaks), use wider Gaussian
    so the model gets a softer target that's easier to learn.
    From 3rd place solution.
    """
    sigma_base = 2.0
    adaptive_factor = 0.4

    # Compute local change in y-position
    local_diff = 0.5 * (
        np.abs(gt_y - np.roll(gt_y, 1)) +
        np.abs(gt_y - np.roll(gt_y, -1))
    )
    # Fix edges
    local_diff[0] = local_diff[1]
    local_diff[-1] = local_diff[-2]

    # Adaptive sigma: wider where signal changes fast
    sigma_arr = sigma_base + adaptive_factor * np.maximum(local_diff - 3 * sigma_base, 0) / 3.0

    # Rebuild mask with adaptive sigma
    H = cfg.crop_h
    W = len(gt_y)
    adaptive_mask = np.zeros((H, W), dtype=np.float32)
    y_coords = np.arange(H, dtype=np.float32)

    for col in range(W):
        sigma = sigma_arr[col]
        adaptive_mask[:, col] = np.exp(-0.5 * ((y_coords - gt_y[col]) / sigma) ** 2)

    # Normalize per column
    col_sum = adaptive_mask.sum(axis=0, keepdims=True) + 1e-8
    adaptive_mask = adaptive_mask / col_sum

    return adaptive_mask


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

        # Convert to grayscale (3 channels for encoder compatibility)
        img = to_grayscale_3ch(img)

        # Load mask
        mask_path = f'{cfg.data_dir}/masks/{sid}.mask-coo.npz'
        full_mask = load_sparse_mask(mask_path)

        # Crop this row
        row_img, row_mask = crop_row(img, full_mask[row_idx], row_idx, cfg)

        # --- Augmentation (training only) ---
        if self.is_train:
            # Horizontal flip
            if np.random.random() > 0.5:
                row_img = np.fliplr(row_img).copy()
                row_mask = np.fliplr(row_mask).copy()

            # Image augmentations
            row_img = augment_image(row_img)

        # --- Extract ground-truth y-position from mask ---
        y_coords = np.arange(cfg.crop_h, dtype=np.float32).reshape(-1, 1)
        col_sum = row_mask.sum(axis=0) + 1e-8
        gt_y = (row_mask * y_coords).sum(axis=0) / col_sum  # (signal_w,)

        # --- Build adaptive sigma mask ---
        if self.is_train and cfg.use_adaptive_sigma:
            row_mask = make_adaptive_mask(row_mask, gt_y, cfg)

        # Convert y-position to millivolts
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
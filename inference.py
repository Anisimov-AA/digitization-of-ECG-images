# Runs our trained model on rectified test images.
# Takes rectified images from Stage 0+1 (hengck23),
# crops each of the 4 rows, predicts millivolt values,
# and saves the result as .npy files.
#
# Used in the Kaggle submission notebook after Stage 0+1.

import os
import cv2
import numpy as np
import torch
from scipy import signal as scipy_signal


# ECG constants (same as training)
ZERO_MV = [703.5, 987.5, 1271.5, 1531.5]
MV_TO_PIXEL = 79.0
CROP_H = 480
SIGNAL_T0 = 301
SIGNAL_T1 = 5301
SIGNAL_W = 5000
IMG_W = 5600
IMG_H = 1700

ROW_TO_LEADS = [
    ['I', 'aVR', 'V1', 'V4'],
    ['II', 'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]


def crop_row_inference(img, row_idx):
    """Crop one signal row from rectified image."""
    half = CROP_H // 2
    baseline = int(ZERO_MV[row_idx])
    h0 = baseline - half
    h1 = baseline + half

    H = img.shape[0]
    src_h0, src_h1 = max(0, h0), min(H, h1)
    dst_h0 = src_h0 - h0
    dst_h1 = dst_h0 + (src_h1 - src_h0)

    row_img = np.full((CROP_H, SIGNAL_W, 3), 255, dtype=np.uint8)
    row_img[dst_h0:dst_h1] = img[src_h0:src_h1, SIGNAL_T0:SIGNAL_T1]
    return row_img


def predict_image(model, rect_path, device, use_tta=True):
    """
    Predict mV values for all 4 rows of a rectified ECG image.

    Args:
        model: trained ECGRowNet
        rect_path: path to rectified .png image
        device: cuda device
        use_tta: if True, average with horizontal flip prediction

    Returns:
        series: (4, SIGNAL_W) array of mV values
    """
    img = cv2.imread(rect_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    half = CROP_H // 2
    series = np.zeros((4, SIGNAL_W), dtype=np.float32)

    for row_idx in range(4):
        row_img = crop_row_inference(img, row_idx)

        # Original prediction
        tensor = torch.from_numpy(row_img.transpose(2, 0, 1)).float().unsqueeze(0)
        batch = {'image': tensor}

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_mv, _ = model(batch)
        pred = pred_mv[0, 0].float().cpu().numpy()

        if use_tta:
            # Horizontal flip TTA
            row_flipped = np.fliplr(row_img).copy()
            tensor_flip = torch.from_numpy(row_flipped.transpose(2, 0, 1)).float().unsqueeze(0)
            batch_flip = {'image': tensor_flip}

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred_mv_flip, _ = model(batch_flip)
            pred_flip = pred_mv_flip[0, 0].float().cpu().numpy()[::-1]

            # Average original and flipped
            pred = (pred + pred_flip) / 2.0

        series[row_idx] = pred

    return series


def series_to_leads(series, lead_lengths):
    """
    Split 4 row predictions into 12 individual leads.

    Args:
        series: (4, signal_w) mV predictions
        lead_lengths: dict {lead_name: number_of_rows} from test.csv

    Returns:
        dict {lead_name: array of mV values}
    """
    result = {}

    for row_idx in range(3):
        leads = ROW_TO_LEADS[row_idx]
        lengths = [lead_lengths[lead] for lead in leads]

        # Lead II in row 1 gets remaining length
        if leads[0] == 'II':
            lengths[0] = lengths[0] - sum(lengths[1:])

        # Resample row to total length then split
        total_len = sum(lengths)
        row_resampled = scipy_signal.resample(series[row_idx], total_len).astype(np.float32)

        idx = np.cumsum(lengths)[:-1]
        splits = np.split(row_resampled, idx)
        for lead, s in zip(leads, splits):
            result[lead] = s

    # Row 3: Lead II full
    result['II'] = scipy_signal.resample(series[3], lead_lengths['II']).astype(np.float32)

    return result


def run_inference(model, test_df, rect_dir, out_dir, device, use_tta=True):
    """
    Run inference on all test images.

    Args:
        model: trained ECGRowNet in eval mode
        test_df: test.csv dataframe
        rect_dir: directory with rectified images
        out_dir: directory to save .npy predictions
        device: cuda device
        use_tta: use horizontal flip TTA
    """
    os.makedirs(out_dir, exist_ok=True)
    sample_ids = test_df['id'].astype(str).unique()

    for n, sid in enumerate(sample_ids):
        rect_path = f'{rect_dir}/{sid}.rect.png'

        if not os.path.exists(rect_path):
            print(f'  SKIP {sid}: no rectified image')
            continue

        series = predict_image(model, rect_path, device, use_tta)
        np.save(f'{out_dir}/{sid}.series.npy', series)

        if (n + 1) % 100 == 0 or n < 3:
            print(f'  {n+1}/{len(sample_ids)} done')

    print(f'  Inference complete: {len(sample_ids)} images')
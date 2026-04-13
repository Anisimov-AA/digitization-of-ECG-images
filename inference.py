# Runs our trained model on rectified test images.
# Takes rectified images from Stage 0+1 (hengck23),
# crops each of the 4 rows, predicts millivolt values,
# and saves the result as .npy files.
#
# Improvements over baseline:
#   1. TTA (horizontal flip) — averages original + flipped prediction
#   2. Split-then-resample — splits row into leads first, resamples each separately
#   3. Einthoven's Law correction — enforces II=I+III and aVR+aVL+aVF=0

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
    FIXED: splits at pixel level first, then resamples each lead separately.
    This avoids boundary artifacts from resampling concatenated signals.

    Args:
        series: (4, SIGNAL_W) mV predictions
        lead_lengths: dict {lead_name: number_of_rows} from test.csv

    Returns:
        dict {lead_name: array of mV values}
    """
    result = {}
    quarter = SIGNAL_W // 4  # 1250 pixels per lead

    for row_idx in range(3):
        leads = ROW_TO_LEADS[row_idx]
        for i, lead in enumerate(leads):
            # Split at pixel level — each lead gets exactly 1/4 of the row
            lead_pixels = series[row_idx, i * quarter:(i + 1) * quarter]
            target_len = lead_lengths[lead]

            # For short II in row 1, use same target as other leads
            if lead == 'II' and row_idx == 1:
                target_len = lead_lengths['I']

            # Resample each lead individually
            result[lead] = scipy_signal.resample(
                lead_pixels, target_len
            ).astype(np.float32)

    # Lead II full rhythm strip
    ii_len = lead_lengths['II']
    result['II'] = scipy_signal.resample(
        series[3], ii_len
    ).astype(np.float32)

    return result


def apply_einthoven(leads):
    """
    Apply Einthoven's Law to correct lead predictions.
    Physical constraints of ECG:
      - II = I + III  (Einthoven's triangle)
      - aVR + aVL + aVF = 0  (Goldberger leads)

    Only applied when predictions are already close to satisfying
    the constraint — otherwise the correction could make things worse.
    """
    # Check if II ≈ I + III
    if 'I' in leads and 'II' in leads and 'III' in leads:
        # Need same length — use shortest
        min_len = min(len(leads['I']), len(leads['II']), len(leads['III']))
        I = leads['I'][:min_len]
        II = leads['II'][:min_len]
        III = leads['III'][:min_len]

        error = np.abs(II - I - III).mean()
        if error < 0.1:  # close enough to correct
            # Average: enforce II = I + III
            corrected_II = (II + I + III) / 2.0
            corrected_sum = I + III
            # Blend: 50% original, 50% corrected
            leads['II'][:min_len] = (II + corrected_sum) / 2.0

    # Check if aVR + aVL + aVF ≈ 0
    if 'aVR' in leads and 'aVL' in leads and 'aVF' in leads:
        min_len = min(len(leads['aVR']), len(leads['aVL']), len(leads['aVF']))
        aVR = leads['aVR'][:min_len]
        aVL = leads['aVL'][:min_len]
        aVF = leads['aVF'][:min_len]

        error = np.abs(aVR + aVL + aVF).mean()
        if error < 0.1:
            # Subtract average offset to enforce sum = 0
            offset = (aVR + aVL + aVF) / 3.0
            leads['aVR'][:min_len] -= offset
            leads['aVL'][:min_len] -= offset
            leads['aVF'][:min_len] -= offset

    return leads
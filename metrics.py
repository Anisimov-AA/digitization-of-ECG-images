# === metrics.py ===
#
# Computes the competition SNR metric on validation data.
#
# During training, the model predicts 4 separate row crops
# per ECG image. To compute the competition metric, we need to:
#   1. Collect all 4 row predictions for the same image
#   2. Resample each row from 5000 pixels to the original
#      signal length (varies by sampling frequency)
#   3. Compute SNR across all 12 leads combined
#
# The competition allows ±0.2 second time shift and removes
# vertical offset before scoring. We don't implement that here
# (it would only improve our numbers), so our validation SNR
# is a conservative lower bound of the actual leaderboard score.
#
# FFT resampling is used (not linear interpolation) because it
# preserves signal fidelity much better, as shown by the 2nd
# place solution.

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from config import CFG

# Which leads appear in which row
ROW_TO_LEADS = [
    ['I', 'aVR', 'V1', 'V4'],
    ['II', 'aVL', 'V2', 'V5'],
    ['III', 'aVF', 'V3', 'V6'],
]


def fft_resample(x, num):
    """Resample signal x to length num using FFT method."""
    return scipy_signal.resample(x, num).astype(np.float32)


def compute_snr_for_record(pred_rows, ecg_path):
    """
    Compute competition-style SNR for one ECG record.

    Args:
        pred_rows: dict {0: array, 1: array, 2: array, 3: array}
                   Each array is (5000,) predicted mV values for that row.
        ecg_path:  path to ground-truth CSV file

    Returns:
        SNR in decibels (higher = better)
    """
    ecg_df = pd.read_csv(ecg_path)

    total_signal = 0.0
    total_noise = 0.0

    # Rows 0-2: each has 4 leads concatenated across 2.5s segments
    for row_idx in range(3):
        leads = ROW_TO_LEADS[row_idx]
        gt_parts = []
        for lead in leads:
            vals = ecg_df[lead].dropna().values.astype(np.float32)
            # Lead II in row 1 uses only first 2.5 seconds
            if lead == 'II' and row_idx == 1:
                ii_short = len(ecg_df['I'].dropna())
                vals = vals[:ii_short]
            gt_parts.append(vals)
        gt_concat = np.concatenate(gt_parts)

        pred_row = pred_rows[row_idx]
        if len(pred_row) != len(gt_concat):
            pred_row = fft_resample(pred_row, len(gt_concat))

        total_signal += (gt_concat ** 2).sum()
        total_noise += ((pred_row - gt_concat) ** 2).sum()

    # Row 3: Lead II full 10 seconds
    lead_ii = ecg_df['II'].dropna().values.astype(np.float32)
    pred_ii = pred_rows[3]
    if len(pred_ii) != len(lead_ii):
        pred_ii = fft_resample(pred_ii, len(lead_ii))

    total_signal += (lead_ii ** 2).sum()
    total_noise += ((pred_ii - lead_ii) ** 2).sum()

    eps = 1e-7
    snr_db = 10 * np.log10(total_signal / (total_noise + eps) + eps)
    return snr_db
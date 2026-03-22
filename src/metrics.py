import numpy as np


def compute_snr_single_lead(pred, gt, fs, max_shift_sec=0.2):
    """Compute signal and noise power for a single lead with alignment."""
    max_shift = int(max_shift_sec * fs)

    # Time alignment via cross-correlation
    if max_shift > 0:
        corr = np.correlate(gt, pred, mode="full")
        mid = len(gt) - 1
        search = corr[mid - max_shift : mid + max_shift + 1]
        best_shift = np.argmax(search) - max_shift

        if best_shift > 0:
            pred_aligned = pred[best_shift:]
            gt_aligned = gt[: len(pred_aligned)]
        elif best_shift < 0:
            gt_aligned = gt[-best_shift:]
            pred_aligned = pred[: len(gt_aligned)]
        else:
            pred_aligned, gt_aligned = pred, gt
    else:
        pred_aligned, gt_aligned = pred, gt

    min_len = min(len(pred_aligned), len(gt_aligned))
    pred_aligned = pred_aligned[:min_len]
    gt_aligned = gt_aligned[:min_len]

    # Vertical offset removal
    offset = np.mean(gt_aligned) - np.mean(pred_aligned)
    pred_aligned = pred_aligned + offset

    signal_power = np.sum(gt_aligned ** 2)
    noise_power = np.sum((gt_aligned - pred_aligned) ** 2)

    return signal_power, noise_power


def compute_snr_record(pred_by_lead, gt_by_lead, fs):
    """Compute SNR (dB) for one ECG record, summing across all 12 leads."""
    total_signal = 0.0
    total_noise = 0.0

    for lead in pred_by_lead:
        if lead in gt_by_lead:
            sp, np_ = compute_snr_single_lead(
                pred_by_lead[lead], gt_by_lead[lead], fs
            )
            total_signal += sp
            total_noise += np_

    if total_noise < 1e-12:
        return 60.0

    return 10 * np.log10(total_signal / total_noise)


def aggregate_snr(snr_list):
    """Competition metric: mean in linear domain, then convert to dB."""
    linear = [10 ** (s / 10) for s in snr_list]
    return 10 * np.log10(np.mean(linear))
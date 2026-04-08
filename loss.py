# === loss.py ===
#
# Two loss functions that work together to train the model.
#
# JSD loss (Jensen-Shannon Divergence):
#   Works in pixel space. Compares the predicted probability
#   distribution with the ground-truth mask column by column.
#   This is the main teacher — it tells the model WHERE the
#   signal is in the image. Top solutions found JSD >> BCE
#   for this column-wise distribution task.
#
# SNR loss (Signal-to-Noise Ratio):
#   Works in millivolt space. Compares predicted mV values
#   against ground truth mV. This is a secondary teacher
#   that directly optimizes the competition metric.
#
# Combined loss = JSD + 0.1 * SNR
#   JSD is weighted more because it provides stable gradients
#   for learning. SNR is noisy early in training when predictions
#   are far off, so we keep its weight small.
#
# To experiment with different loss combinations,
# only this file needs to change.

import torch


def jsd_loss(pred_prob, target_mask):
    """
    Column-wise Jensen-Shannon Divergence.

    For each column, both pred_prob and target_mask are treated
    as probability distributions over the vertical axis.
    JSD measures how different these two distributions are.
    JSD = 0 means identical, JSD = ln(2) means maximally different.

    Args:
        pred_prob:   (B, 1, H, W) predicted distribution from soft-argmax
        target_mask: (B, 1, H, W) ground-truth mask (sparse, will be normalized)
    Returns:
        scalar loss value
    """
    eps = 1e-8

    # Normalize both to proper distributions per column
    p = pred_prob + eps
    p = p / p.sum(dim=2, keepdim=True)

    q = target_mask + eps
    q = q / q.sum(dim=2, keepdim=True)

    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = (p+q)/2
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=2)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=2)
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd.mean()


def snr_loss(pred_mv, target_mv):
    """
    Negative SNR in decibels.

    Measures how much of the true signal power is preserved
    vs how much error (noise) the prediction introduces.
    Returns negative because we minimize loss but want to
    maximize SNR.

    Args:
        pred_mv:   (B, 1, W) predicted millivolts
        target_mv: (B, 1, W) ground-truth millivolts
    Returns:
        scalar loss value (negative dB, lower = better predictions)
    """
    eps = 1e-7
    signal_power = (target_mv ** 2).sum()
    noise_power = ((pred_mv - target_mv) ** 2).sum()
    snr = signal_power / (noise_power + eps)
    snr_db = 10 * torch.log10(snr + eps)
    return -snr_db


def combined_loss(pred_mv, pred_prob, gt_mv, gt_mask):
    """
    Combined JSD + SNR loss.

    Args:
        pred_mv:   (B, 1, W) predicted millivolts from soft-argmax
        pred_prob: (B, 1, H, W) predicted probability distribution
        gt_mv:     (B, 1, W) ground-truth millivolts
        gt_mask:   (B, 1, H, W) ground-truth mask
    Returns:
        total_loss, jsd_value, snr_value (for logging)
    """
    l_jsd = jsd_loss(pred_prob, gt_mask)
    l_snr = snr_loss(pred_mv, gt_mv)
    total = l_jsd + 0.1 * l_snr
    return total, l_jsd, l_snr
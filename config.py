# This is the single source of truth for all settings.
# Every other file reads from here, so you never need to
# hunt through code to change a number.
#
# Sections:
#   - Paths: where data lives and where checkpoints get saved
#   - Fold: which split of data to use for validation
#   - Image layout: the geometry of a rectified ECG image
#   - ECG physics: how pixel positions map to millivolts
#   - Model: which backbone and decoder to use
#   - Training: learning rate, batch size, epochs, etc.

import torch

class CFG:
    seed = 42
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # --- Paths ---
    data_dir = 'data'
    save_dir = '/kaggle/working/checkpoints'

    # --- Fold ---
    fold = 0

    # --- Image layout ---
    img_w = 5600
    img_h = 1700
    crop_h = 480
    signal_t0 = 301
    signal_t1 = 5301
    signal_w = 5000

    # --- ECG physics ---
    zero_mv = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79.0

    # --- Model ---
    backbone = 'resnet34.a3_in1k'
    pretrained = False
    decoder_dims = [256, 128, 64, 32]
    n_leads = 1
    temperature = 0.5

    # --- Training ---
    epochs = 60
    batch_size = 4
    lr = 5e-5
    weight_decay = 1e-4
    num_workers = 2
    use_adaptive_sigma = True
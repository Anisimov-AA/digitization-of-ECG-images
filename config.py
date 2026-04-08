# This is the single source of truth for all settings.
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
    save_dir = '/content/drive/MyDrive/ecg/checkpoints'
    
    # --- Fold ---
    # We split 977 images into 5 groups. One group is held out
    # for validation, the rest are used for training.
    fold = 0

    # --- Image layout ---
    # After rectification every ECG image is exactly 1700 x 5600.
    # The actual signal trace lives in columns 301–5300 (5000 px).
    # We crop a 480px tall strip around each of the 4 baselines,
    # giving us 4 small images per ECG instead of one huge one.
    img_w = 5600
    img_h = 1700
    crop_h = 480
    signal_t0 = 301
    signal_t1 = 5301
    signal_w = 5000

    # --- ECG physics ---
    # Each row has a known baseline y-position (in pixels) where
    # the signal reads 0 mV. The scale is 79 pixels per millivolt.
    # Row 0: leads I, aVR, V1, V4
    # Row 1: leads II(short), aVL, V2, V5
    # Row 2: leads III, aVF, V3, V6
    # Row 3: lead II full 10-second rhythm strip
    zero_mv = [703.5, 987.5, 1271.5, 1531.5]
    mv_to_pixel = 79.0

    # --- Model ---
    # ResNet34 UNet with coordinate-aware decoder.
    # Soft-argmax head predicts one y-position per pixel column,
    # which we convert to millivolts using the physics above.
    backbone = 'resnet34.a3_in1k'
    decoder_dims = [256, 128, 64, 32]
    n_leads = 1
    temperature = 0.5

    # --- Training ---
    epochs = 30
    batch_size = 4
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 2
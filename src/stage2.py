import cv2
import numpy as np
import torch
import pickle
import os
import multiprocessing as mp

from stage2_smp_model import Net as WholeModel
from stage2_lead_model import Net as LeadModel
from stage2_common import timer, time_to_str, draw_lead_pixel

from .postprocess import pixel_to_series_exp


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def compute_stage2_constants(cfg):
    """Derive all Stage 2 constants from config."""
    s = cfg["stage2"]
    x_scale = s["img_w_scale"] / s["img_w_denom"]
    add_x = s["add_x"]

    c = {
        "IMG_H": s["img_h"],
        "IMG_W": int(2200 * x_scale) + add_x,
        "OFFSET": s["offset"],
        "WINDOW_SIZE": s["window_size"],
        "IGNORE_EDGE": s["ignore_edge"],
        "x0": s["x0"],
        "x1": s["x1"],
        "y0": s["y0"],
        "y1": s["y1"],
        "zero_mv": s["zero_mv"],
        "zero_mv_trimed": [p - s["offset"] for p in s["zero_mv"]],
        "mv_to_pixel": s["mv_to_pixel"],
        "t0": int(s["t0_raw"] * x_scale) + add_x,
        "t1": int(s["t1_raw"] * x_scale) + add_x,
    }

    # Ensemble regions
    height_after_trimed = s["y1"] - s["offset"]
    ens_regions = []
    for zmv in c["zero_mv_trimed"]:
        trim_upper = int(zmv) - s["window_size"]
        trim_lower = int(zmv) + s["window_size"]
        lead_upper = s["ignore_edge"]
        lead_lower = -s["ignore_edge"]
        if trim_lower > height_after_trimed:
            lead_lower = (trim_lower - height_after_trimed + s["ignore_edge"]) * -1
            trim_lower = height_after_trimed
        trim_upper += s["ignore_edge"]
        trim_lower -= s["ignore_edge"]
        ens_regions.append([trim_upper, trim_lower, lead_upper, lead_lower])
    c["ens_regions"] = ens_regions

    return c


def read_images(path, consts):
    """Read rectified image → trim_image + lead_images."""
    IMG_W = consts["IMG_W"]
    IMG_H = consts["IMG_H"]
    OFFSET = consts["OFFSET"]
    WINDOW_SIZE = consts["WINDOW_SIZE"]
    x0, x1 = consts["x0"], consts["x1"]
    y0, y1 = consts["y0"], consts["y1"]
    zero_mv = consts["zero_mv"]

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    trim_image = image.copy()[OFFSET:y1, x0:x1]

    image = image[y0:y1, x0:x1]
    H, W, _ = image.shape
    lead_images = []
    for zmv in zero_mv:
        h0, h1 = int(zmv) - WINDOW_SIZE, int(zmv) + WINDOW_SIZE
        src_h0, src_h1 = max(0, h0), min(H, h1)
        dst_h0 = src_h0 - h0
        dst_h1 = dst_h0 + (src_h1 - src_h0)
        lead_img = np.zeros((WINDOW_SIZE * 2, W, 3))
        lead_img[dst_h0:dst_h1, :, :] = image[src_h0:src_h1, :, :]
        lead_images.append(lead_img)

    return trim_image, np.stack(lead_images)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_whole_model(encoder_name, weight_path, device):
    model = WholeModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        decoder_name="unet",
        use_coord_conv=True,
        pretrained=False,
    )
    sd = torch.load(weight_path, map_location="cpu")
    print(model.load_state_dict(sd, strict=False))
    model.to(device)
    model.eval()
    model.output_type = ["infer"]
    return model


def get_lead_model(encoder_name, weight_path, fusion_type, device):
    model = LeadModel(
        encoder_name=encoder_name,
        encoder_weights=None,
        fusion_type=fusion_type,
    )
    sd = torch.load(weight_path, map_location="cpu")
    print(model.load_state_dict(sd, strict=False))
    model.to(device)
    model.eval()
    model.output_type = ["infer"]
    return model


def load_models(cfg, device):
    """Load all whole + lead models from config."""
    someya_dir = cfg["paths"]["someya_dir"]

    whole_models = [
        get_whole_model(m["encoder"], f"{someya_dir}/{m['weight']}", device)
        for m in cfg["models"]["whole"]
    ]
    lead_models = [
        get_lead_model(m["encoder"], f"{someya_dir}/{m['weight']}", m["fusion"], device)
        for m in cfg["models"]["lead"]
    ]
    return whole_models, lead_models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_one(sample_id, trim_image, lead_images, whole_models, lead_models,
              consts, tta, float_type, length):
    """Run Stage 2 inference for a single sample. Returns series (4, length)."""
    ens_regions = consts["ens_regions"]
    zero_mv_trimed = consts["zero_mv_trimed"]
    mv_to_pixel = consts["mv_to_pixel"]
    t0, t1 = consts["t0"], consts["t1"]

    pixel_ens = np.zeros((4, trim_image.shape[0], trim_image.shape[1]))

    # --- Whole models ---
    batch = {
        "image": torch.from_numpy(
            np.ascontiguousarray(trim_image.transpose(2, 0, 1))
        ).unsqueeze(0),
    }
    batch_tta = {
        "image": torch.from_numpy(
            np.ascontiguousarray(np.fliplr(trim_image).copy().transpose(2, 0, 1))
        ).unsqueeze(0),
    }

    with torch.amp.autocast("cuda", dtype=float_type):
        with torch.no_grad():
            for model in whole_models:
                for flip in tta:
                    if flip:
                        output = model(batch_tta)
                        pixel = output["pixel"].float().data.cpu().numpy()[0]
                        pixel = np.flip(pixel, axis=flip)
                    else:
                        output = model(batch)
                        pixel = output["pixel"].float().data.cpu().numpy()[0]
                    pixel_ens += pixel

    # --- Lead models ---
    lead_tensor = torch.from_numpy(
        lead_images.transpose(0, 3, 1, 2)
    ).contiguous()
    batch_l = {"image": lead_tensor.unsqueeze(0)}
    batch_l_tta = {"image": torch.flip(lead_tensor, dims=[3]).unsqueeze(0)}

    with torch.amp.autocast("cuda", dtype=float_type):
        with torch.no_grad():
            for model in lead_models:
                for flip in tta:
                    if flip:
                        output = model(batch_l_tta)
                        pixel = output["pixel"].float().data.cpu().numpy()[0].squeeze(1)
                        pixel = np.flip(pixel, axis=flip)
                    else:
                        output = model(batch_l)
                        pixel = output["pixel"].float().data.cpu().numpy()[0].squeeze(1)

                    for i in range(4):
                        tu, tl, lu, ll = ens_regions[i]
                        pixel_ens[i][tu:tl] += pixel[i][lu:ll]

    # --- Weighted average ---
    ens_weight = (
        np.ones((trim_image.shape[0], trim_image.shape[1]))
        * len(whole_models) * len(tta)
    )
    for i in range(4):
        tu, tl, _, _ = ens_regions[i]
        ens_weight[tu:tl] += len(lead_models) * len(tta)
    pixel_ens /= ens_weight

    # --- Pixel → time series ---
    series_in_pixel = pixel_to_series_exp(pixel_ens[..., t0:t1], zero_mv_trimed, length)
    series = (np.array(zero_mv_trimed).reshape(4, 1) - series_in_pixel) / mv_to_pixel

    return series


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_stage2(gpu_id, assigned_ids, valid_df, cfg, consts,
               prev_fail_ids=None, fail_id_file=None):
    device = f"cuda:{gpu_id}"
    if prev_fail_ids is None:
        prev_fail_ids = []
    local_fail_id = []

    out_dir = cfg["paths"]["out_dir"]
    float_type = getattr(torch, cfg["inference"]["float_type"])
    tta = cfg["inference"]["tta"]

    whole_models, lead_models = load_models(cfg, device)

    start = timer()
    for n, sid in enumerate(assigned_ids):
        ts = time_to_str(timer() - start, "sec")
        print(f"\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sid}", ts, end="", flush=True)

        if sid in prev_fail_ids:
            continue

        from .utils import read_sampling_length
        length = read_sampling_length(sid, valid_df)
        trim_image, lead_images = read_images(
            f"{out_dir}/rectified/{sid}.rect.png", consts
        )

        try:
            series = infer_one(
                sid, trim_image, lead_images,
                whole_models, lead_models,
                consts, tta, float_type, length,
            )
            np.save(f"{out_dir}/digitalised/{sid}.series.npy", series)
        except:
            local_fail_id.append(sid)

        torch.cuda.empty_cache()

    print(f"\n[GPU{gpu_id}] Stage2 completed. Failed: {len(local_fail_id)}")

    if fail_id_file:
        with open(fail_id_file, "wb") as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage2_parallel(valid_ids, valid_df, cfg, consts, prev_fail_ids=None):
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(f"{out_dir}/digitalised", exist_ok=True)
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f"Only {n_gpus} GPU(s), running single GPU")
        return run_stage2(0, valid_ids, valid_df, cfg, consts, prev_fail_ids)

    mid = len(valid_ids) // 2
    ids_0, ids_1 = valid_ids[:mid], valid_ids[mid:]
    print(f"Stage2: GPU0={len(ids_0)} | GPU1={len(ids_1)}")

    ff0 = f"{out_dir}/fail_stage2_gpu0.pkl"
    ff1 = f"{out_dir}/fail_stage2_gpu1.pkl"

    p0 = mp.Process(target=run_stage2, args=(0, ids_0, valid_df, cfg, consts, prev_fail_ids, ff0))
    p1 = mp.Process(target=run_stage2, args=(1, ids_1, valid_df, cfg, consts, prev_fail_ids, ff1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [ff0, ff1]:
        if os.path.exists(ff):
            with open(ff, "rb") as f:
                fail_id.extend(pickle.load(f))
    if prev_fail_ids:
        fail_id.extend(prev_fail_ids)

    print(f"FAIL_ID (Stage2): {fail_id}")
    return fail_id
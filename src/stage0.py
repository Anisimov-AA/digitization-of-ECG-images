import cv2
import numpy as np
import torch
import pickle
import os
import multiprocessing as mp

from stage0_model import Net as Stage0Net
from stage0_common import (
    load_net, image_to_batch, output_to_predict,
    normalise_by_homography, timer, time_to_str,
)


def run_stage0(gpu_id, assigned_ids, read_image_fn, out_dir, weight_dir,
               float_type=torch.float16, fail_id_file=None):
    device = f"cuda:{gpu_id}"
    local_fail_id = []

    net = Stage0Net(pretrained=False)
    net = load_net(net, f"{weight_dir}/stage0-last.checkpoint.pth")
    net.to(device)

    start = timer()
    for n, sid in enumerate(assigned_ids):
        ts = time_to_str(timer() - start, "sec")
        print(f"\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sid}", ts, end="", flush=True)

        image = read_image_fn(sid)
        batch = image_to_batch(image)

        with torch.amp.autocast("cuda", dtype=float_type):
            with torch.no_grad():
                output = net(batch)
                try:
                    rotated, keypoint = output_to_predict(image, batch, output)
                    normalised, keypoint, homo = normalise_by_homography(rotated, keypoint)
                    cv2.imwrite(
                        f"{out_dir}/normalised/{sid}.norm.png",
                        cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR),
                    )
                    np.save(f"{out_dir}/normalised/{sid}.homo.npy", homo)
                except:
                    local_fail_id.append(sid)

        torch.cuda.empty_cache()

    print(f"\n[GPU{gpu_id}] Stage0 completed. Failed: {len(local_fail_id)}")

    if fail_id_file:
        with open(fail_id_file, "wb") as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage0_parallel(valid_ids, read_image_fn, out_dir, weight_dir,
                        float_type=torch.float16):
    os.makedirs(f"{out_dir}/normalised", exist_ok=True)
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f"Only {n_gpus} GPU(s), running single GPU")
        return run_stage0(0, valid_ids, read_image_fn, out_dir, weight_dir, float_type)

    mid = len(valid_ids) // 2
    ids_0, ids_1 = valid_ids[:mid], valid_ids[mid:]
    print(f"Stage0: GPU0={len(ids_0)} | GPU1={len(ids_1)}")

    ff0 = f"{out_dir}/fail_stage0_gpu0.pkl"
    ff1 = f"{out_dir}/fail_stage0_gpu1.pkl"

    p0 = mp.Process(target=run_stage0, args=(0, ids_0, read_image_fn, out_dir, weight_dir, float_type, ff0))
    p1 = mp.Process(target=run_stage0, args=(1, ids_1, read_image_fn, out_dir, weight_dir, float_type, ff1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [ff0, ff1]:
        if os.path.exists(ff):
            with open(ff, "rb") as f:
                fail_id.extend(pickle.load(f))

    print(f"FAIL_ID (Stage0): {fail_id}")
    return fail_id
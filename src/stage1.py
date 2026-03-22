import cv2
import numpy as np
import torch
import pickle
import os
import multiprocessing as mp

from stage1_model import Net as Stage1Net
from stage1_common import (
    load_net, output_to_predict, rectify_image, timer, time_to_str,
)


def run_stage1(gpu_id, assigned_ids, out_dir, weight_dir, prev_fail_ids=None,
               float_type=torch.float16, fail_id_file=None):
    device = f"cuda:{gpu_id}"
    if prev_fail_ids is None:
        prev_fail_ids = []
    local_fail_id = []

    net = Stage1Net(pretrained=False)
    net = load_net(net, f"{weight_dir}/stage1-last.checkpoint.pth")
    net.to(device)

    start = timer()
    for n, sid in enumerate(assigned_ids):
        ts = time_to_str(timer() - start, "sec")
        print(f"\r\t [GPU{gpu_id}] {n:4d}/{len(assigned_ids)} {sid}", ts, end="", flush=True)

        if sid in prev_fail_ids:
            continue

        image = cv2.imread(f"{out_dir}/normalised/{sid}.norm.png", cv2.IMREAD_COLOR_RGB)
        batch = {
            "image": torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            ).unsqueeze(0),
        }

        with torch.amp.autocast("cuda", dtype=float_type):
            with torch.no_grad():
                output = net(batch)
                try:
                    gridpoint_xy, more = output_to_predict(image, batch, output)
                    rectified = rectify_image(image, gridpoint_xy)
                    cv2.imwrite(
                        f"{out_dir}/rectified/{sid}.rect.png",
                        cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR),
                    )
                    np.save(f"{out_dir}/rectified/{sid}.gridpoint_xy.npy", gridpoint_xy)
                except:
                    local_fail_id.append(sid)

        torch.cuda.empty_cache()

    print(f"\n[GPU{gpu_id}] Stage1 completed. Failed: {len(local_fail_id)}")

    if fail_id_file:
        with open(fail_id_file, "wb") as f:
            pickle.dump(local_fail_id, f)

    return local_fail_id


def run_stage1_parallel(valid_ids, out_dir, weight_dir, prev_fail_ids=None,
                        float_type=torch.float16):
    os.makedirs(f"{out_dir}/rectified", exist_ok=True)
    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print(f"Only {n_gpus} GPU(s), running single GPU")
        return run_stage1(0, valid_ids, out_dir, weight_dir, prev_fail_ids, float_type)

    mid = len(valid_ids) // 2
    ids_0, ids_1 = valid_ids[:mid], valid_ids[mid:]
    print(f"Stage1: GPU0={len(ids_0)} | GPU1={len(ids_1)}")

    ff0 = f"{out_dir}/fail_stage1_gpu0.pkl"
    ff1 = f"{out_dir}/fail_stage1_gpu1.pkl"

    p0 = mp.Process(target=run_stage1, args=(0, ids_0, out_dir, weight_dir, prev_fail_ids, float_type, ff0))
    p1 = mp.Process(target=run_stage1, args=(1, ids_1, out_dir, weight_dir, prev_fail_ids, float_type, ff1))
    p0.start(); p1.start()
    p0.join();  p1.join()

    fail_id = []
    for ff in [ff0, ff1]:
        if os.path.exists(ff):
            with open(ff, "rb") as f:
                fail_id.extend(pickle.load(f))
    if prev_fail_ids:
        fail_id.extend(prev_fail_ids)

    print(f"FAIL_ID (Stage1): {fail_id}")
    return fail_id
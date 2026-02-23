import cv2, numpy as np, json, matplotlib.pyplot as plt

gray = cv2.imread("outputs/clean.png", cv2.IMREAD_GRAYSCALE)
config = json.load(open("outputs/config.json"))

leads = config["leads"]
px_per_mm = config["px_per_mm"]

# from ECG standard: 25mm/s horizontal, 10mm/mV vertical
mv_per_px = 1.0 / (px_per_mm * 10)
sec_per_px = 1.0 / (px_per_mm * 25)


def trace_lead(lead_gray, baseline_local):
    """weighted center of mass — darker pixels count more"""
    weights = (255 - lead_gray).astype(float)
    weights[weights < 50] = 0  # ignore near-white
    rows = np.arange(lead_gray.shape[0])
    trace = np.zeros(lead_gray.shape[1])
    for col in range(lead_gray.shape[1]):
        wt = weights[:, col]
        s = wt.sum()
        trace[col] = np.dot(rows, wt) / s if s > 0 else baseline_local
    return trace


# trace all leads
all_signals = {}
for name, info in leads.items():
    lead_gray = gray[info["y1"]:info["y2"], info["x1"]:info["x2"]]
    bl = info["baseline"] - info["y1"]

    signal_y = trace_lead(lead_gray, bl)
    signal_mv = (bl - signal_y) * mv_per_px
    time_s = np.arange(len(signal_mv)) * sec_per_px

    all_signals[name] = {"mv": signal_mv, "time": time_s}
    print(f"  {name:>7}: {len(signal_mv)} samples, {time_s[-1]:.2f}s")

# save for step 9
np.savez("outputs/signals.npz",
         **{f"{n}_mv": s["mv"] for n, s in all_signals.items()},
         **{f"{n}_time": s["time"] for n, s in all_signals.items()})

# --- vis ---

display_order = ["I", "aVR", "II", "aVL", "III", "aVF",
                 "V1", "V4", "V2", "V5", "V3", "V6"]

# (a) trace overlay on cleaned leads
fig, axes = plt.subplots(6, 2, figsize=(18, 16))
for idx, name in enumerate(display_order):
    ax = axes[idx // 2, idx % 2]
    info = leads[name]
    lead_gray = gray[info["y1"]:info["y2"], info["x1"]:info["x2"]]
    bl = info["baseline"] - info["y1"]
    signal_y = trace_lead(lead_gray, bl)

    ax.imshow(lead_gray, cmap="gray")
    ax.plot(range(len(signal_y)), signal_y, color="green", lw=0.5)
    ax.set_title(name, fontweight="bold"); ax.axis("off")
fig.suptitle("Trace overlay (green)")
fig.savefig("outputs/step8a_trace_overlay.png", dpi=120); plt.close()

# (b) millivolt signals
fig, axes = plt.subplots(6, 2, figsize=(18, 16))
for idx, name in enumerate(display_order):
    ax = axes[idx // 2, idx % 2]
    sig = all_signals[name]
    ax.plot(sig["time"], sig["mv"], color="black", lw=0.5)
    ax.axhline(0, color="red", lw=0.5, ls="--")
    ax.set_title(name, fontweight="bold")
    ax.set_ylabel("mV"); ax.set_xlabel("s")
fig.suptitle("Extracted signals (mV)")
fig.savefig("outputs/step8b_signals_mv.png", dpi=120); plt.close()
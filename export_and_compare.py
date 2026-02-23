import numpy as np, csv, json, matplotlib.pyplot as plt

# --- load GT ---
with open("1006427285.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    raw = [row for row in reader]

fs = len(raw) / 10.0

# GT has all 10250 rows, each lead fills its own time window
# row index / fs = time in seconds
gt = {}
gt_time = {}
for col_idx, name in enumerate(header):
    for i, r in enumerate(raw):
        val = r[col_idx].strip()
        if val:
            if name not in gt:
                gt[name] = []
                gt_time[name] = []
            gt[name].append(float(val))
            gt_time[name].append(i / fs)  # actual time position in recording

for name in gt:
    gt[name] = np.array(gt[name])
    gt_time[name] = np.array(gt_time[name])
    print(f"  GT {name:>4}: {len(gt[name])} samples, t={gt_time[name][0]:.2f}-{gt_time[name][-1]:.2f}s")

# --- load our signals ---
data = np.load("outputs/signals.npz")

# our leads need time offsets to match GT time windows
# I,III start at t=0, aVR,aVL,aVF at t=2.5, V1-V3 at t=5.0, V4-V6 at t=7.5

shift = 0.022
offsets = {
    "I": 0+0.022, "III": 0+0.015,
    "aVR": 2.5+shift, "aVL": 2.5+shift, "aVF": 2.5+shift,
    "V1": 5.0+shift, "V2": 5.0+shift, "V3": 5.0+shift,
    "V4": 7.5+shift, "V5": 7.5+shift, "V6": 7.5+shift,
}

# --- plot all on one timeline ---
order = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

fig, axes = plt.subplots(6, 2, figsize=(18, 18))
for idx, name in enumerate(order):
    ax = axes[idx // 2, idx % 2]

    # GT at its actual time
    ax.plot(gt_time[name], gt[name], color="blue", lw=0.5, label="GT")

    # ours shifted to match
    key = "II_long" if name == "II" else name
    our_mv = data[f"{key}_mv"]
    our_t = data[f"{key}_time"]
    if name != "II":
        our_t = our_t + offsets[name]

    ax.plot(our_t, our_mv, color="red", lw=0.5, label="ours")
    ax.set_title(name, fontweight="bold")
    ax.set_ylabel("mV"); ax.set_xlabel("s")
    if idx == 0: ax.legend()

fig.suptitle("Blue = GT, Red = ours (at GT time positions)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/step9_compare.png", dpi=120)
plt.show()

print("  Saved: outputs/step9_compare.png")
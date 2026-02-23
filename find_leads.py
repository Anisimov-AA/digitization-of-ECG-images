import cv2, numpy as np, json, matplotlib.pyplot as plt

gray = cv2.imread("outputs/clean.png", cv2.IMREAD_GRAYSCALE)
config = json.load(open("outputs/config.json"))

bands = config["bands"]
left, right = config["left"], config["right"]
baselines = config["baselines"]

lead_names = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
    ["II_long"],
]

leads = {}

for i, (y1, y2) in enumerate(bands):
    row = gray[y1:y2, left:right]
    col_sum = (row < 128).sum(axis=0)

    # find non-zero regions = leads
    active = col_sum > 0
    changes = np.diff(active.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if active[0]: starts = np.insert(starts, 0, 0)
    if active[-1]: ends = np.append(ends, len(active))

    segs = list(zip((starts + left).tolist(), (ends + left).tolist()))

    # merge tiny gaps (< 5px noise)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    names = lead_names[i]
    if len(merged) != len(names):
        print(f"  WARNING Row {i}: expected {len(names)}, got {len(merged)}")

    for j, name in enumerate(names):
        if j < len(merged):
            leads[name] = {
                "y1": y1, "y2": y2,
                "x1": merged[j][0], "x2": merged[j][1],
                "baseline": baselines[i]
            }
            print(f"  {name:>7}: x={merged[j][0]}-{merged[j][1]} ({merged[j][1]-merged[j][0]}px)")

# save to config
config["leads"] = leads
json.dump(config, open("outputs/config.json", "w"), indent=2)

# --- vis ---

colors = ["red", "blue", "green", "orange"]
fig, axes = plt.subplots(4, 1, figsize=(16, 10))
for i, (y1, y2) in enumerate(bands):
    axes[i].imshow(gray[y1:y2, left:right], cmap="gray")
    for j, name in enumerate(lead_names[i]):
        if name in leads:
            info = leads[name]
            x1l = info["x1"] - left
            x2l = info["x2"] - left
            c = colors[j % len(colors)]
            axes[i].axvline(x1l, color=c, lw=2)
            axes[i].axvline(x2l, color=c, lw=2, ls="--")
            axes[i].text((x1l + x2l) / 2, 10, name, color=c,
                         fontsize=11, fontweight="bold", ha="center")
    axes[i].set_title(f"Row {i}"); axes[i].axis("off")
fig.savefig("outputs/step7_leads.png", dpi=120); plt.close()
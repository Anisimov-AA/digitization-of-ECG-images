import cv2, numpy as np, json, matplotlib.pyplot as plt

gray = cv2.cvtColor(cv2.imread("outputs/no_text.png"), cv2.COLOR_BGR2GRAY)
config = json.load(open("outputs/config.json"))
bands = config["bands"]
left = config["left"]
px_per_mm = config["px_per_mm"]

# only look at the left ~200px of each row — that's where the pulse is
SEARCH = 200
WALL_THRESH = 40

baselines = []
signal_starts = []
pulse_ranges = []

for i, (y1, y2) in enumerate(bands):
    row = gray[y1:y2, left:left + SEARCH]
    col_sum = (row < 128).sum(axis=0)

    # columns with lots of dark pixels = the vertical walls of the pulse
    walls = np.where(col_sum > WALL_THRESH)[0]
    lw, rw = int(walls[0]), int(walls[-1])
    signal_starts.append(rw + left)
    pulse_ranges.append([lw, rw])

    # between the walls, the bottom-most dark pixel = baseline
    region = (row < 128)[:, lw:rw + 1]
    dark_rows = np.where(region.any(axis=1))[0]
    top, bottom = int(dark_rows[0]), int(dark_rows[-1])
    baseline = y1 + bottom
    baselines.append(baseline)

    pulse_h = bottom - top
    print(f"  Row {i}: baseline={baseline}, pulse={pulse_h}px (expect ~{px_per_mm*10:.0f})")

# save to config
config.update({"baselines": baselines, "signal_starts": signal_starts, "pulse_ranges": pulse_ranges})
json.dump(config, open("outputs/config.json", "w"), indent=2)

# --- vis ---

# show each row with baseline (red) and pulse top (blue dashed)
fig, axes = plt.subplots(4, 1, figsize=(16, 10))
img_rgb = cv2.cvtColor(cv2.imread("outputs/no_text.png"), cv2.COLOR_BGR2RGB)
for i, (y1, y2) in enumerate(bands):
    axes[i].imshow(img_rgb[y1:y2, left:config["right"]])
    axes[i].axhline(baselines[i] - y1, color="red", lw=2, label=f"baseline y={baselines[i]}")
    # pulse top = baseline - pulse_height
    region = (gray[y1:y2, left:left+SEARCH] < 128)[:, pulse_ranges[i][0]:pulse_ranges[i][1]+1]
    top = np.where(region.any(axis=1))[0][0]
    axes[i].axhline(top, color="blue", lw=1, ls="--", label="1mV top")
    axes[i].legend(loc="upper right"); axes[i].axis("off")
    axes[i].set_title(f"Row {i}")
fig.savefig("outputs/step5_baselines&pulses.png", dpi=120); plt.close()
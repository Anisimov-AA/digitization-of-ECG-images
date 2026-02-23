import cv2, numpy as np, json, matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- tuneable padding ---
PULSE_PAD_LEFT = 3   # extra pixels to cut left of first pulse peak
PULSE_PAD_RIGHT = 3  # extra pixels to cut right of second pulse peak
BAR_PAD_LEFT = 2     # extra pixels to cut left of bar
BAR_PAD_RIGHT = 2    # extra pixels to cut right of bar

gray = cv2.imread("outputs/no_text.png", cv2.IMREAD_GRAYSCALE)
config = json.load(open("outputs/config.json"))

bands = config["bands"]
left, right = config["left"], config["right"]

result = gray.copy()
result[:] = 255

for i, (y1, y2) in enumerate(bands):
    row_gray = gray[y1:y2, left:right]
    dark = (row_gray < 128).astype(np.uint8)
    clean = dark.copy()
    col_sum = clean.sum(axis=0).astype(float)

    peaks, props = find_peaks(col_sum, height=20, distance=5)
    heights = props["peak_heights"]

    # --- detect and remove pulse ---
    pulse_end = 0
    for j in range(len(peaks) - 1):
        h1, h2 = heights[j], heights[j+1]
        dist = peaks[j+1] - peaks[j]
        if abs(h1 - h2) / max(h1, h2) < 0.1 and dist < 50:
            p1, p2 = peaks[j], peaks[j+1]
            cut_l = max(0, p1 - PULSE_PAD_LEFT)
            cut_r = min(p2 + PULSE_PAD_RIGHT, clean.shape[1])
            clean[:, cut_l:cut_r] = 0
            pulse_end = p2 + 5
            print(f"  Row {i} PULSE: peaks={p1},{p2} | blank {cut_l}-{cut_r}")
            break

    # --- detect and remove bars ---
    diff = np.abs(np.diff(col_sum))
    run_start = None
    for k in range(pulse_end, len(diff)):
        if diff[k] < 2 and col_sum[k] > 15:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None:
                run_len = k - run_start
                if run_len >= 4:
                    cut_l = max(0, run_start - BAR_PAD_LEFT)
                    cut_r = min(k + BAR_PAD_RIGHT, clean.shape[1])
                    clean[:, cut_l:cut_r] = 0
                    print(f"  Row {i} BAR: flat={run_start}-{k} | blank {cut_l}-{cut_r}")
                run_start = None

    row_result = np.full_like(row_gray, 255)
    row_result[clean == 1] = row_gray[clean == 1]
    result[y1:y2, left:right] = row_result

cv2.imwrite("outputs/clean.png", result)

# --- vis ---

# (a) row 0 detection
y1, y2 = bands[0]
row_gray = gray[y1:y2, left:right]
col_sum = (row_gray < 128).sum(axis=0).astype(float)

peaks, props = find_peaks(col_sum, height=20, distance=5)
heights = props["peak_heights"]
pulse_vis_l, pulse_vis_r = 0, 0
for j in range(len(peaks) - 1):
    h1, h2 = heights[j], heights[j+1]
    if abs(h1 - h2) / max(h1, h2) < 0.1 and peaks[j+1] - peaks[j] < 50:
        pulse_vis_l = max(0, peaks[j] - PULSE_PAD_LEFT)
        pulse_vis_r = peaks[j+1] + PULSE_PAD_RIGHT
        break

diff = np.abs(np.diff(col_sum))
bar_vis = []
run_start = None
for k in range(pulse_vis_r + 5, len(diff)):
    if diff[k] < 2 and col_sum[k] > 15:
        if run_start is None: run_start = k
    else:
        if run_start is not None:
            if k - run_start >= 4:
                bar_vis.append((run_start - BAR_PAD_LEFT, k + BAR_PAD_RIGHT))
            run_start = None

fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
axes[0].imshow(row_gray, cmap="gray"); axes[0].axis("off")
axes[0].set_title("Row 0 — before")
axes[1].plot(col_sum, color="black", lw=0.5)
axes[1].axvspan(pulse_vis_l, pulse_vis_r, color="red", alpha=0.2)
for s, e in bar_vis:
    axes[1].axvspan(s, e, color="blue", alpha=0.3)
axes[1].set_ylabel("count")
axes[1].set_title("Red = pulse, Blue = bars")
fig.savefig("outputs/step6a_detection.png", dpi=120); plt.close()

# (b) before/after projections
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
for i, (y1, y2) in enumerate(bands):
    before = (gray[y1:y2, left:right] < 128).sum(axis=0)
    after = (result[y1:y2, left:right] < 128).sum(axis=0)
    axes[i].plot(before, color="red", lw=0.5, label="before")
    axes[i].plot(after, color="black", lw=0.5, label="after")
    axes[i].set_title(f"Row {i}"); axes[i].set_ylabel("count")
    if i == 0: axes[i].legend()
axes[3].set_xlabel("x position")
fig.savefig("outputs/step6b_projections.png", dpi=120); plt.close()
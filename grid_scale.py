import cv2, numpy as np, json, matplotlib.pyplot as plt
from scipy.signal import find_peaks

img = cv2.imread(INPUT)
h, w = img.shape[:2]

# grid is red, red absorbs green -> grid shows as dips in green channel
green = img[:, :, 1]

# to measure those dips we need a clean slice with no signal
# so pick an empty row and column
ROW_Y, COL_X = h - 10, w - 10
row, col = green[ROW_Y, :], green[:, COL_X]

# find_peaks needs maxima not dips, so we flip
flipped_h = 255 - row.astype(float)
flipped_v = 255 - col.astype(float)

# only detect the tall peaks
HEIGHT_THRESH = 100
peaks_h, _ = find_peaks(flipped_h, height=HEIGHT_THRESH, distance=20)
peaks_v, _ = find_peaks(flipped_v, height=HEIGHT_THRESH, distance=20)

# big grid lines are 5mm  
# so spacing / 5 = px per mm
px_per_mm_h = np.diff(peaks_h).mean() / 5
px_per_mm_v = np.diff(peaks_v).mean() / 5
px_per_mm = (px_per_mm_h + px_per_mm_v) / 2

print(f"  H: {px_per_mm_h:.2f}  V: {px_per_mm_v:.2f}  ratio: {px_per_mm_h/px_per_mm_v:.3f}")
print(f"  1 sec = {px_per_mm*25:.1f}px   1 mV = {px_per_mm*10:.1f}px")

# save — every later step needs this number
json.dump({"px_per_mm": round(float(px_per_mm), 4)},
          open("outputs/config.json", "w"), indent=2)

# --- vis ---

# (a) where we sampled
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.axhline(ROW_Y, color="yellow", lw=2); ax.axvline(COL_X, color="cyan", lw=2)
ax.axis("off"); fig.savefig("outputs/step1a_sampling.png", dpi=120); plt.close()

# (b) raw green signal - dips = grid lines
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5))
a1.plot(row, color="green", lw=0.5); a1.set_title(f"Raw green - horizontal (row {ROW_Y})")
a2.plot(col, color="green", lw=0.5); a2.set_title(f"Raw green - vertical (col {COL_X})")
fig.savefig("outputs/step1b_raw_green.png", dpi=120); plt.close()

# (dc) flipped with detected big grid lines + threshold line
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5))
a1.plot(flipped_h, color="gray", lw=0.5)
a1.plot(peaks_h, flipped_h[peaks_h], "rv", ms=3)
a1.axhline(HEIGHT_THRESH, color="red", lw=1, ls="--", alpha=0.5)
a1.set_title(f"Flipped - H: {len(peaks_h)} big lines, {px_per_mm_h:.2f} px/mm")
a2.plot(flipped_v, color="gray", lw=0.5)
a2.plot(peaks_v, flipped_v[peaks_v], "rv", ms=3)
a2.axhline(HEIGHT_THRESH, color="red", lw=1, ls="--", alpha=0.5)
a2.set_title(f"Flipped - V: {len(peaks_v)} big lines, {px_per_mm_v:.2f} px/mm")
fig.savefig("outputs/step1c_peaks.png", dpi=120); plt.close()


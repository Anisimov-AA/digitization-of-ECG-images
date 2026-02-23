import cv2, numpy as np, json, matplotlib.pyplot as plt

img = cv2.imread("outputs/no_text.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# count dark pixels in each row of the cropped area
TOP = 10
BOT = h - 10
LEFT = 10
RIGHT = w - 10
cropped = gray[TOP:BOT, LEFT:RIGHT] < 128
row_sum = cropped.sum(axis=1)

# rows with signal have counts > 0, gaps between rows are zero
# find where it switches from zero to non-zero = row boundaries
active = row_sum > 0
changes = np.diff(active.astype(int))
starts = np.where(changes == 1)[0] + 1
ends = np.where(changes == -1)[0] + 1
if active[0]: starts = np.insert(starts, 0, 0)
if active[-1]: ends = np.append(ends, len(active))

bands = list(zip((starts + TOP).tolist(), (ends + TOP).tolist()))

for i, (y1, y2) in enumerate(bands):
    print(f"  Row {i}: y={y1}-{y2} ({y2-y1}px)")

# save to config
config = json.load(open("outputs/config.json"))
config.update({"bands": bands, "left": LEFT, "right": RIGHT})
json.dump(config, open("outputs/config.json", "w"), indent=2)

# --- vis ---

# (a) horizontal projection — 4 bumps = 4 rows
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8),
    gridspec_kw={"width_ratios": [1, 4]})
y_pos = np.arange(TOP, BOT)
a1.plot(row_sum, y_pos, color="black", lw=0.5)
a1.invert_yaxis(); a1.set_xlabel("dark px count"); a1.set_ylabel("y")
a1.set_title("Projection")
a2.imshow(cv2.cvtColor(img[TOP:BOT, LEFT:RIGHT], cv2.COLOR_BGR2RGB))
a2.axis("off"); a2.set_title("Cropped area")
fig.savefig("outputs/step4a_projection.png", dpi=120); plt.close()

# (b) show full image with split lines
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
colors = ["red", "blue", "green", "orange"]
for i, (y1, y2) in enumerate(bands):
    c = colors[i % len(colors)]
    ax.axhline(y1, color=c, lw=1.5)
    ax.axhline(y2, color=c, lw=1.5, ls="--")
    ax.text(RIGHT + 5, (y1+y2)//2, f"Row {i}", color=c, fontsize=10, fontweight="bold")
ax.axis("off")
ax.set_title(f"Found {len(bands)} rows")
fig.savefig("outputs/step4b_rows.png", dpi=120); plt.close()
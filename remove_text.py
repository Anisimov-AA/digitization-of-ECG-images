import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread("outputs/no_grid.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# binary mask: 1 = dark pixel, 0 = background
dark = (gray < 128).astype(np.uint8)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)

# kill small blobs (text), keep big ones (signal + pulse + bars)
clean = dark.copy()
removed = 0
for lbl in range(1, num_labels):
    cw = stats[lbl, cv2.CC_STAT_WIDTH]
    area = stats[lbl, cv2.CC_STAT_AREA]
    if cw < 40 and area < 500:
        clean[labels == lbl] = 0
        removed += 1

# white canvas - only write back original gray where clean mask = 1
# this kills gray anti-aliased text edges automatically
result = np.full_like(gray, 255)
result[clean == 1] = gray[clean == 1]

# save as grayscale (no need for color anymore)
cv2.imwrite("outputs/no_text.png", result)
print(f"  Removed {removed} text blobs out of {num_labels - 1} total")

# --- vis ---

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
a1.imshow(gray, cmap="gray"); a1.set_title("Before"); a1.axis("off")
a2.imshow(result, cmap="gray"); a2.set_title(f"After ({removed} blobs removed)"); a2.axis("off")
fig.savefig("outputs/step3_before_after.png", dpi=120); plt.close()
import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread(INPUT)

# grid is red ink: high red, low green
# signal is black ink: low red, low green
# so R - G is big for grid, small for signal
r = img[:, :, 2].astype(float)
g = img[:, :, 1].astype(float)
diff = r - g

# anything with R-G > 5 is grid -> make it white
mask = diff > 5
result = img.copy()
result[mask] = [255, 255, 255]

cv2.imwrite("outputs/no_grid.png", result)
print(f"  Removed {mask.sum()} grid pixels")

# --- vis ---

# (a) R-G histogram — see the split between grid and non-grid
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(diff.ravel(), bins=100, color="gray", edgecolor="black")
ax.axvline(5, color="red", lw=2, ls="--", label="threshold=5")
ax.set_xlabel("R - G"); ax.set_ylabel("pixels"); ax.legend()
ax.set_title("grid is red ink: high red, low green, signal is black ink: low red, low green \nso R - G is big for grid, small for signal")
fig.savefig("outputs/step2a_rg_histogram.png", dpi=120); plt.close()

# (b) before / after
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
a1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a1.set_title("Before"); a1.axis("off")
a2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); a2.set_title("After"); a2.axis("off")
fig.savefig("outputs/step2b_before_after.png", dpi=120); plt.close()
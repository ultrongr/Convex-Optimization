import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rng = np.random.default_rng(7)

img = mpimg.imread("Data/flower.png").astype(float)[:, :, :3]
m, n, _ = img.shape
R0, G0, B0 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
gray = 0.299 * R0 + 0.587 * G0 + 0.114 * B0    # the only luminance we get to see
known_counts = [500, 1000, 1500, 2000]


def pick_known(k):
    flat = rng.choice(m * n, size=k, replace=False)
    return flat // n, flat % n


# recover the colour channels from the grayscale image plus the colours we know,
# choosing the smoothest colouring (smallest total variation across R,G,B together)
def colorize(rows, cols):
    R = cp.Variable((m, n))
    G = cp.Variable((m, n))
    B = cp.Variable((m, n))

    match_gray = 0.299 * R + 0.587 * G + 0.114 * B == gray
    in_range = [R >= 0, R <= 1, G >= 0, G <= 1, B >= 0, B <= 1]
    known = [R[rows, cols] == R0[rows, cols],
             G[rows, cols] == G0[rows, cols],
             B[rows, cols] == B0[rows, cols]]

    prob = cp.Problem(cp.Minimize(cp.tv(R, G, B)), [match_gray] + in_range + known)
    prob.solve()

    recovered = np.clip(np.dstack([R.value, G.value, B.value]), 0, 1)
    return recovered, prob.value


results = []
print(f"{'known':>7}{'tv cost':>12}")
print("-" * 19)
for k in known_counts:
    rows, cols = pick_known(k)
    recovered, cost = colorize(rows, cols)
    # what the solver sees: gray everywhere, true colour only at the known pixels
    hint = np.dstack([gray, gray, gray])
    hint[rows, cols] = img[rows, cols]
    results.append((k, hint, recovered))
    print(f"{k:>7}{cost:>12.3f}")


fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
axes[0, 0].imshow(img);  axes[0, 0].set_title("original")
axes[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=1); axes[0, 1].set_title("grayscale input")
for ax, (k, hint, recovered) in zip([axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], results):
    ax.imshow(recovered)
    ax.set_title(f"{k} known colours")
for ax in axes.ravel():
    ax.axis('off')
fig.tight_layout()
fig.savefig("task3_colorization.png", dpi=120)


# per-case view: the gray + known-colour hint next to the colouring it produced
fig2, axes2 = plt.subplots(len(known_counts), 2, figsize=(6, 12))
for row, (k, hint, recovered) in zip(axes2, results):
    row[0].imshow(hint);      row[0].set_title(f"grayscale + {k} known colours")
    row[1].imshow(recovered); row[1].set_title(f"colourised ({k})")
for ax in axes2.ravel():
    ax.axis('off')
fig2.tight_layout()
fig2.savefig("task3_cases.png", dpi=120)

print("\nsaved plots to task3_colorization.png and task3_cases.png")
plt.show()

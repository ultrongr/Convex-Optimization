import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rng = np.random.default_rng(7)

img = mpimg.imread("Data/server64.png").astype(float)
m, n = img.shape
known_counts = [3840, 3584, 3072, 2048]
trials = 10                 # random instances averaged per known-pixel count


# pick which pixels we get to keep, spread randomly over the whole image
def pick_known(k):
    flat = rng.choice(m * n, size=k, replace=False)
    return flat // n, flat % n


# fill in the missing pixels by making the image as smooth as possible while
# leaving the known ones untouched. metric picks squared vs absolute roughness.
def reconstruct(rows, cols, metric):
    full = cp.Variable((m, n))
    keep = [full[rows, cols] == img[rows, cols]]

    down = full[1:, :] - full[:-1, :]
    right = full[:, 1:] - full[:, :-1]
    if metric == "squared":
        roughness = cp.sum_squares(down) + cp.sum_squares(right)
    else:
        roughness = cp.sum(cp.abs(down)) + cp.sum(cp.abs(right))

    cp.Problem(cp.Minimize(roughness), keep).solve()
    return full.value


def mse(recon):
    return np.mean((recon - img) ** 2)


sq_curve = []
tv_curve = []
print(f"{'|K|':>6}{'squared MSE':>16}{'TV MSE':>16}")
print("-" * 38)
for k in known_counts:
    sq_errors, tv_errors = [], []
    for _ in range(trials):
        rows, cols = pick_known(k)
        sq_errors.append(mse(reconstruct(rows, cols, "squared")))
        tv_errors.append(mse(reconstruct(rows, cols, "tv")))
    sq_curve.append(np.mean(sq_errors))
    tv_curve.append(np.mean(tv_errors))
    print(f"{k:>6}{sq_curve[-1]:>16.6f}{tv_curve[-1]:>16.6f}")


# one worked-out reconstruction per known-pixel count
examples = []
for show_k in known_counts:
    rows, cols = pick_known(show_k)
    sampled = np.full((m, n), 0.5)
    sampled[rows, cols] = img[rows, cols]
    recon_sq = reconstruct(rows, cols, "squared")
    recon_tv = reconstruct(rows, cols, "tv")
    examples.append((show_k, sampled, recon_sq, recon_tv))


fig1, ax = plt.subplots()
ax.plot(known_counts, sq_curve, 's-', label='squared variation')
ax.plot(known_counts, tv_curve, 'o-', label='total variation')
ax.set_xlabel('|K|  (known pixels)')
ax.set_ylabel('mean squared error')
ax.set_title(f'reconstruction error vs known pixels ({trials} instances each)')
ax.invert_xaxis()
ax.legend()
fig1.tight_layout()
fig1.savefig("task2_mse.png", dpi=120)

fig2, axes = plt.subplots(len(examples), 4, figsize=(13, 3.2 * len(examples)))
for row, (show_k, sampled, recon_sq, recon_tv) in enumerate(examples):
    for a, pic, title in [(axes[row, 0], img, "original"),
                          (axes[row, 1], sampled, f"known pixels (|K|={show_k})"),
                          (axes[row, 2], recon_sq, "squared variation"),
                          (axes[row, 3], recon_tv, "total variation")]:
        a.imshow(pic, cmap='gray', vmin=0, vmax=1)
        a.set_title(title)
        a.axis('off')
fig2.tight_layout()
fig2.savefig("task2_reconstruction.png", dpi=120)

print("\nsaved plots to task2_mse.png and task2_reconstruction.png")
plt.show()

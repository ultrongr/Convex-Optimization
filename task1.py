import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

m = 5                 # -> 2m lamps
n = 11                # number of patches (odd)
A, h = 5.0, 1.0       # lamp height and its jitter
mu = 1.0              # lamp spacing
delta = 0.8           # patch endpoint spacing
hprime = 0.3          # patch endpoint vertical jitter
pmax = 1.0
Ides = 0.12


def build_scene():
    lamps = []
    for i in range(1, 2 * m + 1):
        x = -mu / 2 - mu * (m - i)
        y = rng.uniform(A - h, A + h)
        lamps.append((x, y))

    nprime = (n + 1) // 2
    ends = []
    for l in range(1, 2 * nprime + 1):
        x = -delta / 2 - delta * (nprime - l)
        y = rng.uniform(-hprime, hprime)
        ends.append((x, y))
    return np.array(lamps), np.array(ends)


def light_matrix(lamps, ends):
    gain = np.zeros((n, 2 * m))
    for k in range(n):
        # midpoint of patch k and a unit normal pointing up toward the lamps
        mx = (ends[k][0] + ends[k + 1][0]) / 2
        my = (ends[k][1] + ends[k + 1][1]) / 2
        tx = ends[k + 1][0] - ends[k][0]
        ty = ends[k + 1][1] - ends[k][1]
        length = np.sqrt(tx ** 2 + ty ** 2)
        nx, ny = -ty / length, tx / length
        for j in range(2 * m):
            dx = lamps[j][0] - mx
            dy = lamps[j][1] - my
            r = np.sqrt(dx ** 2 + dy ** 2)
            cos_theta = (dx * nx + dy * ny) / r
            gain[k][j] = max(cos_theta, 0) / r ** 2
    return gain


lamps, ends = build_scene()
gain = light_matrix(lamps, ends)


def intensity(p):
    return gain @ p                      # light reaching each patch


def cost(p):
    I = np.maximum(intensity(p), 1e-12)  # clamp so log never sees 0
    return np.max(np.abs(np.log(I) - np.log(Ides)))


# 1. same power on every lamp -> just try many values and keep the best
def uniform_power():
    best_p, best_cost = 0.0, np.inf
    for power in np.linspace(0, pmax, 1000):
        c = cost(np.full(2 * m, power))
        if c < best_cost:
            best_cost, best_p = c, power
    return np.full(2 * m, best_p)


# 2. plain least squares, then push anything out of range back into [0, pmax]
def clipped_lstsq():
    target = np.full(n, Ides)
    p = np.linalg.lstsq(gain, target, rcond=None)[0]
    return p, np.clip(p, 0, pmax)


# 3. least squares with an extra pull toward the middle power pmax/2,
#    strengthened each round until no lamp is out of range
def reweighted_lstsq():
    target = np.full(n, Ides)
    middle = pmax / 2
    w = np.full(2 * m, 1e-4)
    for step in range(1, 200):
        # stack the pull equations under the light equations and solve one lstsq
        pull = np.diag(np.sqrt(w))
        big_matrix = np.vstack([gain, pull])
        big_target = np.concatenate([target, np.sqrt(w) * middle])
        p = np.linalg.lstsq(big_matrix, big_target, rcond=None)[0]

        outside = (p < -1e-9) | (p > pmax + 1e-9)
        if not outside.any():
            return np.clip(p, 0, pmax), step
        w[outside] *= 2
    return np.clip(p, 0, pmax), step


# 4. minimize the worst linear error max_k |I_k - Ides| as a linear program.
#    unknowns are the 2m powers plus one slack t that bounds the worst error.
def minmax_lp():
    objective = np.zeros(2 * m + 1)
    objective[-1] = 1                    # minimize t

    A_ub, b_ub = [], []
    for k in range(n):
        row = list(gain[k]) + [-1]       #  I_k - t <= Ides
        A_ub.append(row); b_ub.append(Ides)
        row = list(-gain[k]) + [-1]      # -I_k - t <= -Ides
        A_ub.append(row); b_ub.append(-Ides)

    bounds = [(0, pmax)] * (2 * m) + [(0, None)]
    res = linprog(objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return res.x[:2 * m]


# 5. exact convex solution for min max_k h(I_k/Ides), h(u)=max(u, 1/u).
def exact_ratio():
    p = cp.Variable(2 * m)
    I = gain @ p
    h = cp.maximum(I / Ides, Ides * cp.inv_pos(I))

    problem = cp.Problem(
        cp.Minimize(cp.max(h)),
        [
            0 <= p,
            p <= pmax,
            I >= 1e-9,
        ],
    )
    problem.solve()
    return p.value


p1 = uniform_power()
p2_raw, p2 = clipped_lstsq()
p3, iters = reweighted_lstsq()
p4 = minmax_lp()
p5 = exact_ratio()

print(f"instance: {2*m} lamps, {n} patches, pmax={pmax}, Ides={Ides}")
print(f"method 3 converged in {iters} iterations\n")
print("method 2 raw least-squares powers:")
print(np.array2string(p2_raw, precision=3, suppress_small=True))
print("method 2 after clipping to [0, pmax]:")
print(np.array2string(p2, precision=3, suppress_small=True), "\n")
print(f"{'method':<34}{'cost':>10}")
print("-" * 44)
for name, p in [("1. uniform power", p1),
                ("2. clipped least squares", p2),
                ("3. reweighted least squares", p3),
                ("4. min-max LP (linear error)", p4),
                ("5. exact  min-max |log ratio|", p5)]:
    print(f"{name:<34}{cost(p):>10.4f}")


patches = np.arange(1, n + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

ax[0].scatter(lamps[:, 0], lamps[:, 1], marker='o', s=90, c='gold',
              edgecolors='k', zorder=3, label='lamps')
ax[0].plot(ends[:, 0], ends[:, 1], '-', color='sienna', lw=2, label='patches')
ax[0].axhline(0, color='0.8', lw=0.8, zorder=0)
ax[0].set_title('problem geometry')
ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].legend()

for name, p, style in [("uniform", p1, 's-'), ("clip lstsq", p2, '^-'),
                       ("reweighted", p3, 'v-'), ("min-max LP", p4, 'd-'),
                       ("exact", p5, 'o-')]:
    ax[1].plot(patches, intensity(p), style, ms=4, label=name)
ax[1].axhline(Ides, color='k', ls='--', lw=1, label='Ides')
ax[1].set_title('illumination per patch')
ax[1].set_xlabel('patch k'); ax[1].set_ylabel('I_k'); ax[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("task1_illumination.png", dpi=120)
print("\nsaved plot to task1_illumination.png")
plt.show()

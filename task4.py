import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# five sensors at known positions, each with a noisy distance estimate to the source
y = np.array([[1.8, 2.5],
              [2.0, 1.7],
              [1.5, 1.5],
              [1.5, 2.0],
              [2.5, 1.5]])
d = np.array([2.00, 1.24, 0.59, 1.31, 1.44])
m = len(d)

# residual r_k = t - 2 y_k^T x + ||y_k||^2 - d_k^2 is affine in z = (x1, x2, t):
# r = A z + c
A = np.column_stack([-2 * y[:, 0], -2 * y[:, 1], np.ones(m)])
c = (y ** 2).sum(axis=1) - d ** 2

AtA = A.T @ A
Atc = A.T @ c
cc = c @ c
D = np.diag([1.0, 1.0, 0.0])          # the constraint x^T x - t = z^T D z + e^T z
e = np.array([0.0, 0.0, -1.0])

# dual of the single-constraint QCQP as an SDP. the unknowns are the equality
# multiplier nu (free) and a scalar gamma that lower-bounds the dual value.
nu = cp.Variable()
gamma = cp.Variable()
p = Atc + (nu / 2) * e
lmi = cp.bmat([[AtA + nu * D,                    cp.reshape(p, (3, 1), order='F')],
               [cp.reshape(p, (1, 3), order='F'), cp.reshape(cc - gamma, (1, 1), order='F')]])
cp.Problem(cp.Maximize(gamma), [lmi >> 0]).solve()

# knowing nu*, KKT stationarity turns the recovery of z* into one linear solve
nu_star = nu.value
z = -np.linalg.solve(AtA + nu_star * D, Atc + (nu_star / 2) * e)
x = z[:2]
t = z[2]


def objective(point):
    return np.sum((np.sum((point - y) ** 2, axis=1) - d ** 2) ** 2)


print(f"nu*        = {nu_star:.4f}")
print(f"x*         = ({x[0]:.4f}, {x[1]:.4f})")
print(f"t*         = {t:.4f}   ||x*||^2 = {x @ x:.4f}   (equal at the optimum)")
print(f"objective  = {objective(x):.4f}   dual value = {gamma.value:.4f}")


colors = plt.cm.tab10(np.arange(m))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
for k in range(m):
    yx, yy = y[k]
    ax.add_patch(plt.Circle((yx, yy), d[k], fill=False, color=colors[k], lw=1.4, alpha=0.9))
    ax.scatter(yx, yy, marker='^', s=100, color=colors[k], zorder=3, label=f"sensor {k + 1}")
    ax.annotate(f"d={d[k]:.2f}", (yx, yy), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.scatter(x[0], x[1], marker='*', s=260, c='black', zorder=4, label='estimated source')

pad = 0.3
ax.set_xlim((y[:, 0] - d).min() - pad, (y[:, 0] + d).max() + pad)
ax.set_ylim((y[:, 1] - d).min() - pad, (y[:, 1] + d).max() + pad)
ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('range circles and estimated source position')
ax.legend()
fig.tight_layout()
fig.savefig("task4_map.png", dpi=120)

print("\nsaved plot to task4_map.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


# each node's local cost is f_i(x) = x^T A_i x + b_i^T x, with A_i a rotated
# diagonal (so it stays symmetric positive definite) and b_i a rotated unit vector
def rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


A, b = [], []
for i in [1, 2, 3, 4]:
    A.append(rot(i * np.pi / 4) @ np.diag([1.0, 1 / 2 ** i]) @ rot(i * np.pi / 4).T)
    b.append(rot(i * np.pi / 8) @ np.array([1.0, 0.0]))


# --- part 1: one central node knows every f_i and solves it directly ---
x_star = -0.5 * np.linalg.solve(sum(A), sum(b))
print("part 1  central optimum x* =", x_star)


# --- part 2: the 4-cycle 1-2-4-3-1, and the matrices it induces ---
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
neighbors = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}

Adj = np.zeros((4, 4))
for i, j in edges:
    Adj[i, j] = Adj[j, i] = 1
degree = Adj.sum(axis=1)
W = Adj / degree[:, None]                       # W_ij = 1/|n(i)|

B = np.zeros((len(edges), 4))                    # weighted edge-vertex incidence
for e, (i, j) in enumerate(edges):
    w = np.sqrt(1 / degree[i])                   # every edge weight is 1/2
    B[e, i], B[e, j] = w, -w
Lam = B.T @ B                                    # Laplacian, equals I - W here

print("\npart 2  W =\n", W)
print("\npart 2  B =\n", B)
print("\npart 2  Lambda = B^T B =\n", Lam)


# --- part 4: distributed gradient descent on the penalty objective ---
# each node keeps its own estimate x_i and only talks to its neighbours
c = 5.0                                          # penalty weight from part 3


def run_dgd(alpha, steps):
    x = np.zeros((4, 2))
    history = []
    for k in range(steps):
        grad = np.zeros((4, 2))
        for i in range(4):
            pull = sum(x[i] - x[j] for j in neighbors[i])
            grad[i] = 2 * A[i] @ x[i] + b[i] + c * pull
        x = x - alpha * grad
        history.append(np.linalg.norm(x - x_star, axis=1))
        if history[-1].max() > 1e6:              # diverged, stop early
            break
    return np.array(history)


good_alpha = 0.06
dgd_hist = run_dgd(good_alpha, 3000)
print(f"\npart 4  DGD (alpha={good_alpha}, c={c}) node errors to x*:",
      dgd_hist[-1].round(4))


# --- part 6: distributed dual ascent, inner minimisation solved in closed form ---
# f_i is quadratic, so argmin_x [ f_i(x) + p_i^T x ] = -0.5 A_i^-1 (b_i + p_i);
# no inner gradient-descent loop is needed, we solve each node exactly
def run_dual(beta, steps):
    nu = np.zeros((len(edges), 2))               # one multiplier per edge
    x = np.zeros((4, 2))
    history = []
    for k in range(steps):
        p = np.zeros((4, 2))
        for e, (i, j) in enumerate(edges):
            p[i] += nu[e]
            p[j] -= nu[e]
        for i in range(4):
            x[i] = -0.5 * np.linalg.solve(A[i], b[i] + p[i])
        for e, (i, j) in enumerate(edges):
            nu[e] += beta * (x[i] - x[j])
        history.append(np.linalg.norm(x - x_star, axis=1))
    return np.array(history)


beta = 0.1
dual_hist = run_dual(beta, 250)
print(f"part 6  dual ascent (beta={beta}) node errors to x*:",
      dual_hist[-1].round(6))


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig1, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for i in range(4):
    ax[0].plot(dgd_hist[:, i], color=colors[i], label=f"node {i + 1}")
ax[0].set_title(f"DGD per-node error (alpha={good_alpha}, c={c})")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("||x_i - x*||"); ax[0].legend()

for alpha in [0.02, 0.06, 0.09, 0.12]:
    h = run_dgd(alpha, 400)
    ax[1].plot(h.max(axis=1), label=f"alpha={alpha}")
ax[1].set_yscale('log')
ax[1].set_title("DGD worst-node error vs step size")
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("max_i ||x_i - x*||"); ax[1].legend()
fig1.tight_layout()
fig1.savefig("task5_dgd.png", dpi=120)

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
for i in range(4):
    ax2.plot(dual_hist[:, i], color=colors[i], label=f"node {i + 1}")
ax2.set_yscale('log')
ax2.set_title(f"dual ascent per-node error (beta={beta})")
ax2.set_xlabel("iteration"); ax2.set_ylabel("||x_i - x*||"); ax2.legend()
fig2.tight_layout()
fig2.savefig("task5_dual.png", dpi=120)

print("\nsaved plots to task5_dgd.png and task5_dual.png")
plt.show()

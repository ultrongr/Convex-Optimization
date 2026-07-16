# Convex Optimization — lab

Five tasks from [ergasia.pdf](ergasia.pdf), one script each. The full write-up with the
maths, the results and the discussion is in `Report.pdf`; this file is just how to run
things and what each script produces.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Needs numpy, scipy, matplotlib and cvxpy.

## Running

Each task is a standalone script — run it from the project root, since tasks 2 and 3
read their images from `Data/`.

```
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
```

Every script prints its results to the terminal, saves its plots as PNGs next to the
script, and then opens them in a window (close it and the script exits). Task 2 is the
slow one — it solves 80 CVXPY problems over a 64×64 image (4 sampling levels × 2
roughness measures × 10 random instances). The rest finish quickly.

Everything is seeded (`default_rng(7)`), so repeated runs give the same numbers.

## Outputs

### Task 1 — illumination

The same instance solved five ways. Prints the instance parameters, how many iterations
the reweighted least squares took, the raw and clipped least-squares powers, and a table
of each method against its worst-case log deviation.

- `task1_illumination.png` — the lamp/patch geometry, and the illumination each method
  puts on every patch against `Ides`.

### Task 2 — image inpainting

Reconstructs `server64.png` from a random subset of its pixels. Prints a table of known
pixel count against the mean squared error for both roughness measures, averaged over 10
random instances.

- `task2_mse.png` — the two MSE curves against the number of known pixels.
- `task2_reconstruction.png` — one row per sampling level: original, known pixels, and
  both reconstructions.

### Task 3 — colourisation

Puts the colour back into a grayscale `flower.png` given the true colour at a few random
pixels. Prints a table of known-colour count against the final tv cost and the RGB MSE.

- `task3_colorization.png` — original, grayscale input, and the four colourings.
- `task3_cases.png` — each case as a pair: the grayscale image with its known colours
  sprinkled in, next to the colouring it produced.

### Task 4 — source localisation

Solves the non-convex range-fitting problem through its SDP dual. Prints the optimal
multiplier `nu*`, the recovered position `x*`, `t*` alongside `||x*||^2` as a feasibility
check, and the objective alongside the dual value.

- `task4_map.png` — the sensors, a range circle around each, and the estimated source.

### Task 5 — distributed optimization

Four nodes on a 4-cycle solving for a common `x*`. Prints the central optimum, the
weight/incidence/Laplacian matrices the graph induces, and the per-node distances to `x*`
reached by distributed gradient descent and by dual ascent.

- `task5_dgd.png` — per-node error for the penalty method, and a step-size sweep showing
  where it diverges.
- `task5_dual.png` — per-node error for dual ascent, on a log scale.

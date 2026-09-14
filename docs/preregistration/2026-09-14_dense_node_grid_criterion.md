# Dense-node threshold grid: pass criterion and sweep plan

**Recorded and committed BEFORE any alternative edge range is evaluated.** The commit that
adds this file must precede the commit that adds the sweep results; that ordering is the
evidence that the criterion was not chosen after seeing which range passes.

PI decision: 2026-09-14. Agent: Claude (Opus), implementing.

## Why this exists

On real dense-node scores (Pythia-160M @ `50f5173d`, EAP, n=300, no BOS, float32), the
pre-registered Q4 grid (anti-diagonal, B=16, node 0.6->0.9, edge 0.99->0.95) produced
largely nested views:

| run | mean pairwise Jaccard | Match | consensus / smallest view | nested pairs |
|---|---|---|---|---|
| ioi seed 0 | 0.477 | no | 73 / 81 | 89/120 |
| ioi seed 1 | 0.491 | no | 81 / 93 | 82/120 |
| greater_than seed 0 | 0.493 | **yes** | 16 / 16 | 116/120 |
| greater_than seed 1 | 0.475 | no | 15 / 16 | 97/120 |

Node and edge counts both rise monotonically along the grid (ioi: 14/81 -> 51/827), so the
node axis dominates and the edge axis walking 0.99->0.95 does not counteract it. The
pre-registered stop rule in `configs/ensemble/default.yaml` ("Jaccard near 1.0", "Match
~100%") is NOT triggered; its purpose - views that prune different parts of the graph -
largely fails. No null has been drawn, no model compressed, nothing frozen.

## What is fixed

- Scope: the **dense-node pipeline only**. Pipeline A keeps the anchored Q4 ranges.
- Design: anti-diagonal, B = 16, node_range = (0.6, 0.9) walked low -> high.
- Semantics: unchanged (thresholds applied to total-effect EAP scores; `dense_prune.py`).
- Only the edge range's LOW end varies; its high end stays 0.99.

## The candidate list, in evaluation order (every one is reported)

`edge_range = (0.99, low)` for `low` in **0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30**.

0.95 is the current pre-registered range, included as the reference row.

## Pass criterion (per candidate)

On **all four** runs (ioi and greater_than, seeds 0 and 1):

1. **Match is false** - the consensus (edges present in all 16 views) is not equal to any
   single view's edge set; **and**
2. **nested view pairs <= 60 of 120** (50%) - a pair (i, j) is nested if one view's edge
   set is a subset of the other's.

Edge sets are the kept edges returned by `prune_dense_graph`, after dangling-node cleanup.

## Selection rule

The **first candidate in the list above that passes** is selected - the smallest departure
from the anchored range. If **none** passes, no range is selected and the question returns
to the PI. No candidate outside this list may be evaluated for selection.

## Input data (fixed)

The EAP scores already computed on 2026-09-14 for the diagnostics above, stored outside the
repo; sha256 of each file:

| file | sha256 |
|---|---|
| `greater_than_seed0.json` | `50f8c43ce61e07dbbc0ef8f99a20b599afb92cb09b45319e6b69582a8563ee0d` |
| `greater_than_seed1.json` | `c758044e1310a21d441248302120c551bd7f2e6bd40e8b04b27f6116a4a7fd3f` |
| `ioi_seed0.json` | `7506684228259d0f92234d29c9d3f8c809d353e39a968e07cd0029a561d552c1` |
| `ioi_seed1.json` | `ac8d11d9601c9a92ddb6fc2d4f4efb91db3ef442077c5eb78f309f363d43a25c` |

## What is reported for every candidate, pass or fail

Per run: Match, nested pairs, mean pairwise Jaccard, consensus size, smallest-view size,
edges per view (min / max), nodes per view (min / max).

## Known costs, stated now

- **C8 confound.** If a range is selected, the dense-node and pipeline-A grids differ; that
  is a second difference inside the basis-drift comparison and must be reported with it.
- **Seeds 0 and 1 are reused.** They will also be seeds of the Stage A ensemble. The grid is
  chosen on dense-model view structure only - never on any null or compression outcome -
  but the reuse is disclosed.
- **Pythia-160M only.** Whether the selected range holds on Pythia-410M is not tested here.

Recorded 2026-09-14.

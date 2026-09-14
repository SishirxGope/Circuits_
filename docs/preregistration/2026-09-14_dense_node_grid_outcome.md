# Dense-node threshold grid: outcome and PI decision

Follows, and does not modify, the two committed records:

1. `2026-09-14_dense_node_grid_criterion.md` — criterion, candidate list and selection rule,
   committed in `d7c2e58` **before** any alternative was evaluated;
2. `2026-09-14_dense_node_grid_sweep_results.json` — every candidate's result, committed in
   `9798bf9` **after** the criterion.

## Outcome

**No candidate passed; nothing was selected.** Under the committed rule the question returned
to the PI.

- IOI passes from `(0.99, 0.80)` downward. **Greater-than never passes**: it avoids Match only
  while ≥ 69/120 view pairs are nested, and once nesting falls, the consensus collapses onto a
  single view (Match).
- Widening the edge range trades one failure for another: nested pairs fall, but mean pairwise
  Jaccard **rises** from ~0.48 to ~0.79 — views become less nested but more alike.

## PI decision (2026-09-14)

**Keep the Q4 grid exactly as pre-registered, for every pipeline, and report both findings.**

Rationale recorded with the decision:

- **CSI is unaffected.** The dense model, every null draw and every compressed model are
  extracted under the identical grid, so the nesting enters the numerator and the denominator
  alike. What weakens is only the reading of s(e) bands as "robust to analyst choices" on the
  dense-node pipeline.
- **C8 gains no second difference.** A dense-node-specific grid would have added a grid
  difference inside the basis-drift comparison with pipeline A.
- **No second forking path.** A second pre-registration with new threshold semantics would be
  a further attempt after a failed one.

## Explicitly not done

Accepting IOI's pass at `(0.99, 0.80)`. That would have replaced the committed "all four runs"
criterion with a per-task one **after** seeing which runs pass — the exact move pre-registration
exists to prevent. It was ruled out before being offered as an option.

## What the paper must report

1. On the dense-node pipeline, the pre-registered anti-diagonal grid yields largely nested
   views: 82–116 of 120 view pairs nested, consensus 90–100% of the smallest view (Pythia-160M,
   seeds 0–1, both tasks). The config's own stop rule (Jaccard near 1.0, Match ≈ 100%) was not
   triggered.
2. Within the pre-stated family of edge ranges, no range made the views non-nested on all four
   runs, and widening the range raised view similarity.
3. The consequence: dense-node s(e) partly ranks edges by threshold depth rather than by
   genuinely different views, so its core/contingent/noise bands carry less "robust to analyst
   choices" meaning than CIRCUS's.

These are diagnostics of the dense-node pipeline on Pythia-160M. They say nothing about
pipeline A, which cannot run on Pythia and whose own grid diagnostics are still to be run on
the primaries before the freeze.

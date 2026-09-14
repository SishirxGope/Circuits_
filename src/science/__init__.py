# [AI-GEN] agent=Claude date=2026-08-08 task=Consolidate src/{nulls,ensemble,compare,metrics,causal} into one novelty-zone package
# reviewed-by: PENDING

"""src.science — THE NOVELTY PROTECTION ZONE (AI_RULES.md §3), one package.

Everything that implements the paper's contribution lives here. AI agents may READ
these modules and PROPOSE diffs, but must not modify them without explicit human
approval recorded in the log. An agent that "simplifies" a distance function or
"optimizes" the perturbation sampler silently changes the scientific estimand, and
the estimand is the paper.

Contents, by proposal section:

    §2.1(a) matched_magnitude.py   matched-magnitude random perturbation (the null)
    §2.1(b) matched_perplexity.py  matched-perplexity control family
    §2.1(c) circus_wrapper.py      CIRCUS-style B configs x S seeds ensemble runner
            threshold_grid.py      the B non-nested threshold configs (Q4)
    §2.1    csi.py                 CSI = D(c) / median(D_null(c)) + bootstrap CI
            perplexity.py          perplexity evaluation (feeds the matched-PPL null)
    §2.3    inclusion_freq.py      edge inclusion frequency s(e)
            decompose.py           core / contingent / noise decomposition
    §2.4    two_level.py           exact-edge AND routing-head comparison (always paired)
            distances.py           D(c): pre-registered primary + ablation alternative
    §2.5    patch_diagnostic.py    NIE = PIE + INT, grouped-vs-single interaction flag

Also in the novelty zone but outside this package (AI_RULES.md §3):
    analysis/cross_audit.py        circuit-vs-feature damage rank agreement (§2.6)
    frozen/                        the frozen null store (append-only; not even proposals)

Consolidated 2026-08-08 from src/nulls, src/ensemble, src/compare, src/metrics and
src/causal. Module file names, contents and public APIs are unchanged; only the
package path moved, so "the novelty zone" is now one directory instead of five.
"""

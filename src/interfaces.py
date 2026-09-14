# [AI-GEN] agent=OpenCode date=2026-08-07 task=Define core Protocol interfaces per ARCHITECTURE.md §4
# reviewed-by: PENDING

"""Core abstraction layer: the seven interfaces from ARCHITECTURE.md §4.

Design intent (from ARCHITECTURE.md §4):
- Interfaces are deliberately minimal so the ablation matrix (PRD.md §3) is a config
  change, not a refactor.
- ``Compressor.weight_delta`` is the SINGLE SOURCE OF TRUTH for the magnitudes the
  Perturber matches; the null can never drift from the compression it is matched to.
- ``EnsembleRunner`` computes raw attribution once per (model, task, seed) and
  re-prunes under all B configs, per the CIRCUS observation that the ensemble is
  near-free after one attribution pass (arXiv:2603.00523). This is the core compute
  optimization.
- Distances, projectors, extractors, and compressors are all selected by the Hydra
  config key; every ablation row in PRD.md §3 corresponds to overriding exactly one
  key.

Only interfaces are declared here. Implementations live in their own modules
(see CLAUDE.md §2 paper-to-code mapping).
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

# -----------------------------------------------------------------------------------
# Shared domain types (paper-facing glossary terms, CLAUDE.md §5)
# -----------------------------------------------------------------------------------

Model = Any
"""A loaded, dense, full-precision model (TransformerLens HookedTransformer, nnsight
LanguageModel, or HF model). Read-only input pinned by HF revision hash
(ARCHITECTURE.md §2); nothing writes back to it."""

CompressedModel = Any
"""A compressed (quantized / pruned) model plus metadata about the compression
setting that produced it (family, level, resolved-config hash). Regenerable from
configs; safe to delete (ARCHITECTURE.md §2)."""

PerTensorFrobenius = dict[str, float]
"""Mapping tensor_name -> ||Delta W||_F of the compression-induced weight delta.
Tensors with zero change are recorded as 0.0. This is exactly what the
matched-magnitude null (src/science/matched_magnitude.py) must match."""

Graph = Any
"""A circuit: computation graph over model components (heads/MLPs as nodes,
information flow as edges) extracted for a task - NOT a set of feature directions
(CLAUDE.md §5). Must serialize to the ARCHITECTURE.md §2 edges.parquet schema:
(src_component, dst_component, config_id, seed, included: bool), with model-agnostic
component IDs L{layer}.{type}{index} (e.g. L11.H3, L7.MLP)."""

FreqVector = Any
"""Vector of edge inclusion frequencies s(e) in [0,1] over a (B configs x S seeds)
extraction ensemble. The reported circuit object is ALWAYS this vector, never a
binary edge list (CLAUDE.md §5)."""

PatchResult = Any
"""Result of an interaction-aware patching diagnostic; carries (NIE, INT_flag) and
enough context that interaction-dominated components can be excluded from headline
stability claims and reported separately (CLAUDE.md §5)."""

Rng = Any
"""Seeded random generator (numpy Generator or torch.Generator) passed in by the
caller. Never a bare module-level np.random.* / torch.rand* call (AI_RULES.md 1.1)."""

# -----------------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------------


@runtime_checkable
class Compressor(Protocol):
    """Applies one cell of the compression grid (proposal §3.2).

    Families: RTN (INT8->INT4), GPTQ, AWQ; magnitude and Wanda pruning 0-60%.
    ``weight_delta`` feeds the null: the matched-magnitude Perturber must draw random
    Delta W whose per-tensor Frobenius norms equal these magnitudes, so the null can
    never drift from the compression it is matched to (ARCHITECTURE.md §4).
    """

    def apply(self, model: Model, cfg: Any) -> CompressedModel:
        """Return the compressed model for one grid cell.

        Must be side-effect-free with respect to ``model``: raw checkpoints are
        read-only inputs (ARCHITECTURE.md §2). Upstream in-place pruners must operate
        on a deep copy / fresh load from the pinned revision. Deterministic given
        (model, cfg); any seed comes from cfg and is logged (AI_RULES.md 1.1).
        """

    def weight_delta(self, model: Model, cfg: Any) -> PerTensorFrobenius:
        """Per-tensor Frobenius norms of the weight delta induced by this grid cell.

        The single source of truth for the magnitudes the null matches. Must be
        deterministic given (model, cfg) and draw no randomness.
        """


@runtime_checkable
class Perturber(Protocol):
    """Null model (a): matched-magnitude random perturbation (proposal §2.1a).

    Draws a random Delta W whose per-tensor Frobenius norms match the supplied
    magnitudes (produced by ``Compressor.weight_delta``) and applies it to a copy of
    the dense model. Every draw uses the caller-provided seeded rng (AI_RULES.md
    1.1). The R draws per cell become D_null and are frozen at Stage B (frozen/);
    the null may never be re-tuned afterward (AI_RULES.md 1.4).
    """

    def apply(self, model: Model, magnitudes: PerTensorFrobenius, rng: Rng) -> Model:
        """Return a perturbed copy of ``model`` with matched-magnitude noise.

        The dense model is never mutated; perturbation applies to a copy.
        """


@runtime_checkable
class CircuitExtractor(Protocol):
    """Extracts ONE circuit graph for a task (proposal §2.2).

    Three implementations (CLAUDE.md §2):
      pipeline A   attribution-patching graph   (wraps circuit-tracer)
      pipeline B   edge-pruning graph
      dense-node   dictionary-free variant measuring the basis-drift confound (C8)

    Each call yields one graph under one (threshold config, seed) ensemble cell. The
    B x S grid is orchestrated by EnsembleRunner, not this class; extraction
    thresholds come from configs/ensemble/ and are never hardcoded (CIRCUS,
    arXiv:2603.00523).
    """

    def extract(self, model: Model, task: Any, config: Any, seed: int) -> Graph:
        """Run one attribution/edge-pruning pass; return the thresholded graph.

        Must be deterministic given (model, task, config, seed): every random draw
        inside is seeded from ``seed`` and logged (AI_RULES.md 1.1). Raises if the
        required upstream artifacts (transcoder set, calibration cache) are absent.
        """


class EnsembleRunner(Protocol):
    """CIRCUS wrapper (proposal §2.1c): B non-nested threshold configs x S seeds.

    Computes raw attribution ONCE per (model, task, seed) and re-prunes under all B
    configs, per the CIRCUS observation (arXiv:2603.00523) that the ensemble is
    near-free after one attribution pass (ARCHITECTURE.md §4; compute plan §5).
    Returns the inclusion-frequency vector s(e) - the only reported circuit object.
    """

    def run(
        self,
        model: Model,
        task: Any,
        configs: Sequence[Any],
        seeds: Sequence[int],
    ) -> FreqVector:
        """Return s(e) over the full (B x S) ensemble.

        All seeds used must be written to the run log (AI_RULES.md 1.1). The B x S
        and seed counts must be legible from the run name (CLAUDE.md §4).
        """


@runtime_checkable
class Distance(Protocol):
    """D(c): distance between pre- and post-compression frequency vectors.

    Pre-registered primary + >=1 ablation alternative (PRD.md §3, P0). The choice is
    a scientific decision (TODO [QUESTION FOR PI], CLAUDE.md §5): CSI is entirely
    downstream of it, so it must be selected in Stage A, pre-registered, and a
    sensitivity check over at least one alternative distance reported in the
    appendix. Never re-implemented inline in analysis scripts (ARCHITECTURE.md §3).
    """

    def __call__(self, f_pre: FreqVector, f_post: FreqVector) -> float:
        """Distance between two s(e) vectors; 0 iff identical."""


@runtime_checkable
class LevelProjector(Protocol):
    """Projects a circuit to a level of description (proposal §2.4).

    identity       -> exact-edge level
    routing-head   -> routing-head-set level (deterministic, versioned in
                      src/science/two_level.py per ARCHITECTURE.md §2)

    Every claim is stated at BOTH levels; the levels can disagree (ref arXiv:2607.18921
    reports exact-edge Jaccard@10 0.14-0.16 vs routing-head 0.55-0.67), so a
    conclusion holding at only one level is reported as level-specific, never
    generalized (claim C2).
    """

    def project(self, g: FreqVector) -> FreqVector:
        """Project a frequency vector onto this comparison level."""


class PatchDiagnostic(Protocol):
    """Interaction-aware patching diagnostic (proposal §2.5; claim C4).

    Computes NIE (= PIE + INT, ref arXiv:2606.27510) and a grouped-vs-single INT
    flag. Components flagged interaction-dominated are EXCLUDED from headline
    stability claims and reported separately (CLAUDE.md §5). The quantitative
    INT-flag rule is pre-registered before Stage D (AI_RULES.md 4.2;
    TODO [QUESTION FOR PI]).
    """

    def run(self, model: Model, edge: Any, siblings: Sequence[Any]) -> PatchResult:
        """Return a PatchResult carrying (NIE, INT_flag) for ``edge``.

        ``siblings`` are the sibling states whose grouping determines the
        grouped-vs-single comparison.
        """

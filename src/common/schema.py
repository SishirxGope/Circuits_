# [AI-GEN] agent=OpenCode date=2026-08-07 task=Define core schemas (ARCHITECTURE.md §2): graphs, freq vectors, parquet columns, run/frozen/CSI metadata
# reviewed-by: PENDING

"""Core schema layer (ARCHITECTURE.md §2, §3).

Defines the canonical data structures and on-disk schemas used by every stage:

- ``Edge`` / ``Graph``: a circuit is a computation graph over components (heads/MLPs
  as nodes, information flow as edges), NOT a set of feature directions
  (CLAUDE.md §5). Component IDs are model-agnostic: ``L{layer}.{type}{index}``
  (e.g. ``L11.H3``, ``L7.MLP``).
- ``FreqVector``: edge inclusion frequencies s(e) in [0,1] over a (B x S) extraction
  ensemble. The reported circuit object is ALWAYS this vector, never a binary edge
  list (CLAUDE.md §5).
- ``edges.parquet`` columns: ``src_component, dst_component, config_id, seed, included``
- ``freq.parquet``   columns: ``edge_id, s_e, band``
- Run metadata (required tracking tags, ARCHITECTURE.md §3).
- Frozen-store ``meta.json`` structure (ARCHITECTURE.md §2).
- CSI table placeholder columns (ARCHITECTURE.md §2).

No scientific logic lives here (distance choice, band cutoffs, CSI aggregation are
NOT in this module); schemas and validators only. Parquet I/O requires pyarrow
(optional dependency; import is deferred and raises an informative error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypedDict

# -----------------------------------------------------------------------------------
# On-disk column schemas (ARCHITECTURE.md §2)
# -----------------------------------------------------------------------------------

EDGES_COLUMNS: tuple[str, ...] = ("src_component", "dst_component", "config_id", "seed", "included")
FREQ_COLUMNS: tuple[str, ...] = ("edge_id", "s_e", "band")
CSI_TABLE_COLUMNS: tuple[str, ...] = (
    "model",
    "task",
    "family",
    "level_param",
    "comparison_level",
    "csi",
    "ci_lo",
    "ci_hi",
    "D",
    "d_normalized",     # Q2 pre-registered 2026-09-12: L1 / |edge union|. Raw D scales
                        # with circuit size and cells differ in edge count, so the two
                        # are reported together and neither replaces the other.
    "dnull_median",
    "null_frozen_hash",
)


def json_safe(obj: Any) -> Any:
    """Recursively replace NaN/±Inf with None so the result is STRICT-valid JSON.

    ``json.dump`` emits bare ``NaN`` / ``Infinity`` by default. Those are Python-only
    extensions: ``json.loads(..., parse_constant=...)`` tolerates them but strict
    parsers (jq, most JS/Go/Rust readers, and anything a reviewer runs on the artifact)
    reject the file outright. NaN is a *normal* result here — an undefined Spearman on
    two cells, a CI that could not be formed — so run reports hit this routinely and
    must still round-trip.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def edge_id(src_component: str, dst_component: str) -> str:
    """Canonical edge identifier: ``"L11.H3->L7.MLP"``.

    Deterministic and model-agnostic; used as the freq.parquet ``edge_id`` key.
    """
    return f"{src_component}->{dst_component}"


# -----------------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One edge-inclusion record for one (config_id, seed) ensemble cell."""

    src_component: str
    dst_component: str
    config_id: str
    seed: int
    included: bool = True

    def edge_id(self) -> str:
        return edge_id(self.src_component, self.dst_component)


@dataclass(frozen=True)
class Graph:
    """A circuit: computation graph over model components for one (config, seed) cell.

    ``edges`` holds the thresholded edge set for this cell (all ``included=True``
    unless a pipeline materializes explicit non-inclusions). The ensemble object of
    record is the inclusion-frequency vector computed over many such graphs
    (CLAUDE.md §5), never a single binary graph.
    """

    edges: tuple[Edge, ...] = ()
    nodes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def included_edge_ids(self) -> tuple[str, ...]:
        """Sorted, de-duplicated edge ids included in this graph."""
        return tuple(sorted({e.edge_id() for e in self.edges if e.included}))


@dataclass
class FreqVector:
    """Edge inclusion frequencies s(e) in [0,1] over a (B x S) ensemble.

    ``bands`` maps edge_id -> band label (core/contingent/noise) once the
    decomposition is applied (src/science/decompose.py); None until then.
    """

    frequencies: dict[str, float] = field(default_factory=dict)
    bands: dict[str, str | None] = field(default_factory=dict)

    def items_sorted(self) -> list[tuple[str, float]]:
        """Deterministic (edge_id, s_e) pairs sorted by edge_id."""
        return sorted(self.frequencies.items(), key=lambda kv: kv[0])

    def band_for(self, edge_id_: str) -> str | None:
        return self.bands.get(edge_id_)


@dataclass(frozen=True)
class EnsembleResult:
    """Output of an EnsembleRunner: per-cell edge records + the frequency vector."""

    records: tuple[dict[str, Any], ...]
    freq: FreqVector
    n_cells: int


# -----------------------------------------------------------------------------------
# Run metadata (ARCHITECTURE.md §3 required tags)
# -----------------------------------------------------------------------------------


class RunMetadata(TypedDict, total=False):
    """Required tracking tags (ARCHITECTURE.md §3), enforced by ``validate_run_tags``.

    ``null_frozen_hash`` is required only for Stage C/D (the hash the CSI division
    must verify; AI_RULES.md 1.4). ``R`` is required only for stages that draw nulls
    (B/C/D). ``git_commit`` / ``config_hash`` are recorded for every run.
    """

    stage: str
    model: str
    task: str
    compression_family: str
    compression_level: str | None
    comparison_level: str
    pipeline: str
    B: int
    S: int
    R: int
    seed: int
    null_frozen_hash: str | None
    git_commit: str | None
    config_hash: str


_COMMON_REQUIRED_TAGS: frozenset[str] = frozenset(
    {"stage", "model", "task", "compression_family", "comparison_level", "pipeline", "B", "S", "seed"}
)


def validate_run_tags(tags: Mapping[str, Any], stage: str | None = None) -> None:
    """Validate required tracking tags; raise ValueError with the missing keys.

    Stage-dependent requirements (ARCHITECTURE.md §3; AI_RULES.md 1.4):
    - every run: stage, model, task, compression_family, comparison_level, pipeline,
      B, S, seed
    - Stage B/C/D additionally require R > 0 (null draws)
    - Stage C/D additionally require a non-null null_frozen_hash (frozen null must
      already exist; the guard enforces it on disk too)
    """
    if stage is None:
        stage = str(tags.get("stage", ""))
    if stage not in ("stageA", "stageB", "stageC", "stageD"):
        raise ValueError(f"invalid stage tag: {stage!r} (expected stageA|stageB|stageC|stageD)")

    missing = sorted(k for k in _COMMON_REQUIRED_TAGS if tags.get(k) in (None, ""))
    if stage in ("stageB", "stageC", "stageD"):
        if not int(tags.get("R", 0) or 0) > 0:
            missing.append("R")
    if stage in ("stageC", "stageD"):
        if tags.get("null_frozen_hash") in (None, ""):
            missing.append("null_frozen_hash")
    if missing:
        raise ValueError(f"run tags missing required fields for {stage}: {missing}")


# -----------------------------------------------------------------------------------
# Frozen-store structures (ARCHITECTURE.md §2; append-only, AI_RULES.md 1.4)
# -----------------------------------------------------------------------------------


class FrozenMeta(TypedDict, total=False):
    """Structure of ``frozen/{model}/{task}/{cell}/meta.json``.

    Contents per ARCHITECTURE.md §2: generation config hash, seeds, R, per-tensor
    Frobenius magnitudes used, timestamp, git commit — plus the cell's
    ``null_frozen_hash`` that CSI must verify before dividing (AI_RULES.md 1.4).
    """

    config_hash: str
    seeds: list[int]
    R: int
    frobenius_magnitudes: dict[str, float]
    timestamp: str
    git_commit: str | None
    null_frozen_hash: str
    dnull_file: str


def validate_frozen_meta(meta: Mapping[str, Any]) -> None:
    """Raise ValueError if a frozen-cell meta.json is missing required keys."""
    missing = sorted(k for k in ("config_hash", "seeds", "R", "null_frozen_hash") if meta.get(k) in (None, ""))
    if missing:
        raise ValueError(f"frozen meta.json missing required fields: {missing}")


# -----------------------------------------------------------------------------------
# CSI table (ARCHITECTURE.md §2 schema; values produced only by Stage C/D)
# -----------------------------------------------------------------------------------


class CSIRow(TypedDict, total=False):
    """One row of the CSI table (ARCHITECTURE.md §2; CLAUDE.md §5 CSI definition)."""

    model: str
    task: str
    family: str
    level_param: str
    comparison_level: str
    csi: float
    ci_lo: float
    ci_hi: float
    D: float
    dnull_median: float
    null_frozen_hash: str


def write_csi_placeholder(path: str) -> None:
    """Write a header-only, visually-marked PLACEHOLDER CSI table (AI_RULES.md 2.3).

    Placeholder files must be marked PLACEHOLDER and can never share a file with
    real results (AI_RULES.md 2.3). Real CSI rows are produced only by Stage C/D.
    """
    lines = ["# PLACEHOLDER - no real results yet (AI_RULES.md 2.3)", ",".join(CSI_TABLE_COLUMNS)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------------
# Record validators (schema tests rely on these)
# -----------------------------------------------------------------------------------


def validate_edges_record(record: Mapping[str, Any]) -> None:
    """Validate one edges.parquet row; raise ValueError on missing/bad fields."""
    missing = [c for c in ("src_component", "dst_component", "config_id", "seed") if record.get(c) in (None, "")]
    if missing:
        raise ValueError(f"edges record missing required fields: {missing}")
    if not isinstance(record["src_component"], str) or not isinstance(record["dst_component"], str):
        raise ValueError("edges record: src_component/dst_component must be strings")
    if not isinstance(record["seed"], int):
        raise ValueError("edges record: seed must be an int")
    if not isinstance(record.get("included", True), bool):
        raise ValueError("edges record: included must be a bool")


def validate_freq_record(record: Mapping[str, Any]) -> None:
    """Validate one freq.parquet row; raise ValueError on missing/bad fields."""
    missing = [c for c in ("edge_id", "s_e") if record.get(c) in (None, "")]
    if missing:
        raise ValueError(f"freq record missing required fields: {missing}")
    if not isinstance(record["s_e"], (int, float)) or not (0.0 <= float(record["s_e"]) <= 1.0):
        raise ValueError(f"freq record: s_e must be in [0, 1], got {record['s_e']!r}")
    band = record.get("band")
    if band is not None and band not in ("core", "contingent", "noise"):
        raise ValueError(f"freq record: band must be core|contingent|noise|None, got {band!r}")


def validate_edges_records(records: Sequence[Mapping[str, Any]]) -> None:
    for r in records:
        validate_edges_record(r)


def validate_freq_records(records: Sequence[Mapping[str, Any]]) -> None:
    for r in records:
        validate_freq_record(r)


# -----------------------------------------------------------------------------------
# Parquet I/O (requires pyarrow)
# -----------------------------------------------------------------------------------


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        return pa
    except ImportError as exc:  # pragma: no cover - dependency not installed
        raise ImportError(
            "pyarrow is required for parquet I/O; add it to requirements (ARCHITECTURE.md §2 schemas)"
        ) from exc


def write_edges_parquet(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Write edges.parquet with columns EDGES_COLUMNS (schema tested in tests/)."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_edges_records(records)
    data = {c: [r.get(c) for r in records] for c in EDGES_COLUMNS}
    pq.write_table(pa.table(data), str(path))


def read_edges_parquet(path: str) -> list[dict[str, Any]]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


def write_freq_parquet(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Write freq.parquet with columns FREQ_COLUMNS (schema tested in tests/)."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    validate_freq_records(records)
    data = {c: [r.get(c) for r in records] for c in FREQ_COLUMNS}
    pq.write_table(pa.table(data), str(path))


def read_freq_parquet(path: str) -> list[dict[str, Any]]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()

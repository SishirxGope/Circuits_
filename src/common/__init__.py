# [AI-GEN] agent=OpenCode date=2026-08-07 task=Create src/common package marker (consolidation of src/io, src/utils, src/guards)
# reviewed-by: PENDING

"""src.common — shared engineering layer (consolidated from src/io + src/utils + src/guards).

- Schema layer: canonical data structures and parquet/JSON schemas
  (ARCHITECTURE.md §2). No scientific logic lives here; schemas only.
- Shared utilities: seeding (AI_RULES.md 1.1), content hashing (AI_RULES.md 1.4),
  and the run-naming convention (CLAUDE.md §4).
- Guards: stage-ordering and frozen-store integrity guards (AI_RULES.md 1.4/1.5).
  Engineering zone; the Stage C/D runners MUST call these before running.

Re-exports the public names of each module so callers can use
``from src.common import schema`` (or ``from src.common.schema import ...``)
without knowing the file layout. Consolidation per the folder-cleanup task
(2026-08-07); every file's content was preserved, only its path changed.
"""

from . import config_guard, hashing, run_naming, schema, seeding, stage_guard
from .config_guard import (
    MODE_ENGINEERING,
    MODE_SCIENTIFIC,
    PI_DECISIONS_DOC,
    assert_engineering_dry_run_limits,
    assert_no_gating_questions,
    mode_of,
    report_open_questions,
    report_provisional_resolutions,
)
from .hashing import generate_manifest, hash_config, hash_dict, hash_file, manifest_entry
from .run_naming import build_configset, build_run_name, sanitize_token, validate_run_name
from .schema import (
    CSIRow,
    EDGES_COLUMNS,
    Edge,
    EnsembleResult,
    FreqVector,
    FrozenMeta,
    Graph,
    RunMetadata,
    edge_id,
    json_safe,
    read_edges_parquet,
    read_freq_parquet,
    validate_edges_record,
    validate_edges_records,
    validate_freq_record,
    validate_freq_records,
    validate_frozen_meta,
    validate_run_tags,
    write_csi_placeholder,
    write_edges_parquet,
    write_freq_parquet,
)
from .seeding import create_seed_generator, derive_child_seed
from .stage_guard import assert_null_frozen_hash, assert_stage_b_frozen

__all__ = [
    "config_guard",
    "hashing",
    "run_naming",
    "schema",
    "seeding",
    "stage_guard",
    "MODE_ENGINEERING",
    "MODE_SCIENTIFIC",
    "PI_DECISIONS_DOC",
    "assert_engineering_dry_run_limits",
    "assert_no_gating_questions",
    "mode_of",
    "report_open_questions",
    "report_provisional_resolutions",
    "generate_manifest",
    "hash_config",
    "hash_dict",
    "hash_file",
    "manifest_entry",
    "build_configset",
    "build_run_name",
    "sanitize_token",
    "validate_run_name",
    "CSIRow",
    "EDGES_COLUMNS",
    "Edge",
    "EnsembleResult",
    "FreqVector",
    "FrozenMeta",
    "Graph",
    "RunMetadata",
    "edge_id",
    "json_safe",
    "read_edges_parquet",
    "read_freq_parquet",
    "validate_edges_record",
    "validate_edges_records",
    "validate_freq_record",
    "validate_freq_records",
    "validate_frozen_meta",
    "validate_run_tags",
    "write_csi_placeholder",
    "write_edges_parquet",
    "write_freq_parquet",
    "create_seed_generator",
    "derive_child_seed",
    "assert_null_frozen_hash",
    "assert_stage_b_frozen",
]

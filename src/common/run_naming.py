# [AI-GEN] agent=OpenCode date=2026-08-07 task=Run-naming convention validation (CLAUDE.md §4)
# reviewed-by: PENDING

"""Run-naming convention (CLAUDE.md §4, mandatory).

Pattern::

    {YYYYMMDD}_{stage}_{model}_{task}_{compression-or-null}_{configset}_seed{S}

- ``stage`` in {stageA, stageB, stageC, stageD} per Algorithm 1. A run whose name
  says stageB (null) can never write into a stageC results directory, and vice versa.
- ``configset`` encodes |B| (pruning configs) and |S| (seeds), plus R (null draws)
  for null runs: ``B16xS5`` or ``B16xS5xR20``, so ensemble size is legible from the
  run name.
- Model/task/setting tokens: letters, digits, hyphens (no underscores — underscore
  is the token separator).
- ``seed{S}`` with an integer S.
"""

from __future__ import annotations

import re
from typing import Any

STAGES: tuple[str, ...] = ("stageA", "stageB", "stageC", "stageD")

_DATE_RE = re.compile(r"^\d{8}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
_CONFIGSET_RE = re.compile(r"^B\d+xS\d+(xR\d+)?$")
_SEED_RE = re.compile(r"^seed(\d+)$")
_SEPARATOR = "_"


def build_configset(B: int, S: int, R: int | None = None) -> str:
    """Config-set token encoding ensemble size (CLAUDE.md §4)."""
    if not (isinstance(B, int) and isinstance(S, int) and B > 0 and S > 0):
        raise ValueError(f"B and S must be positive ints, got B={B!r}, S={S!r}")
    token = f"B{B}xS{S}"
    if R is not None:
        if not (isinstance(R, int) and R > 0):
            raise ValueError(f"R must be a positive int when provided, got {R!r}")
        token += f"xR{R}"
    return token


def _require_token(kind: str, value: str) -> None:
    if not _TOKEN_RE.match(value):
        raise ValueError(f"malformed run name: {kind} token {value!r} must match {_TOKEN_RE.pattern}")


def sanitize_token(value: str) -> str:
    """Coerce a config value into a legal run-name token.

    Run-name tokens allow only ``[A-Za-z0-9-]`` — underscore is the field separator, so
    it cannot appear inside a field, and other punctuation would make the name
    unparseable. But the values that feed the ``setting`` token come from config and
    naturally contain dots and slashes: a compression level is ``0.30``, a family/level
    pair is ``magnitude-0.30``. Left unsanitized, every Stage C cell whose level has a
    decimal point crashes at run-name construction.

    Mapping: any run of illegal characters becomes a single ``-``; leading/trailing
    ``-`` are trimmed. ``magnitude-0.30`` -> ``magnitude-0-30``, ``rtn/int4`` ->
    ``rtn-int4``. Deterministic and stable, so the same cell always names the same run.
    Raises if nothing legal survives, rather than emitting an unnamed run.
    """
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", str(value)).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if not cleaned or not _TOKEN_RE.match(cleaned):
        raise ValueError(
            f"cannot build a legal run-name token from {value!r}: nothing usable remains "
            f"after sanitizing to [A-Za-z0-9-]"
        )
    return cleaned


def validate_run_name(name: str) -> dict[str, Any]:
    """Validate a run name against CLAUDE.md §4; return parsed components.

    Raises ValueError (with the reason) on any malformed component.
    """
    parts = name.split(_SEPARATOR)
    if len(parts) != 7:
        raise ValueError(
            f"malformed run name {name!r}: expected 7 underscore-separated tokens "
            f"{{date}}_{{stage}}_{{model}}_{{task}}_{{setting}}_{{configset}}_seed{{S}}, got {len(parts)}"
        )
    date, stage, model, task, setting, configset, seed_token = parts

    if not _DATE_RE.match(date):
        raise ValueError(f"malformed run name: date {date!r} must be YYYYMMDD")
    if stage not in STAGES:
        raise ValueError(f"malformed run name: stage {stage!r} not in {STAGES}")
    _require_token("model", model)
    _require_token("task", task)
    _require_token("setting", setting)
    if not _CONFIGSET_RE.match(configset):
        raise ValueError(
            f"malformed run name: configset {configset!r} must encode B and S (and R for null runs), "
            f"e.g. B16xS5 or B16xS5xR20"
        )
    m = _SEED_RE.match(seed_token)
    if m is None:
        raise ValueError(f"malformed run name: expected seed token _seed{{S}}, got {seed_token!r}")

    return {
        "date": date,
        "stage": stage,
        "model": model,
        "task": task,
        "setting": setting,
        "configset": configset,
        "seed": int(m.group(1)),
    }


def build_run_name(
    date: str,
    stage: str,
    model: str,
    task: str,
    setting: str,
    configset: str,
    seed: int,
) -> str:
    """Build and validate a run name (CLAUDE.md §4).

    ``model``, ``task`` and ``setting`` are sanitized here rather than at each call
    site. They come straight from config, where underscores are normal and legal
    (``name: greater_than``, cell ``rtn_int8``), but underscore is this format's field
    separator, so an unsanitized value silently adds a field and the name fails
    validation with a confusing "got 8" - which is exactly how every greater_than run
    died. Sanitizing centrally means no stage runner can forget: before this, NONE of
    the four sanitized model or task, and only stages C and D sanitized setting.
    """
    name = _SEPARATOR.join([
        date, stage,
        sanitize_token(model), sanitize_token(task), sanitize_token(setting),
        configset, f"seed{seed}",
    ])
    validate_run_name(name)
    return name


def resolve_run_name(
    supplied: str | None,
    date: str,
    stage: str,
    model: str,
    task: str,
    setting: str,
    configset: str,
    seed: int,
) -> str:
    """Return the run name to use, refusing a supplied name that misstates the run.

    A stage runner knows its own (B, S, R) and grid coordinates, so it can always
    derive a correct name. When a name is supplied anyway (config override, resumed
    run), every component is checked against the derived one — in particular the
    configset token, which must encode B, S and R (CLAUDE.md §4). A supplied
    ``B4xS2`` on a run with R=3 nulls is refused rather than silently recorded: a run
    name that hides the null count is exactly the one that misleads later.
    """
    derived = build_run_name(date, stage, model, task, setting, configset, seed)
    if not supplied:
        return derived

    parsed = validate_run_name(supplied)
    # Compare against the SANITIZED values, matching what build_run_name emits;
    # otherwise a correct supplied name would be rejected for not equalling the raw
    # config value it was legitimately derived from.
    expected = {"stage": stage, "model": sanitize_token(model), "task": sanitize_token(task),
                "setting": sanitize_token(setting), "configset": configset, "seed": int(seed)}
    mismatches = {
        key: (parsed[key], want)
        for key, want in expected.items()
        if parsed[key] != want
    }
    if mismatches:
        detail = "; ".join(f"{k}: name says {got!r} but the run is {want!r}"
                           for k, (got, want) in sorted(mismatches.items()))
        raise ValueError(
            f"supplied run_name {supplied!r} disagrees with the run it names — {detail}. "
            f"Derived name for this run is {derived!r}. Leave run_name unset to let the "
            f"stage derive it (CLAUDE.md §4)."
        )
    return supplied

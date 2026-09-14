# [AI-GEN] agent=Claude date=2026-08-08 task=Static Hydra-config integrity tests (regression for three composition bugs found 2026-08-08)
# reviewed-by: PENDING

"""Hydra config integrity (ARCHITECTURE.md §3).

hydra-core is an optional dependency and is NOT installed in the test environment,
so composition errors used to be invisible until someone ran the entrypoint on the
GPU box. These tests validate the config tree statically with PyYAML, and are
regressions for three real bugs found on 2026-08-08:

1. ``- null: placeholder`` in the defaults list. In YAML a bare ``null`` key is the
   null LITERAL, so the entry parsed as ``{None: "placeholder"}`` and Hydra never
   composed the group. The group directory is now ``configs/nulls/``.
2. ``- ensemble/decompose: decompose_provisional`` pointed at a nested group whose
   directory (``configs/ensemble/decompose/``) did not exist.
3. ``configset: B${B}xS${S}`` inside a group file. Interpolation is ROOT-relative
   after composition, so it must be ``${ensemble.B}``.

They also pin that the Hydra entrypoint and experiments/dry_run_stage_a.py compose
the same B/S/R, which they did not (Hydra R=20 vs driver R=3).
"""

import pathlib
import re

import pytest
import yaml

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"
ROOT_CONFIG = CONFIGS / "config.yaml"

_INTERPOLATION = re.compile(r"\$\{([^}:]+)\}")
_HYDRA_RESOLVERS = ("now", "env", "oc.", "hydra")


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _interpolation_refs(node) -> list[str]:
    """Every ``${ref}`` appearing in a parsed config's VALUES (comments excluded)."""
    if isinstance(node, str):
        return [m.strip() for m in _INTERPOLATION.findall(node)]
    if isinstance(node, dict):
        return [r for v in node.values() for r in _interpolation_refs(v)]
    if isinstance(node, list):
        return [r for v in node for r in _interpolation_refs(v)]
    return []


def _default_entries():
    """(group, option) pairs from the root defaults list, excluding _self_."""
    out = []
    for entry in _load(ROOT_CONFIG)["defaults"]:
        if entry == "_self_":
            continue
        assert isinstance(entry, dict), f"unexpected defaults entry: {entry!r}"
        for group, option in entry.items():
            out.append((group, option))
    return out


class TestDefaultsList:
    def test_root_config_parses(self):
        assert isinstance(_load(ROOT_CONFIG), dict)

    def test_no_default_key_is_a_yaml_null(self):
        """`- null: x` parses as a None key and the group is silently never composed."""
        for group, option in _default_entries():
            assert group is not None, (
                "a defaults entry has a null key — a group literally named `null` must be "
                "quoted or renamed (this repo uses configs/nulls/)"
            )
            assert isinstance(group, str) and group.strip(), f"bad group key: {group!r}"

    def test_no_option_is_a_yaml_null_or_bool(self):
        """YAML coerces bare yes/no/on/off/null; an option name must survive as a string."""
        for group, option in _default_entries():
            assert isinstance(option, str), f"group {group!r} option {option!r} is not a string"

    def test_every_referenced_group_option_file_exists(self):
        for group, option in _default_entries():
            path = CONFIGS / pathlib.Path(group) / f"{option}.yaml"
            assert path.is_file(), (
                f"defaults reference `{group}: {option}` but {path.relative_to(CONFIGS.parent)} "
                "does not exist — Hydra would fail at composition time"
            )

    def test_self_is_last_so_root_values_win(self):
        assert _load(ROOT_CONFIG)["defaults"][-1] == "_self_"


class TestGroupFiles:
    def test_every_group_directory_has_at_least_one_option(self):
        for group_dir in (p for p in CONFIGS.rglob("*") if p.is_dir()):
            assert list(group_dir.glob("*.yaml")) or list(group_dir.iterdir()), (
                f"empty config group directory: {group_dir}"
            )

    def test_all_yaml_files_parse(self):
        for path in CONFIGS.rglob("*.yaml"):
            try:
                _load(path)
            except yaml.YAMLError as exc:  # pragma: no cover - failure path
                pytest.fail(f"{path} is not valid YAML: {exc}")

    def test_group_local_interpolations_are_root_qualified(self):
        """`${B}` inside configs/ensemble/x.yaml resolves against the ROOT, not the group.

        Scans parsed VALUES (not raw text) so prose in comments is not flagged.
        """
        for path in CONFIGS.rglob("*.yaml"):
            if path == ROOT_CONFIG:
                continue
            group = path.parent.relative_to(CONFIGS).as_posix()
            if group == ".":
                continue
            for ref in _interpolation_refs(_load(path)):
                if ref.startswith(_HYDRA_RESOLVERS):
                    continue
                assert "." in ref, (
                    f"{path.name}: interpolation ${{{ref}}} is group-local but Hydra resolves "
                    f"interpolations against the ROOT config — write ${{{group}.{ref}}}"
                )

    def test_no_option_file_declares_a_null_name(self):
        """`name: null` makes the option unidentifiable in the resolved config / run log."""
        for path in CONFIGS.rglob("*.yaml"):
            if path == ROOT_CONFIG:
                continue
            cfg = _load(path)
            if "name" in cfg and path.name != "placeholder.yaml":
                assert cfg["name"] is not None, f"{path} declares `name: null`"


class TestEntrypointsAgree:
    """experiments/dry_run_stage_a.py claims to mirror the Hydra composition."""

    def _composed(self):
        cfg = {}
        for group, option in _default_entries():
            group_cfg = _load(CONFIGS / pathlib.Path(group) / f"{option}.yaml")
            keys = group.split("/")
            node = cfg
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = group_cfg
        return cfg

    def test_hydra_and_driver_agree_on_B_S_R(self):
        from experiments.dry_run_stage_a import engineering_stage_a_cfg

        composed = self._composed()
        driver = engineering_stage_a_cfg(seed=0)
        assert composed["ensemble"]["B"] == driver["ensemble"]["B"]
        assert composed["ensemble"]["S"] == driver["ensemble"]["S"]
        assert composed["nulls"]["R"] == driver["nulls"]["R"], (
            "the Hydra composition and the dry-run driver disagree on R; the driver "
            "docstring promises they mirror each other"
        )

    def test_hydra_and_driver_agree_on_band_cutoffs(self):
        from experiments.dry_run_stage_a import engineering_stage_a_cfg

        composed = self._composed()["ensemble"]["decompose"]
        driver = engineering_stage_a_cfg(seed=0)["ensemble"]["decompose"]
        assert composed["core_threshold"] == driver["core_threshold"]
        assert composed["noise_threshold"] == driver["noise_threshold"]
        assert composed.get("noise_strict") == driver.get("noise_strict"), (
            "the strict-noise flag decides whether s(e) = 0.5 is contingent (CIRCUS) "
            "or noise; a driver that disagrees with the composition silently bands "
            "half the boundary edges differently"
        )

    def test_hydra_default_mode_is_the_engineering_dry_run(self):
        from src.common.config_guard import MODE_ENGINEERING

        assert self._composed()["mode"]["name"] == MODE_ENGINEERING

    def test_default_composition_is_synthetic_end_to_end(self):
        """The default must never be able to touch a real model or dataset."""
        composed = self._composed()
        assert composed["model"].get("synthetic") is True
        assert composed["task"].get("synthetic") is True
        assert composed["mode"]["allow_model_download"] is False
        assert composed["mode"]["allow_stage_c_real_compression"] is False
        assert composed["mode"]["allow_frozen_writes"] is False


class TestOpenDecisionsAreStillMarked:
    """The guard detects OPEN questions from config values; those markers must survive edits."""

    def test_final_band_cutoffs_are_the_preregistered_circus_taxonomy(self):
        """Q1 pre-registered 2026-09-12. Was: assert both cutoffs are still null.

        Asserted by value. CIRCUS section 3.2 is core s(e) = 1, contingent
        0.5 <= s(e) < 1, noise s(e) < 0.5 -- and the strict flag is load-bearing: with
        an inclusive comparison, s(e) = 0.5 would be banded NOISE rather than
        contingent, which at B = 16 mislabels every edge sitting at exactly half the
        views.
        """
        cfg = _load(CONFIGS / "ensemble" / "decompose" / "final.yaml")
        assert cfg["core_threshold"] == 1.0, "CIRCUS core is strict consensus, s(e) = 1"
        assert cfg["noise_threshold"] == 0.5
        assert cfg["noise_strict"] is True, (
            "without noise_strict the 0.5 cutoff is inclusive and contradicts CIRCUS; "
            "the only inclusive encoding is (ceil(B/2) - 1)/B, which hard-codes B"
        )

    def test_the_preregistered_bands_actually_reproduce_circus(self):
        """The config values are only as good as what decompose() does with them."""
        from src.science.decompose import decompose

        cfg = _load(CONFIGS / "ensemble" / "decompose" / "final.yaml")
        bands = decompose(
            {"all": 1.0, "most": 0.75, "half": 0.5, "under": 0.4375, "none": 0.0},
            core_threshold=cfg["core_threshold"],
            noise_threshold=cfg["noise_threshold"],
            noise_strict=cfg["noise_strict"],
        )
        assert bands == {
            "all": "core",           # s = 1 exactly
            "most": "contingent",
            "half": "contingent",    # CIRCUS puts s = 0.5 in contingent, NOT noise
            "under": "noise",
            "none": "noise",
        }

    def test_q3_is_confirmed_at_b16_s5_r20(self):
        """Q3 confirmed 2026-09-14 by Run_Plan's pre-stated rule on the measured timing.

        Was: assert the ensemble is not PI-confirmed. Asserted by value, so a silent change
        to B, S or R after the decision fails here.
        """
        ensemble = _load(CONFIGS / "ensemble" / "default.yaml")
        nulls = _load(CONFIGS / "nulls" / "default.yaml")
        assert (ensemble["B"], ensemble["S"], nulls["R"]) == (16, 5, 20)
        assert ensemble["pi_confirmed"] is True and nulls["pi_confirmed"] is True

    def test_the_chance_floor_universe_decision_is_recorded(self):
        """Decided 2026-09-14; flagged unimplemented so Stage C cannot silently use U*(U-1)."""
        stage_c = _load(CONFIGS / "config.yaml")["stage_c"]
        assert stage_c["chance_floor_universe"] == "structural_edges_among_observed_nodes"
        assert stage_c["chance_floor_universe_implemented"] is False

    def test_q10_q11_are_preregistered(self):
        cfg = _load(CONFIGS / "comparison" / "final.yaml")
        assert cfg["level2_scheme"] == "layer"
        assert cfg["position_policy"] == "aggregate"
        assert cfg["final_pre_registration"] is True
        assert cfg["level2_falsification_condition"], (
            "Q10 was pre-registered WITH a falsification condition; a layer level "
            "recorded without it is not the decision that was approved"
        )

    def test_the_preregistered_grid_is_non_nested(self):
        """Q4's whole point. A nested grid inflates every s(e) toward the loosest view.

        Checked against the generator at the pre-registered B, not just asserted in
        prose: a 4x4 crossed product would pass a 'B == 16' check and fail this one.
        """
        from src.science.threshold_grid import generate_anti_diagonal_grid, is_non_nested

        ens = _load(CONFIGS / "ensemble" / "default.yaml")
        grid_cfg = ens["threshold_grid"]
        assert grid_cfg["design"] == "anti_diagonal"
        node_range = tuple(grid_cfg["node_range"])
        edge_range = tuple(grid_cfg["edge_range"])
        assert node_range[0] < node_range[1], "node axis is walked LOW -> HIGH"
        assert edge_range[0] > edge_range[1], "edge axis must be walked in the OPPOSITE direction"

        grid = generate_anti_diagonal_grid(ens["B"], node_range, edge_range)
        assert len(grid) == ens["B"]
        assert is_non_nested(grid), "the pre-registered grid is nested; see CIRCUS 3.2"

    def test_a_crossed_product_would_be_nested(self):
        """Pins the reasoning behind Q4 so nobody 'simplifies' it back to a product."""
        from itertools import combinations, product

        from src.science.threshold_grid import _dominates

        crossed = [
            {"node_threshold": n, "edge_threshold": e}
            for n, e in product([0.6, 0.7, 0.8, 0.9], [0.99, 0.98, 0.97, 0.95])
        ]
        nested = [
            1 for a, b in combinations(crossed, 2)
            if _dominates(a, b) or _dominates(b, a)
        ]
        assert len(crossed) == 16 and len(nested) == 84

    def test_task_configs_match_their_q6_pre_registration(self):
        """Q6 pre-registered 2026-09-12 for ioi + greater_than; docstring stays OPEN.

        Was: assert all three are unconfirmed. Docstring is deliberately still False --
        its source (MIB vs the original release) has not been verified, and guessing it
        would be exactly the invention AI_RULES 2.2 forbids.
        """
        for name in ("ioi", "greater_than"):
            cfg = _load(CONFIGS / "task" / f"{name}.yaml")
            assert cfg["pi_confirmed"] is True, f"{name}: Q6 was pre-registered"
            assert cfg["n_prompts"] == 300
            assert cfg["n_templates"] and cfg["n_templates"] > 0
            assert cfg["bootstrap_resampling_unit"] == "template", (
                "prompts from one template are correlated; resampling prompts would "
                "treat correlated draws as independent and narrow every CI"
            )
            assert cfg["dataset_source"], f"{name}: Q6 requires a recorded source"

        doc = _load(CONFIGS / "task" / "docstring.yaml")
        assert doc["pi_confirmed"] is False, (
            "docstring's source is still unverified (MIB coverage unchecked); it must "
            "not be marked confirmed until someone looks"
        )
        assert doc["n_templates"] is None, "an unverified template count is not invented"

    def test_greater_than_is_the_published_year_span_task(self):
        """Guards a correction, not a preference.

        HUMAN_DECISIONS Q6 described greater-than as a day-of-month construction. The
        published task ACDC implements is a year-span completion. A circuit extracted
        on the wrong task is not comparable to the published one, which is the only
        reason the task is in this project at all.
        """
        cfg = _load(CONFIGS / "task" / "greater_than.yaml")
        assert "year" in cfg["template"], cfg["template"]
        assert "month" not in cfg["template"].lower()
        assert cfg["attribution_target"] == "year_suffix_logits"

    def test_q7_uses_two_different_corpora(self):
        """Calibration and perplexity are different datasets in the reference work."""
        cfg = _load(CONFIGS / "calibration" / "final.yaml")
        calib, ppl = cfg["calibration"], cfg["perplexity"]
        assert calib["corpus"] == "fineweb-edu"
        assert ppl["corpus"] == "wikitext-2-raw"
        assert calib["corpus"] != ppl["corpus"], (
            "calibrating on the perplexity corpus means our pruned model is not the "
            "reference work's pruned model, and C5 compares two interventions"
        )
        assert calib["n_tokens"] == 300000
        assert calib["seed"] == 7, (
            "their pruning is deterministic GIVEN the calibration cache; a different "
            "seed gives a different mask with nothing in the output to show it"
        )
        # the protocol that exists because getting it wrong moved Gemma-2 by 180x
        assert ppl["window"] == 1024 and ppl["stride"] == 512
        assert ppl["bos_per_window"] is True
        assert ppl["dtype"] == "bfloat16"

    def test_real_models_carry_their_pinned_revision(self):
        """Q5 resolved 2026-09-12. Was: assert every pin is still null.

        The pins are asserted by value, not merely as "not null". A silent re-pin is
        the failure this guards: it would make a Stage A dense reference and a Stage C
        re-extraction different models while every hash in run_meta.json still looked
        self-consistent. Changing a pin must be a deliberate edit here too.
        """
        expected = {
            "gemma2_2b": "c5ebcd40d208330abc697524c919956e692655cf",
            "llama32_1b": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
            "pythia160m": "50f5173d932e8e61f858120bcb800b97af589f46",
            "pythia410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
        }
        for name, pin in expected.items():
            got = _load(CONFIGS / "model" / f"{name}.yaml")["hf_revision"]
            assert got == pin, (
                f"{name}: hf_revision is {got!r}, expected {pin!r}. If this is a "
                "deliberate re-pin, update docs/HUMAN_DECISIONS.md Q5, "
                "configs/provisional_defaults.yaml and this test together — and treat "
                "every run made against the old pin as a different model."
            )
            assert len(got) == 40 and all(c in "0123456789abcdef" for c in got)

    def test_transcoder_sets_are_pinned_wherever_pipeline_a_is_available(self):
        """A pinned model with a floating transcoder set is still a floating circuit."""
        for name in ("gemma2_2b", "llama32_1b"):
            cfg = _load(CONFIGS / "model" / f"{name}.yaml")
            assert cfg["transcoder_set"] is not None
            rev = cfg.get("transcoder_revision")
            assert rev is not None and len(rev) == 40, (
                f"{name}: transcoder_set {cfg['transcoder_set']} has no 40-char "
                "transcoder_revision; pipeline A would extract against a moving target"
            )
        for name in ("pythia160m", "pythia410m"):
            assert _load(CONFIGS / "model" / f"{name}.yaml")["transcoder_set"] is None

    def test_distance_is_preregistered_as_l1_primary_js_ablation(self):
        """Q2 pre-registered 2026-09-12. Was: assert it is NOT pre-registered."""
        cfg = _load(CONFIGS / "distance" / "provisional_l1_js.yaml")
        assert cfg["pre_registered_for_stage_c"] is True
        assert cfg["name"] == "l1", (
            "L1 must stay PRIMARY: it is the only one of the two that can see a uniform "
            "loss of inclusion mass, which is exactly what heavy pruning causes"
        )
        assert cfg["alternative"] == "jensen_shannon"
        assert cfg["also_report"] == "normalized_l1"

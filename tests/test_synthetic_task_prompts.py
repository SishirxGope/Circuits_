# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: synthetic seeded task prompts (Q6 provisional)
# reviewed-by: PENDING

import pytest

from src.synthetic.synthetic_tasks import load_synthetic_prompts

IOI_TASK = {"name": "synthetic-ioi", "family": "indirect-object-identification-synthetic",
            "synthetic": True, "n_prompts": 24, "max_seq_len": 32, "seed": 0}
GT_TASK = {"name": "synthetic-greater-than", "family": "greater-than-synthetic",
           "synthetic": True, "n_prompts": 12, "max_seq_len": 16, "seed": 1}
DOC_TASK = {"name": "synthetic-docstring", "family": "docstring-completion-synthetic",
            "synthetic": True, "n_prompts": 6, "max_seq_len": 64, "seed": 2}


def test_deterministic_given_config():
    assert load_synthetic_prompts(IOI_TASK) == load_synthetic_prompts(IOI_TASK)


def test_prompt_count_matches_config():
    assert len(load_synthetic_prompts(IOI_TASK)) == 24
    assert len(load_synthetic_prompts(GT_TASK)) == 12
    assert len(load_synthetic_prompts(DOC_TASK)) == 6


def test_prompts_are_nonempty_strings():
    for task in (IOI_TASK, GT_TASK, DOC_TASK):
        for p in load_synthetic_prompts(task):
            assert isinstance(p, str) and p.strip()


def test_seed_changes_prompts():
    assert load_synthetic_prompts(IOI_TASK) != load_synthetic_prompts({**IOI_TASK, "seed": 1})


def test_families_differ_in_shape():
    ioi = load_synthetic_prompts(IOI_TASK)[0]
    doc = load_synthetic_prompts(DOC_TASK)[0]
    assert "went to the park" in ioi
    assert "def helper_" in doc


def test_unknown_family_rejected():
    with pytest.raises(ValueError, match="unknown synthetic family"):
        load_synthetic_prompts({"name": "x", "family": "bogus", "synthetic": True, "n_prompts": 4, "seed": 0})


def test_real_task_rejected_by_synthetic_loader():
    with pytest.raises(ValueError, match="synthetic tasks"):
        load_synthetic_prompts({"name": "ioi", "family": "indirect-object-identification",
                                "synthetic": False, "n_prompts": 4, "seed": 0})

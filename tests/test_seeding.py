# [AI-GEN] agent=OpenCode date=2026-08-07 task=Unit tests: seeded RNG discipline (AI_RULES.md 1.1)
# reviewed-by: PENDING

import numpy as np
import pytest

from src.common.seeding import create_seed_generator, derive_child_seed


def test_same_seed_same_sequence():
    g1, _ = create_seed_generator(42)
    g2, _ = create_seed_generator(42)
    a = g1.integers(0, 1_000_000, size=20)
    b = g2.integers(0, 1_000_000, size=20)
    assert (a == b).all()


def test_different_seeds_different_sequences():
    g1, _ = create_seed_generator(42)
    g2, _ = create_seed_generator(43)
    a = g1.integers(0, 1_000_000, size=20)
    b = g2.integers(0, 1_000_000, size=20)
    assert not (a == b).all()


def test_no_global_rng_dependence():
    before = np.random.get_state()
    create_seed_generator(1)
    create_seed_generator(2)
    create_seed_generator(999)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert (before[1] == after[1]).all()
    assert before[2:] == after[2:]


def test_torch_generator_seeded_when_available():
    torch = pytest.importorskip("torch")
    _, t1 = create_seed_generator(7)
    _, t2 = create_seed_generator(7)
    assert t1 is not None and t2 is not None
    a = torch.randint(0, 1_000_000, (20,), generator=t1)
    b = torch.randint(0, 1_000_000, (20,), generator=t2)
    assert (a == b).all()


def test_derive_child_seed_deterministic_and_bounded():
    assert derive_child_seed(0, "config-7") == derive_child_seed(0, "config-7")
    assert derive_child_seed(0, "config-7") != derive_child_seed(0, "config-8")
    assert derive_child_seed(0, "config-7") != derive_child_seed(1, "config-7")
    for seed in (0, 1, 2 ** 40):
        v = derive_child_seed(seed, "a", "b")
        assert 0 <= v < 2 ** 63

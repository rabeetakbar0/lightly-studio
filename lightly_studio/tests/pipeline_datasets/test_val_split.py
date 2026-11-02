from __future__ import annotations

import numpy as np
import pytest
from pytest_mock import MockerFixture

from lightly_studio.pipeline_datasets import val_split
from lightly_studio.pipeline_datasets.val_split import Unit


def _vec(x: float, y: float) -> np.ndarray:  # type: ignore[type-arg]
    return np.array([x, y], dtype=np.float32)


def _blob_units(points: list[tuple[float, float]]) -> list[Unit]:
    return [Unit(members=(i,), vector=_vec(*p)) for i, p in enumerate(points)]


@pytest.mark.parametrize(
    ("n_units", "fraction", "groups", "expected"),
    [
        (10, 0.1, None, 1),  # One hold-out: nothing to spread.
        (20, 0.1, None, 2),
        (50, 0.1, None, 5),
        (1000, 0.1, None, 22),
        (100_000, 0.1, None, 224),  # round(sqrt(100000 / 2)).
        (50, 0.1, 20, 5),  # A user value is bounded by the hold-out count.
        (50, 0.5, 3, 3),
        (2_000_000, 0.5, None, 512),
    ],
)
def test_choose_groups(n_units: int, fraction: float, groups: int | None, expected: int) -> None:
    assert (
        val_split.choose_groups(n_units=n_units, holdout_fraction=fraction, groups=groups)
        == expected
    )


@pytest.mark.parametrize(
    ("sizes", "fraction", "expected"),
    [
        ([10, 10, 10], 0.1, [1, 1, 1]),
        ([7, 7, 6], 0.25, [2, 2, 1]),
        ([3], 0.1, [1]),  # At least one hold-out.
        ([1, 1, 1, 1], 0.5, [1, 1, 0, 0]),
        ([5, 5], 0.0, [0, 0]),
        ([1], 0.5, [0]),  # A single image is never held out.
    ],
)
def test_allocate(sizes: list[int], fraction: float, expected: list[int]) -> None:
    assert val_split.allocate(sizes=sizes, fraction=fraction) == expected


def test_cluster() -> None:
    vectors = np.array(
        [(0, 0)] * 4 + [(10, 0)] * 4 + [(0, 10)] * 4, dtype=np.float32
    ) + np.random.default_rng(0).normal(scale=0.1, size=(12, 2)).astype(np.float32)
    labels = val_split.cluster(vectors=vectors, k=3, seed=0)
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:8])) == 1
    assert len(set(labels[8:])) == 1
    assert set(labels) == {0, 1, 2}


def test_cluster__k_above_distinct_vectors() -> None:
    vectors = np.ones((6, 2), dtype=np.float32)
    labels = val_split.cluster(vectors=vectors, k=3, seed=0)
    assert set(labels) == {0}


def test_cluster__minibatch_for_large_inputs(mocker: MockerFixture) -> None:
    mocker.patch.object(val_split, "_MINIBATCH_FROM", 4)
    spy = mocker.spy(val_split, "MiniBatchKMeans")
    vectors = np.random.default_rng(0).normal(size=(8, 2)).astype(np.float32)
    val_split.cluster(vectors=vectors, k=2, seed=0)
    assert spy.call_count == 1


def test_build_units__by_image() -> None:
    units = val_split.build_units(
        tags=["a", "a", ""], split_by="image", vectors=[_vec(3, 4), None, _vec(0, 0)]
    )
    assert [unit.members for unit in units] == [(0,), (1,), (2,)]
    assert units[0].vector is not None
    np.testing.assert_allclose(units[0].vector, [0.6, 0.8])
    assert units[1].vector is None
    assert units[2].vector is None  # A zero vector cannot be normalised.


def test_build_units__by_tag() -> None:
    units = val_split.build_units(
        tags=["a", "", "a", "b"],
        split_by="tag",
        vectors=[_vec(1, 0), _vec(1, 1), _vec(0, 1), None],
    )
    assert [unit.members for unit in units] == [(0, 2), (1,), (3,)]
    assert units[0].vector is not None
    np.testing.assert_allclose(units[0].vector, [0.7071, 0.7071], atol=1e-3)
    assert units[2].vector is None


def test_holdout_split__by_image_takes_from_every_group() -> None:
    points = [(1.0 + 0.01 * i, 0.0) for i in range(10)] + [(0.0, 1.0 + 0.01 * i) for i in range(10)]
    split = val_split.holdout_split(
        units=_blob_units(points), val_fraction=0.2, test_fraction=0.2, split_by="image", groups=2
    )
    val = {i for i, kind in split.assignment.items() if kind == "val"}
    test = {i for i, kind in split.assignment.items() if kind == "test"}
    assert len(val) == 4
    assert len(test) == 4
    assert val.isdisjoint(test)
    assert len(val & set(range(10))) == 2
    assert len(test & set(range(10))) == 2
    assert split.info.groups == 2
    assert split.info.strategy == "embedding"
    assert sorted((c.train, c.val, c.test) for c in split.info.group_counts) == [
        (6, 2, 2),
        (6, 2, 2),
    ]
    assert split.info.fallback_images == 0


def test_holdout_split__by_tag_keeps_tags_whole() -> None:
    # Four tags of five images, two visual groups; half of each group is held out.
    tags = ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5
    vectors = [_vec(1, 0)] * 10 + [_vec(0, 1)] * 10
    units = val_split.build_units(tags=tags, split_by="tag", vectors=vectors)
    split = val_split.holdout_split(
        units=units, val_fraction=0.5, test_fraction=0.0, split_by="tag", groups=2
    )
    val = sorted(i for i, kind in split.assignment.items() if kind == "val")
    assert len(val) == 10
    held_tags = {tags[i] for i in val}
    assert len(held_tags) == 2
    assert len(held_tags & {"a", "b"}) == 1
    assert len(held_tags & {"c", "d"}) == 1


def test_holdout_split__units_without_vectors_split_at_random() -> None:
    units = _blob_units([(1.0, 0.0)] * 6) + [Unit(members=(i,)) for i in range(6, 12)]
    split = val_split.holdout_split(
        units=units, val_fraction=0.5, test_fraction=0.0, split_by="image", groups=None
    )
    val = {i for i, kind in split.assignment.items() if kind == "val"}
    assert len(val) == 6
    assert len(val & set(range(6))) == 3
    assert len(val & set(range(6, 12))) == 3
    assert split.info.fallback_images == 6
    assert split.info.groups == 0  # Six identical vectors: one group, nothing to spread.


@pytest.mark.parametrize(
    ("units", "val_fraction", "test_fraction"),
    [
        ([Unit(members=(0, 1, 2))], 0.5, 0.0),  # One unit.
        ([Unit(members=(0,)), Unit(members=(1,))], 0.0, 0.0),
        ([Unit(members=(0,))], 0.9, 0.0),
    ],
)
def test_holdout_split__nothing_to_hold_out(
    units: list[Unit], val_fraction: float, test_fraction: float
) -> None:
    split = val_split.holdout_split(
        units=units, val_fraction=val_fraction, test_fraction=test_fraction, split_by="image"
    )
    assert split.assignment == {}


def test_holdout_split__never_everything() -> None:
    split = val_split.holdout_split(
        units=[Unit(members=(0,)), Unit(members=(1,))],
        val_fraction=0.5,
        test_fraction=0.45,
        split_by="image",
    )
    assert len(split.assignment) == 1


def test_random_split() -> None:
    units = _blob_units([(1.0, 0.0)] * 10)
    split = val_split.random_split(
        units=units, val_fraction=0.2, test_fraction=0.1, split_by="image"
    )
    assert sorted(split.assignment.values()) == ["test", "val", "val"]
    assert split.info.strategy == "random"
    assert split.info.groups == 0
    assert split.info.fallback_images == 10

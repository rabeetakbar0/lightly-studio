"""Hold-out splits that are spread over the whole dataset.

The images (or whole tags) are grouped by visual similarity with k-means on their
embeddings, and the validation and test shares are taken from every group. Units without an
embedding go to one random pool that is split by the same fractions. Pure numpy and
scikit-learn; the database is read by the caller.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]

STRATEGY_RANDOM = "random"
STRATEGY_EMBEDDING = "embedding"
HOLDOUT_VAL = "val"
HOLDOUT_TEST = "test"
SPLIT_BY_TAG = "tag"
SPLIT_BY_IMAGE = "image"

MAX_GROUPS = 512
_MINIBATCH_FROM = 10_000
_MIN_IMAGES = 2
_POOL = -1


@dataclass(frozen=True, eq=False)
class Unit:
    """What the split moves as a whole: one image, or all images of a tag.

    Attributes:
        members: Indices of the images in the export list.
        vector: L2-normalised embedding; None puts the unit into the random pool.
    """

    members: tuple[int, ...]
    vector: NDArray[np.float32] | None = None

    @property
    def size(self) -> int:
        """Number of images."""
        return len(self.members)


@dataclass
class GroupCounts:
    """How one similarity group was split."""

    train: int = 0
    val: int = 0
    test: int = 0


@dataclass
class SplitInfo:
    """How the hold-outs were chosen; stored in the dataset manifest.

    Attributes:
        strategy: "embedding" or "random".
        split_by: "tag" or "image".
        val_fraction: Share held out as validation.
        test_fraction: Share held out as test.
        groups: Similarity groups used; 0 when nothing was clustered.
        groups_requested: The user's group count, None for automatic.
        group_counts: Train, val and test images per group.
        fallback_images: Images split at random because they had no embedding.
    """

    strategy: str
    split_by: str
    val_fraction: float
    test_fraction: float
    groups: int = 0
    groups_requested: int | None = None
    group_counts: list[GroupCounts] = field(default_factory=list)
    fallback_images: int = 0


@dataclass(frozen=True)
class Split:
    """The chosen hold-outs: image index to "val" or "test"; absent images train."""

    assignment: dict[int, str]
    info: SplitInfo


def build_units(
    tags: Sequence[str],
    split_by: str,
    vectors: Sequence[NDArray[np.float32] | None] | None,
) -> list[Unit]:
    """Turns the export list into units.

    Args:
        tags: One tag per image, "" for none.
        split_by: "image" makes one unit per image, "tag" one per tag with untagged images
            as single-image units.
        vectors: One embedding per image or None; None for all means no embeddings.

    Returns:
        The units in first-appearance order.
    """
    normalised = [
        None if vectors is None else _normalise(vectors[index]) for index in range(len(tags))
    ]
    if split_by == SPLIT_BY_IMAGE:
        return [Unit(members=(index,), vector=normalised[index]) for index in range(len(tags))]
    members: dict[str, list[int]] = defaultdict(list)
    for index, tag in enumerate(tags):
        members[tag or f"\0{index}"].append(index)
    units = []
    for indices in members.values():
        present = [v for v in (normalised[i] for i in indices) if v is not None]
        vector = _normalise(np.mean(np.stack(present), axis=0)) if present else None
        units.append(Unit(members=tuple(indices), vector=vector))
    return units


def holdout_split(  # noqa: PLR0913
    units: Sequence[Unit],
    val_fraction: float,
    test_fraction: float,
    split_by: str,
    groups: int | None = None,
    seed: int = 0,
) -> Split:
    """Chooses validation and test units from every similarity group.

    Units with a vector are clustered; the others form one pool. Each group gives its share
    of images to validation, then to test, whole units at a time. The result never holds out
    every image.

    Args:
        units: The units to split.
        val_fraction: Share of images for validation, 0 for none.
        test_fraction: Share of images for test, 0 for none.
        split_by: Recorded in the result.
        groups: Number of similarity groups, None for automatic.
        seed: Seed of the random choices inside the groups.
    """
    return _split(
        units=units,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_by=split_by,
        groups=groups,
        seed=seed,
        strategy=STRATEGY_EMBEDDING,
    )


def random_split(
    units: Sequence[Unit],
    val_fraction: float,
    test_fraction: float,
    split_by: str,
    seed: int = 0,
) -> Split:
    """Chooses validation and test units at random, ignoring embeddings."""
    return _split(
        units=[Unit(members=unit.members) for unit in units],
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_by=split_by,
        groups=None,
        seed=seed,
        strategy=STRATEGY_RANDOM,
    )


def choose_groups(n_units: int, holdout_fraction: float, groups: int | None) -> int:
    """How many similarity groups to use.

    Automatic is sqrt(n/2), bounded by the number of held-out units so every group can give
    at least one, and by MAX_GROUPS. A user value is bounded the same way. 1 means no
    clustering: fewer than two units would be held out.
    """
    n_holdout = math.floor(n_units * holdout_fraction)
    if n_holdout < _MIN_IMAGES:
        return 1
    wanted = groups if groups is not None else round(math.sqrt(n_units / 2))
    return max(2, min(wanted, n_holdout, MAX_GROUPS))


def cluster(vectors: NDArray[np.float32], k: int, seed: int) -> NDArray[np.int64]:
    """K-means labels for the rows of vectors, compacted to 0..groups-1.

    Duplicate vectors can leave clusters empty; those labels are removed.
    """
    k = min(k, len(vectors))
    if k <= 1:
        return np.zeros(len(vectors), dtype=np.int64)
    estimator = (
        KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=seed)
        if len(vectors) < _MINIBATCH_FROM
        else MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=seed)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = estimator.fit_predict(vectors)
    _, compact = np.unique(labels, return_inverse=True)
    return np.asarray(compact, dtype=np.int64)


def allocate(sizes: Sequence[int], fraction: float) -> list[int]:
    """Images to hold out per group so that the total is round(fraction * all images).

    Largest-remainder rounding: every group gets the floor of its share, the rest goes to the
    groups with the largest fractional parts. At least one image is held out when the
    fraction is positive and there are at least two images.
    """
    total = sum(sizes)
    if fraction <= 0 or total < _MIN_IMAGES:
        return [0] * len(sizes)
    target = min(max(1, round(fraction * total)), total - 1)
    shares = [fraction * size for size in sizes]
    quotas = [math.floor(share) for share in shares]
    remainders = sorted(
        range(len(sizes)), key=lambda i: (shares[i] - quotas[i], sizes[i]), reverse=True
    )
    for index in remainders:
        if sum(quotas) >= target:
            break
        if quotas[index] < sizes[index]:
            quotas[index] += 1
    while sum(quotas) > target:
        index = max(range(len(sizes)), key=lambda i: quotas[i])
        quotas[index] -= 1
    return quotas


def _split(  # noqa: PLR0913
    units: Sequence[Unit],
    val_fraction: float,
    test_fraction: float,
    split_by: str,
    groups: int | None,
    seed: int,
    strategy: str,
) -> Split:
    info = SplitInfo(
        strategy=strategy,
        split_by=split_by,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        groups_requested=groups,
    )
    n_images = sum(unit.size for unit in units)
    if n_images < _MIN_IMAGES or val_fraction + test_fraction <= 0:
        return Split(assignment={}, info=info)

    labels = _labels(
        units=units, holdout_fraction=val_fraction + test_fraction, groups=groups, seed=seed
    )
    by_label: dict[int, list[Unit]] = defaultdict(list)
    for unit, label in zip(units, labels):
        by_label[label].append(unit)
    order = sorted(by_label)
    sizes = [sum(unit.size for unit in by_label[label]) for label in order]
    val_quotas = allocate(sizes=sizes, fraction=val_fraction)
    test_quotas = allocate(sizes=sizes, fraction=test_fraction)

    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    for label, val_quota, test_quota in zip(order, val_quotas, test_quotas):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        val_units = _fill(candidates, quota=val_quota)
        taken = {id(unit) for unit in val_units}
        rest = [unit for unit in candidates if id(unit) not in taken]
        test_units = _fill(rest, quota=test_quota)
        counts = GroupCounts(train=sizes[order.index(label)])
        for chosen, kind in ((val_units, HOLDOUT_VAL), (test_units, HOLDOUT_TEST)):
            for unit in chosen:
                assignment.update(dict.fromkeys(unit.members, kind))
                counts.train -= unit.size
                setattr(counts, kind, getattr(counts, kind) + unit.size)
        if label != _POOL:
            info.group_counts.append(counts)
        else:
            info.fallback_images = sizes[order.index(label)]
    _keep_some_training(units=units, assignment=assignment, info=info)
    info.groups = len(info.group_counts) if len(info.group_counts) > 1 else 0
    return Split(assignment=assignment, info=info)


def _labels(
    units: Sequence[Unit], holdout_fraction: float, groups: int | None, seed: int
) -> list[int]:
    """One group label per unit; _POOL for units without a vector."""
    clustered = [index for index, unit in enumerate(units) if unit.vector is not None]
    labels = [_POOL] * len(units)
    if not clustered:
        return labels
    k = choose_groups(n_units=len(clustered), holdout_fraction=holdout_fraction, groups=groups)
    vectors = np.stack(
        [vector for vector in (units[i].vector for i in clustered) if vector is not None]
    )
    cluster_labels = cluster(vectors=vectors, k=k, seed=seed)
    for index, label in zip(clustered, cluster_labels):
        labels[index] = int(label)
    return labels


def _fill(candidates: list[Unit], quota: int) -> list[Unit]:
    """Takes whole units in the given order until the quota is reached.

    A unit that would overshoot the quota by more than the current shortfall is skipped, so
    a smaller unit later in the order can fit instead. Exact for single-image units.
    """
    chosen: list[Unit] = []
    count = 0
    for unit in candidates:
        if count >= quota:
            break
        if count + unit.size - quota > quota - count:
            continue
        chosen.append(unit)
        count += unit.size
    return chosen


def _keep_some_training(units: Sequence[Unit], assignment: dict[int, str], info: SplitInfo) -> None:
    """Returns the smallest held-out unit to training when nothing would be left."""
    n_images = sum(unit.size for unit in units)
    if len(assignment) < n_images:
        return
    held = [unit for unit in units if unit.members and unit.members[0] in assignment]
    smallest = min(held, key=lambda unit: unit.size)
    kind = assignment[smallest.members[0]]
    for index in smallest.members:
        del assignment[index]
    for counts in info.group_counts:
        if getattr(counts, kind) >= smallest.size:
            setattr(counts, kind, getattr(counts, kind) - smallest.size)
            counts.train += smallest.size
            break


def _normalise(vector: NDArray[np.float32] | None) -> NDArray[np.float32] | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return np.asarray(array / norm, dtype=np.float32)

"""Writes, merges and lists training datasets in the COCO folder layout.

The layout is the one YOLOX and most detectors read directly::

    <datasets dir>/<name>/
        annotations/instances_train2017.json   file_name is the bare file name
        annotations/instances_val2017.json
        train2017/<image files>                hard links to the source, or copies
        val2017/<image files>
        dataset.json                           manifest for the GUI
        README.md

Category ids are 1-based and contiguous. Every image entry carries the tag it had in
LightlyStudio as `recording`, and the dataset it came from as `source_dataset`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lightly_studio.pipeline_datasets import val_split
from lightly_studio.pipeline_datasets.val_split import GroupCounts, Split, SplitInfo

TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
TEST_SPLIT = "test2017"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
HOLDOUT_SPLIT = {val_split.HOLDOUT_VAL: VAL_SPLIT, val_split.HOLDOUT_TEST: TEST_SPLIT}
KIND_TRAIN = "train"
KIND_TEST = "test"
MANIFEST_FILE = "dataset.json"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MIN_SOURCES = 2
_LICENSES = [{"id": 1, "name": "unknown", "url": ""}]


@dataclass(frozen=True)
class ImageRecord:
    """One image to export.

    Attributes:
        path: Where the image file is now.
        width: Width in pixels.
        height: Height in pixels.
        tag: The tag the image carries in LightlyStudio, used to split by recording.
    """

    path: Path
    width: int
    height: int
    tag: str = ""
    sample_id: str = ""


@dataclass(frozen=True)
class BoxRecord:
    """One object-detection box, in pixels of its image.

    Attributes:
        image_index: Index into the image list passed along with the boxes.
        class_name: The annotation class.
        x, y, width, height: Top-left corner and size.
    """

    image_index: int
    class_name: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class SplitSummary:
    """Image and box counts of one split."""

    images: int = 0
    boxes: int = 0


@dataclass(frozen=True)
class PoolImage:
    """One image of an existing dataset, ready to be copied into a new one.

    Attributes:
        source_dir: The dataset folder it comes from.
        split: The split folder inside that dataset.
        image: Its COCO image entry.
        annotations: Its COCO annotations.
        category_names: The source's category id to name.
    """

    source_dir: Path
    split: str
    image: dict[str, Any]
    annotations: tuple[dict[str, Any], ...]
    category_names: dict[int, str]

    @property
    def file_name(self) -> str:
        """The bare file name."""
        return str(self.image["file_name"])

    @property
    def sample_id(self) -> str:
        """The LightlyStudio sample the image was exported from, "" when unknown."""
        return str(self.image.get("sample_id", ""))


@dataclass
class ImagePool:
    """All images of some datasets, every file name once.

    Attributes:
        images: The images, sources in the given order, first occurrence of a name wins.
        class_names: Union of the sources' classes in first-appearance order.
        sources: The dataset names.
    """

    images: list[PoolImage] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class DatasetSummary:
    """What the GUI shows about a dataset folder.

    Attributes:
        name: Folder name.
        path: Absolute folder path.
        created_at: ISO timestamp from the manifest.
        classes: Class names in category id order.
        splits: Counts per split name.
        sources: Datasets this one was merged from, empty for an export.
        annotation_source: The LightlyStudio annotation source it was exported from.
        size_bytes: Size of the image files on disk, counting hard links once per dataset.
        kind: "train" for a training dataset, "test" for a test-only dataset.
        split: How the hold-outs were chosen, None for merged datasets.
    """

    name: str
    path: str
    created_at: str
    classes: list[str] = field(default_factory=list)
    splits: dict[str, SplitSummary] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    annotation_source: str = ""
    size_bytes: int = 0
    kind: str = KIND_TRAIN
    split: SplitInfo | None = None

    @property
    def images(self) -> int:
        """Images over all splits."""
        return sum(split.images for split in self.splits.values())

    @property
    def boxes(self) -> int:
        """Boxes over all splits."""
        return sum(split.boxes for split in self.splits.values())


def validate_name(name: str) -> str:
    """Returns the name when it is safe as a folder name, otherwise raises ValueError."""
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "Dataset name must be 1-64 characters of letters, digits, '.', '_' or '-', "
            "and start with a letter or digit."
        )
    return name


def write_dataset(  # noqa: PLR0913
    datasets_dir: Path,
    name: str,
    images: list[ImageRecord],
    boxes: list[BoxRecord],
    class_names: list[str],
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    split_by: str = "tag",
    seed: int = 0,
    annotation_source: str = "",
    copy_files: bool = False,
    split: Split | None = None,
) -> DatasetSummary:
    """Writes a new dataset folder from images and their boxes.

    Args:
        datasets_dir: Parent folder of all datasets.
        name: Folder name; must not exist yet.
        images: The images, in any order. File names must be unique.
        boxes: Boxes referring to the images by index.
        class_names: Category order. Classes used by boxes but missing here are appended.
        val_fraction: Share of images held out as validation, 0 for none.
        test_fraction: Share of images held out as test, 0 for none.
        split_by: "tag" holds out whole tags (recordings) so frames of one video never
            straddle the split, "image" splits single images.
        seed: Random seed for the split.
        annotation_source: Recorded in the manifest.
        copy_files: Copy image files instead of hard-linking them.
        split: A precomputed split over the image indices; None picks at random.

    Returns:
        The summary of the written dataset.

    Raises:
        ValueError: On a bad name, an existing folder, duplicate file names, or no images.
    """
    validate_name(name=name)
    out_dir = datasets_dir / name
    if out_dir.exists():
        raise ValueError(f"Dataset '{name}' already exists.")
    if not images:
        raise ValueError("There are no images to export.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to below 1.")
    names = [image.path.name for image in images]
    if len(set(names)) != len(names):
        raise ValueError("Image file names must be unique inside one dataset.")

    categories = _categories(class_names=class_names, boxes=boxes)
    if split is None:
        split = val_split.random_split(
            units=val_split.build_units(
                tags=[image.tag for image in images], split_by=split_by, vectors=None
            ),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_by=split_by,
            seed=seed,
        )
    boxes_by_image: dict[int, list[BoxRecord]] = defaultdict(list)
    for box in boxes:
        boxes_by_image[box.image_index].append(box)

    out_dir.mkdir(parents=True)
    summary = DatasetSummary(
        name=name,
        path=str(out_dir.resolve()),
        created_at=_now(),
        classes=[category["name"] for category in categories],
        annotation_source=annotation_source,
        split=split.info,
    )
    holdout_of = {HOLDOUT_SPLIT[kind]: kind for kind in HOLDOUT_SPLIT}
    for split_name in SPLITS:
        indices = [
            i
            for i in range(len(images))
            if split.assignment.get(i, TRAIN_SPLIT) == holdout_of.get(split_name, TRAIN_SPLIT)
        ]
        if split_name != TRAIN_SPLIT and not indices:
            continue
        summary.splits[split_name] = _write_export_split(
            out_dir=out_dir,
            split=split_name,
            name=name,
            images=[images[i] for i in indices],
            boxes_by_image=[boxes_by_image.get(i, []) for i in indices],
            categories=categories,
            copy_files=copy_files,
        )
    summary.size_bytes = _size_bytes(out_dir)
    _write_manifest(out_dir=out_dir, summary=summary)
    _write_readme(out_dir=out_dir, summary=summary)
    return summary


def merge_datasets(
    datasets_dir: Path, name: str, sources: list[str], copy_files: bool = False
) -> DatasetSummary:
    """Merges existing datasets into a new one, keeping each source's own split.

    Categories are matched by name; the first source's order comes first and new classes
    are appended. Image file names must not collide across the sources.

    Raises:
        ValueError: On a bad name, fewer than two sources, a missing source, an existing
            target, or colliding file names.
    """
    validate_name(name=name)
    if len(sources) < _MIN_SOURCES or len(set(sources)) != len(sources):
        raise ValueError("Pick at least two different datasets to merge.")
    out_dir = datasets_dir / name
    if out_dir.exists():
        raise ValueError(f"Dataset '{name}' already exists.")
    class_names, per_split = _load_sources(datasets_dir=datasets_dir, sources=sources)
    categories = [
        {"id": i, "name": class_name, "supercategory": "object"}
        for i, class_name in enumerate(class_names, start=1)
    ]

    out_dir.mkdir(parents=True)
    summary = DatasetSummary(
        name=name,
        path=str(out_dir.resolve()),
        created_at=_now(),
        classes=class_names,
        sources=list(sources),
    )
    for split in SPLITS:
        if not per_split.get(split):
            continue
        try:
            summary.splits[split] = _merge_split(
                out_dir=out_dir,
                split=split,
                description=f"{name}: merged from {', '.join(sources)}",
                parts=per_split[split],
                categories=categories,
                copy_files=copy_files,
            )
        except ValueError:
            shutil.rmtree(out_dir)
            raise
    if set(summary.splits) == {TEST_SPLIT}:
        summary.kind = KIND_TEST
    summary.size_bytes = _size_bytes(out_dir)
    _write_manifest(out_dir=out_dir, summary=summary)
    _write_readme(out_dir=out_dir, summary=summary)
    return summary


def load_pool(datasets_dir: Path, sources: list[str]) -> ImagePool:
    """Reads every split of the given datasets into one pool of images.

    Raises:
        ValueError: If no source is given or one does not exist.
    """
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Pick at least one dataset, each once.")
    class_names, per_split = _load_sources(datasets_dir=datasets_dir, sources=sources)
    pool = ImagePool(class_names=class_names, sources=list(sources))
    seen: set[str] = set()
    for split in SPLITS:
        for source_dir, coco in per_split.get(split, []):
            by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for annotation in coco["annotations"]:
                by_image[annotation["image_id"]].append(annotation)
            category_names = {c["id"]: c["name"] for c in coco["categories"]}
            for image in coco["images"]:
                if image["file_name"] in seen:
                    continue
                seen.add(image["file_name"])
                pool.images.append(
                    PoolImage(
                        source_dir=source_dir,
                        split=split,
                        image=image,
                        annotations=tuple(by_image.get(image["id"], [])),
                        category_names=category_names,
                    )
                )
    return pool


def write_test_dataset(  # noqa: PLR0913
    datasets_dir: Path,
    name: str,
    pool: ImagePool,
    test_indices: list[int],
    split: SplitInfo | None = None,
    copy_files: bool = False,
) -> DatasetSummary:
    """Writes a test-only dataset from chosen images of the pool.

    Raises:
        ValueError: On a bad name, an existing folder, or no chosen images.
    """
    validate_name(name=name)
    out_dir = datasets_dir / name
    if out_dir.exists():
        raise ValueError(f"Dataset '{name}' already exists.")
    if not test_indices:
        raise ValueError("No images were chosen for the test dataset.")
    categories = [
        {"id": i, "name": class_name, "supercategory": "object"}
        for i, class_name in enumerate(pool.class_names, start=1)
    ]
    category_id = {category["name"]: category["id"] for category in categories}
    coco: dict[str, Any] = {
        "info": _info(description=f"{name}: test dataset drawn from {', '.join(pool.sources)}"),
        "licenses": _LICENSES,
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    out_dir.mkdir(parents=True)
    (out_dir / TEST_SPLIT).mkdir()
    annotation_id = 1
    for image_id, index in enumerate(sorted(test_indices), start=1):
        entry = pool.images[index]
        _place(
            src=entry.source_dir / entry.split / entry.file_name,
            dst=out_dir / TEST_SPLIT / entry.file_name,
            copy_files=copy_files,
        )
        coco["images"].append(
            {
                **entry.image,
                "id": image_id,
                "source_dataset": entry.image.get("source_dataset") or entry.source_dir.name,
            }
        )
        for annotation in entry.annotations:
            coco["annotations"].append(
                {
                    **annotation,
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id[entry.category_names[annotation["category_id"]]],
                }
            )
            annotation_id += 1
    _write_json(out_dir / "annotations" / f"instances_{TEST_SPLIT}.json", coco)
    summary = DatasetSummary(
        name=name,
        path=str(out_dir.resolve()),
        created_at=_now(),
        classes=list(pool.class_names),
        splits={
            TEST_SPLIT: SplitSummary(images=len(coco["images"]), boxes=len(coco["annotations"]))
        },
        sources=list(pool.sources),
        kind=KIND_TEST,
        split=split,
    )
    summary.size_bytes = _size_bytes(out_dir)
    _write_manifest(out_dir=out_dir, summary=summary)
    _write_readme(out_dir=out_dir, summary=summary)
    return summary


def list_datasets(datasets_dir: Path) -> list[DatasetSummary]:
    """Returns the datasets in datasets_dir, newest first.

    Folders without a manifest are skipped.
    """
    if not datasets_dir.is_dir():
        return []
    summaries = []
    for folder in datasets_dir.iterdir():
        summary = read_summary(dataset_dir=folder)
        if summary is not None:
            summaries.append(summary)
    return sorted(summaries, key=lambda s: s.created_at, reverse=True)


def read_summary(dataset_dir: Path) -> DatasetSummary | None:
    """Reads the manifest of one dataset folder; None when there is none."""
    manifest = dataset_dir / MANIFEST_FILE
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return DatasetSummary(
        name=data["name"],
        path=str(dataset_dir.resolve()),
        created_at=data["created_at"],
        classes=list(data.get("classes", [])),
        splits={split: SplitSummary(**counts) for split, counts in data.get("splits", {}).items()},
        sources=list(data.get("sources", [])),
        annotation_source=data.get("annotation_source", ""),
        size_bytes=int(data.get("size_bytes", 0)),
        kind=str(data.get("kind", KIND_TRAIN)),
        split=_split_info(data.get("split")),
    )


def delete_dataset(datasets_dir: Path, name: str) -> None:
    """Removes a dataset folder.

    Raises:
        ValueError: If the folder is not a dataset of datasets_dir.
    """
    validate_name(name=name)
    dataset_dir = datasets_dir / name
    if read_summary(dataset_dir=dataset_dir) is None:
        raise ValueError(f"Dataset '{name}' does not exist.")
    shutil.rmtree(dataset_dir)


def _load_sources(
    datasets_dir: Path, sources: list[str]
) -> tuple[list[str], dict[str, list[tuple[Path, dict[str, Any]]]]]:
    """Reads the sources' instance files; returns the class order and the parts per split.

    Raises:
        ValueError: If a source does not exist.
    """
    class_names: list[str] = []
    per_split: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for source in sources:
        source_dir = datasets_dir / source
        if not (source_dir / "annotations").is_dir():
            raise ValueError(f"Dataset '{source}' does not exist.")
        for split in SPLITS:
            json_path = source_dir / "annotations" / f"instances_{split}.json"
            if not json_path.is_file():
                continue
            coco = json.loads(json_path.read_text(encoding="utf-8"))
            for category in coco["categories"]:
                if category["name"] not in class_names:
                    class_names.append(category["name"])
            per_split[split].append((source_dir, coco))
    return class_names, per_split


def _write_export_split(  # noqa: PLR0913
    out_dir: Path,
    split: str,
    name: str,
    images: list[ImageRecord],
    boxes_by_image: list[list[BoxRecord]],
    categories: list[dict[str, Any]],
    copy_files: bool,
) -> SplitSummary:
    """Writes one split's image folder and instances JSON; returns its counts."""
    category_id = {category["name"]: category["id"] for category in categories}
    coco: dict[str, Any] = {
        "info": _info(description=f"{name}: exported from LightlyStudio"),
        "licenses": _LICENSES,
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    (out_dir / split).mkdir()
    annotation_id = 1
    for image_id, (image, boxes) in enumerate(zip(images, boxes_by_image), start=1):
        _place(src=image.path, dst=out_dir / split / image.path.name, copy_files=copy_files)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image.path.name,
                "width": image.width,
                "height": image.height,
                "license": 1,
                "recording": image.tag,
                "source_dataset": name,
                "sample_id": image.sample_id,
            }
        )
        for box in boxes:
            clipped = _clip(box=box, width=image.width, height=image.height)
            if clipped is None:
                continue
            x, y, w, h = clipped
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id[box.class_name],
                    "bbox": [x, y, w, h],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    _write_json(out_dir / "annotations" / f"instances_{split}.json", coco)
    return SplitSummary(images=len(coco["images"]), boxes=len(coco["annotations"]))


def _merge_split(  # noqa: PLR0913
    out_dir: Path,
    split: str,
    description: str,
    parts: list[tuple[Path, dict[str, Any]]],
    categories: list[dict[str, Any]],
    copy_files: bool,
) -> SplitSummary:
    """Concatenates the sources' JSONs of one split with fresh ids; returns its counts.

    Raises:
        ValueError: If a file name occurs in more than one source.
    """
    category_id = {category["name"]: category["id"] for category in categories}
    merged: dict[str, Any] = {
        "info": _info(description=description),
        "licenses": _LICENSES,
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    (out_dir / split).mkdir()
    seen_names: set[str] = set()
    image_id = annotation_id = 1
    for source_dir, coco in parts:
        source_category = {c["id"]: c["name"] for c in coco["categories"]}
        new_image_id: dict[int, int] = {}
        for image in coco["images"]:
            if image["file_name"] in seen_names:
                raise ValueError(
                    f"File name '{image['file_name']}' exists in more than one source."
                )
            seen_names.add(image["file_name"])
            _place(
                src=source_dir / split / image["file_name"],
                dst=out_dir / split / image["file_name"],
                copy_files=copy_files,
            )
            new_image_id[image["id"]] = image_id
            merged["images"].append(
                {
                    **image,
                    "id": image_id,
                    "source_dataset": image.get("source_dataset") or source_dir.name,
                }
            )
            image_id += 1
        for annotation in coco["annotations"]:
            merged["annotations"].append(
                {
                    **annotation,
                    "id": annotation_id,
                    "image_id": new_image_id[annotation["image_id"]],
                    "category_id": category_id[source_category[annotation["category_id"]]],
                }
            )
            annotation_id += 1
    _write_json(out_dir / "annotations" / f"instances_{split}.json", merged)
    return SplitSummary(images=len(merged["images"]), boxes=len(merged["annotations"]))


def _categories(class_names: list[str], boxes: list[BoxRecord]) -> list[dict[str, Any]]:
    ordered = list(class_names)
    for box in boxes:
        if box.class_name not in ordered:
            ordered.append(box.class_name)
    return [
        {"id": i, "name": class_name, "supercategory": "object"}
        for i, class_name in enumerate(ordered, start=1)
    ]


def _clip(box: BoxRecord, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1 = min(max(box.x, 0.0), width)
    y1 = min(max(box.y, 0.0), height)
    x2 = min(max(box.x + box.width, 0.0), width)
    y2 = min(max(box.y + box.height, 0.0), height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)


def _place(src: Path, dst: Path, copy_files: bool) -> None:
    """Hard-links src to dst, or copies when asked or when linking is not possible."""
    if not copy_files:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # Different file system, or links not supported: copy instead.
    shutil.copy2(src, dst)


def _size_bytes(out_dir: Path) -> int:
    return sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())


def _info(description: str) -> dict[str, Any]:
    today = dt.datetime.now(dt.timezone.utc).date()
    return {
        "description": description,
        "version": "1.0",
        "year": today.year,
        "contributor": "LightlyStudio",
        "date_created": today.isoformat(),
    }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_manifest(out_dir: Path, summary: DatasetSummary) -> None:
    data = asdict(summary)
    data.pop("path")
    _write_json(out_dir / MANIFEST_FILE, data)


def _write_readme(out_dir: Path, summary: DatasetSummary) -> None:
    rows = "\n".join(
        f"| {split} | {counts.images} | {counts.boxes} |"
        for split, counts in summary.splits.items()
    )
    classes = ", ".join(f"{i}={name}" for i, name in enumerate(summary.classes, start=1))
    if summary.sources and summary.kind == KIND_TEST:
        origin = f"Test dataset drawn from: {', '.join(summary.sources)}."
    elif summary.sources:
        origin = f"Merged from: {', '.join(summary.sources)}. Each source keeps its own split."
    else:
        origin = f"Exported from CrossingVision, annotation source '{summary.annotation_source}'."
    exp_lines = [
        f"    self.num_classes = {len(summary.classes)}",
        f'    self.data_dir = "{summary.path}"',
        f'    self.train_ann = "instances_{TRAIN_SPLIT}.json"',
        f'    self.val_ann = "instances_{VAL_SPLIT}.json"',
    ]
    if TEST_SPLIT in summary.splits:
        exp_lines.append(f'    self.test_ann = "instances_{TEST_SPLIT}.json"')
    out_dir.joinpath("README.md").write_text(
        f"# {summary.name}\n\n{origin}\nCreated {summary.created_at}. Images are hard links "
        "where possible; the sources are unchanged.\n\n| split | images | boxes |\n|---|---|---|\n"
        f"{rows}\n\nClasses: {classes}\n{_split_section(summary.split)}\n"
        "## YOLOX exp settings\n\n" + "\n".join(exp_lines) + "\n",
        encoding="utf-8",
    )


def _split_section(info: SplitInfo | None) -> str:
    if info is None:
        return ""
    lines = [
        "\n## Hold-out split\n",
        f"Strategy: {info.strategy}, split by {info.split_by}, validation fraction "
        f"{info.val_fraction}, test fraction {info.test_fraction}.",
    ]
    if info.groups:
        lines.append(
            f"The hold-outs were taken from each of {info.groups} similarity groups"
            + (f" (requested {info.groups_requested})" if info.groups_requested else "")
            + ".\n\n| group | train | val | test |\n|---|---|---|---|"
        )
        lines.extend(
            f"| {i} | {counts.train} | {counts.val} | {counts.test} |"
            for i, counts in enumerate(info.group_counts, start=1)
        )
    if info.fallback_images:
        lines.append(f"\n{info.fallback_images} images without an embedding were split at random.")
    return "\n".join(lines) + "\n"


def _split_info(data: dict[str, Any] | None) -> SplitInfo | None:
    if not data:
        return None
    return SplitInfo(
        strategy=str(data.get("strategy", "")),
        split_by=str(data.get("split_by", "")),
        val_fraction=float(data.get("val_fraction", 0.0)),
        test_fraction=float(data.get("test_fraction", 0.0)),
        groups=int(data.get("groups", 0)),
        groups_requested=data.get("groups_requested"),
        group_counts=[GroupCounts(**counts) for counts in data.get("group_counts", [])],
        fallback_images=int(data.get("fallback_images", 0)),
    )

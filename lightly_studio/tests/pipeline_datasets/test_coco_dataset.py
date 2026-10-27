import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lightly_studio.pipeline_datasets import coco_dataset
from lightly_studio.pipeline_datasets.coco_dataset import BoxRecord, ImageRecord


def _images(tmp_path: Path, count: int, tag_of: Callable[[int], str]) -> list[ImageRecord]:
    folder = tmp_path / "src"
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(count):
        path = folder / f"frame_{i:03d}.jpg"
        path.write_bytes(b"jpg" + bytes([i]))
        images.append(ImageRecord(path=path, width=640, height=480, tag=tag_of(i)))
    return images


def test_write_dataset(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    images = _images(tmp_path, 4, lambda i: "clipA" if i < 2 else "clipB")
    boxes = [
        BoxRecord(image_index=0, class_name="LedLight", x=10, y=10, width=20, height=20),
        BoxRecord(
            image_index=0, class_name="GateArm", x=600, y=400, width=100, height=100
        ),  # Clipped.
        BoxRecord(image_index=3, class_name="GateArm", x=-5, y=0, width=3, height=50),  # Too thin.
    ]
    summary = coco_dataset.write_dataset(
        datasets_dir=datasets_dir,
        name="data1",
        images=images,
        boxes=boxes,
        class_names=["GateArm", "LedLight"],
        val_fraction=0.0,
        annotation_source="annotation",
    )
    assert summary.classes == ["GateArm", "LedLight"]
    assert summary.splits["train2017"].images == 4
    assert summary.splits["train2017"].boxes == 2
    assert "val2017" not in summary.splits
    out = datasets_dir / "data1"
    coco = json.loads((out / "annotations" / "instances_train2017.json").read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "GateArm", "supercategory": "object"},
        {"id": 2, "name": "LedLight", "supercategory": "object"},
    ]
    assert coco["images"][0]["file_name"] == "frame_000.jpg"
    assert coco["images"][0]["recording"] == "clipA"
    assert coco["annotations"][1]["bbox"] == [600.0, 400.0, 40.0, 80.0]
    assert sorted(p.name for p in (out / "train2017").iterdir()) == [
        f"frame_{i:03d}.jpg" for i in range(4)
    ]
    # Hard link: same inode as the source.
    assert (out / "train2017" / "frame_000.jpg").stat().st_ino == images[0].path.stat().st_ino
    assert (out / "README.md").read_text().startswith("# data1")
    listed = coco_dataset.list_datasets(datasets_dir=datasets_dir)
    assert [d.name for d in listed] == ["data1"]
    assert listed[0].images == 4
    assert listed[0].boxes == 2
    assert listed[0].annotation_source == "annotation"


def test_write_dataset__val_split_by_tag_keeps_tags_whole(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    images = _images(tmp_path, 10, lambda i: f"clip{i // 5}")
    summary = coco_dataset.write_dataset(
        datasets_dir=datasets_dir,
        name="d",
        images=images,
        boxes=[],
        class_names=["GateArm"],
        val_fraction=0.5,
        split_by="tag",
    )
    assert summary.splits["val2017"].images == 5
    val = json.loads((datasets_dir / "d" / "annotations" / "instances_val2017.json").read_text())
    assert len({image["recording"] for image in val["images"]}) == 1


def test_write_dataset__val_split_by_image(tmp_path: Path) -> None:
    images = _images(tmp_path, 10, lambda _: "same")
    summary = coco_dataset.write_dataset(
        datasets_dir=tmp_path / "datasets",
        name="d",
        images=images,
        boxes=[],
        class_names=[],
        val_fraction=0.2,
        split_by="image",
    )
    assert summary.splits["val2017"].images == 2
    assert summary.splits["train2017"].images == 8


@pytest.mark.parametrize("name", ["", "bad/name", "-leading", "x" * 65])
def test_write_dataset__bad_name(tmp_path: Path, name: str) -> None:
    images = _images(tmp_path, 1, lambda _: "")
    with pytest.raises(ValueError, match=r"Dataset name"):
        coco_dataset.write_dataset(
            datasets_dir=tmp_path / "datasets", name=name, images=images, boxes=[], class_names=[]
        )


def test_write_dataset__existing(tmp_path: Path) -> None:
    images = _images(tmp_path, 1, lambda _: "")
    coco_dataset.write_dataset(
        datasets_dir=tmp_path / "datasets", name="d", images=images, boxes=[], class_names=[]
    )
    with pytest.raises(ValueError, match=r"already exists"):
        coco_dataset.write_dataset(
            datasets_dir=tmp_path / "datasets", name="d", images=images, boxes=[], class_names=[]
        )


def test_merge_datasets(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    a_images = _images(tmp_path / "a", 3, lambda _: "a")
    b_images = _images(tmp_path / "b", 2, lambda _: "b")
    for image in b_images:  # Unique names across sources.
        image.path.rename(image.path.with_name("b_" + image.path.name))
    b_images = [
        ImageRecord(path=i.path.with_name("b_" + i.path.name), width=640, height=480, tag="b")
        for i in b_images
    ]
    coco_dataset.write_dataset(
        datasets_dir=datasets_dir,
        name="a",
        images=a_images,
        boxes=[BoxRecord(0, "GateArm", 1, 1, 10, 10)],
        class_names=["GateArm"],
        val_fraction=0.34,
    )
    coco_dataset.write_dataset(
        datasets_dir=datasets_dir,
        name="b",
        images=b_images,
        boxes=[BoxRecord(0, "LedLight", 1, 1, 10, 10), BoxRecord(1, "GateArm", 1, 1, 10, 10)],
        class_names=["LedLight", "GateArm"],
    )

    summary = coco_dataset.merge_datasets(datasets_dir=datasets_dir, name="ab", sources=["a", "b"])

    assert summary.classes == ["GateArm", "LedLight"]
    assert summary.sources == ["a", "b"]
    # "a" has one tag, so the tag split holds out 2 of its 3 images; "b" has no validation.
    assert summary.splits["train2017"].images == 1 + 2
    assert summary.splits["val2017"].images == 2
    train = json.loads(
        (datasets_dir / "ab" / "annotations" / "instances_train2017.json").read_text()
    )
    assert [image["id"] for image in train["images"]] == [1, 2, 3]
    assert {image["source_dataset"] for image in train["images"]} == {"a", "b"}
    # b's LedLight had id 1 in b and becomes id 2 in the merge.
    led = [ann for ann in train["annotations"] if ann["category_id"] == 2]
    assert len(led) == 1
    assert len(list((datasets_dir / "ab" / "train2017").iterdir())) == 3


def test_merge_datasets__errors(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    with pytest.raises(ValueError, match=r"at least two"):
        coco_dataset.merge_datasets(datasets_dir=datasets_dir, name="m", sources=["a"])
    with pytest.raises(ValueError, match=r"does not exist"):
        coco_dataset.merge_datasets(datasets_dir=datasets_dir, name="m", sources=["a", "b"])


def test_delete_dataset(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    images = _images(tmp_path, 1, lambda _: "")
    coco_dataset.write_dataset(
        datasets_dir=datasets_dir, name="d", images=images, boxes=[], class_names=[]
    )
    coco_dataset.delete_dataset(datasets_dir=datasets_dir, name="d")
    assert coco_dataset.list_datasets(datasets_dir=datasets_dir) == []
    with pytest.raises(ValueError, match=r"does not exist"):
        coco_dataset.delete_dataset(datasets_dir=datasets_dir, name="d")

"""Stores uploaded media files and adds them to an image collection.

Uploaded images are added as image samples. Uploaded videos are decoded into still frames
first, and the frames are added as image samples, so that they are annotated like images.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO
from uuid import UUID

from sqlmodel import Session

from lightly_studio.core.file_outcome_report import AllInputFilesFailedError
from lightly_studio.core.image import add_images
from lightly_studio.dataset import env
from lightly_studio.embed import embed_samples
from lightly_studio.resolvers import tag_resolver
from lightly_studio.upload import video_frames
from lightly_studio.upload.frame_selection import FrameSelectionConfig

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"})


@dataclass
class VideoUploadOutcome:
    """What happened to one uploaded video.

    Attributes:
        file_name: Name of the uploaded file.
        duration_seconds: Length of the video.
        frames_extracted: Number of still frames written to disk.
        frames_added: Number of those frames that became new image samples.
        tag: The tag put on the frames, empty when none.
    """

    file_name: str
    duration_seconds: float
    frames_extracted: int
    frames_added: int
    tag: str = ""


@dataclass
class UploadOutcome:
    """Summary of one upload request.

    Attributes:
        added_sample_ids: Ids of the image samples created by this upload.
        skipped: Number of files that did not become a sample, because they were already
            present, unreadable or not a supported media type.
        embedded: Whether the new samples got embeddings.
        warnings: Human-readable problems that did not fail the upload.
        videos: One entry per uploaded video.
    """

    added_sample_ids: list[UUID] = field(default_factory=list)
    skipped: int = 0
    embedded: bool = False
    warnings: list[str] = field(default_factory=list)
    videos: list[VideoUploadOutcome] = field(default_factory=list)


def add_uploaded_images(
    session: Session,
    collection_id: UUID,
    files: list[tuple[str, IO[bytes]]],
    embed: bool,
    tag: str = "",
) -> UploadOutcome:
    """Stores uploaded image files and adds them to the collection.

    Args:
        session: The database session.
        collection_id: The image collection to add to.
        files: Pairs of original file name and readable byte stream.
        embed: Whether to compute embeddings for the new samples.
        tag: Tag to put on the new samples. Empty means no tag.

    Returns:
        The outcome. Files that already exist in the collection are counted as skipped.

    Raises:
        ValueError: If the collection does not exist or is not an image collection, or if
            no file has a supported image extension.
    """
    target_dir = _collection_dir(collection_id=collection_id) / "images"
    paths: list[Path] = []
    outcome = UploadOutcome()
    for file_name, stream in files:
        if Path(file_name).suffix.lower() not in IMAGE_EXTENSIONS:
            outcome.skipped += 1
            outcome.warnings.append(f"{file_name}: not a supported image type, skipped.")
            continue
        paths.append(save_stream(stream=stream, file_name=file_name, target_dir=target_dir))
    if not paths:
        raise ValueError("None of the uploaded files is a supported image.")

    _add_images(session=session, collection_id=collection_id, paths=paths, outcome=outcome)
    _tag_samples(
        session=session, collection_id=collection_id, tag=tag, sample_ids=outcome.added_sample_ids
    )
    _embed(session=session, collection_id=collection_id, outcome=outcome, embed=embed)
    return outcome


def add_uploaded_videos(  # noqa: PLR0913
    session: Session,
    collection_id: UUID,
    files: list[tuple[str, IO[bytes]]],
    config: FrameSelectionConfig,
    embed: bool,
    tags: list[str] | None = None,
) -> UploadOutcome:
    """Stores uploaded videos, extracts frames from them, and adds the frames as images.

    Args:
        session: The database session.
        collection_id: The image collection to add the frames to.
        files: Pairs of original file name and readable byte stream.
        config: Which timestamps to extract from each video.
        embed: Whether to compute embeddings for the new samples.
        tags: One tag per file, in file order, put on that video's frames. None or a
            shorter list means the video's file stem is used; an empty string means no tag.

    Returns:
        The outcome, with one video entry per decodable upload.

    Raises:
        ValueError: If the collection does not exist or is not an image collection, if the
            config is invalid, or if no file could be decoded as a video.
    """
    config.validate()
    collection_dir = _collection_dir(collection_id=collection_id)
    outcome = UploadOutcome()
    frame_paths: list[list[Path]] = []
    video_tags: list[str] = []
    for index, (file_name, stream) in enumerate(files):
        if Path(file_name).suffix.lower() not in VIDEO_EXTENSIONS:
            outcome.skipped += 1
            outcome.warnings.append(f"{file_name}: not a supported video type, skipped.")
            continue
        video_path = save_stream(
            stream=stream, file_name=file_name, target_dir=collection_dir / "videos"
        )
        try:
            duration = video_frames.get_duration_seconds(video_path=video_path)
            frames = video_frames.extract_frames(
                video_path=video_path,
                output_dir=collection_dir / "frames" / video_path.stem,
                config=config,
            )
        except ValueError as exc:
            outcome.skipped += 1
            outcome.warnings.append(str(exc))
            continue
        outcome.videos.append(
            VideoUploadOutcome(
                file_name=file_name,
                duration_seconds=duration,
                frames_extracted=len(frames),
                frames_added=0,
            )
        )
        frame_paths.append([frame.path for frame in frames])
        video_tags.append(
            tags[index] if tags is not None and index < len(tags) else Path(file_name).stem
        )
    if not outcome.videos:
        raise ValueError("None of the uploaded files could be decoded as a video.")

    for video_outcome, paths, tag in zip(outcome.videos, frame_paths, video_tags):
        before = len(outcome.added_sample_ids)
        _add_images(session=session, collection_id=collection_id, paths=paths, outcome=outcome)
        video_outcome.frames_added = len(outcome.added_sample_ids) - before
        video_outcome.tag = tag
        _tag_samples(
            session=session,
            collection_id=collection_id,
            tag=tag,
            sample_ids=outcome.added_sample_ids[before:],
        )
    _embed(session=session, collection_id=collection_id, outcome=outcome, embed=embed)
    return outcome


def save_stream(stream: IO[bytes], file_name: str, target_dir: Path) -> Path:
    """Writes a stream to target_dir under a safe, unique version of file_name.

    Args:
        stream: The bytes to write, read from its current position.
        file_name: The name the client gave the file. Directory parts and unsafe
            characters are dropped.
        target_dir: Created when missing.

    Returns:
        The path of the written file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_file_name(file_name=file_name)
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    counter = 0
    while True:
        path = target_dir / (safe_name if counter == 0 else f"{stem}_{counter}{suffix}")
        counter += 1
        try:
            # Exclusive create: two concurrent uploads of the same name cannot share a path.
            target = path.open("xb")
        except FileExistsError:
            continue
        with target:
            shutil.copyfileobj(stream, target)
        return path


def _add_images(
    session: Session, collection_id: UUID, paths: list[Path], outcome: UploadOutcome
) -> None:
    if not paths:
        return
    try:
        created = add_images.load_into_dataset_from_paths(
            session=session,
            root_collection_id=collection_id,
            image_paths=[str(path) for path in paths],
            show_progress=False,
        )
    except AllInputFilesFailedError as exc:
        outcome.skipped += len(paths)
        outcome.warnings.append(str(exc))
        return
    outcome.added_sample_ids.extend(created.values())
    outcome.skipped += len(paths) - len(created)


def _tag_samples(session: Session, collection_id: UUID, tag: str, sample_ids: list[UUID]) -> None:
    """Puts the tag on the samples, creating it in the collection when missing."""
    if not tag or not sample_ids:
        return
    tag_row = tag_resolver.get_or_create_sample_tag_by_name(
        session=session, collection_id=collection_id, tag_name=tag
    )
    tag_resolver.add_sample_ids_to_tag_id(
        session=session, tag_id=tag_row.tag_id, sample_ids=sample_ids
    )


def _embed(session: Session, collection_id: UUID, outcome: UploadOutcome, embed: bool) -> None:
    if not embed or not outcome.added_sample_ids:
        return
    try:
        embed_samples.embed_image_samples(
            session=session, collection_id=collection_id, sample_ids=outcome.added_sample_ids
        )
        outcome.embedded = True
    except Exception as exc:
        # The samples are already stored. A failed embedding must not undo the upload.
        logger.warning("Embedding the uploaded samples failed: %s", exc)
        outcome.warnings.append(f"Samples were added but not embedded: {exc}")


def _collection_dir(collection_id: UUID) -> Path:
    return env.LIGHTLY_STUDIO_UPLOAD_DIR / str(collection_id)


def _safe_file_name(file_name: str) -> str:
    name = Path(file_name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload"

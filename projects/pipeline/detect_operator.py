"""An operator that runs one detection model over the images in the current view.

One instance is registered per model found in the models folder, so each model appears
under its own name in the Plugins menu. Boxes go into an annotation source of their own, so
they never mix with the ground truth annotation source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import cv2
from lightly_studio.api.routes.api.validators import Paginated
from lightly_studio.models.annotation.annotation_base import AnnotationCreate, AnnotationType
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.models.collection import SampleType
from lightly_studio.models.image import ImageTable
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult, OperatorStatus
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    BoolParameter,
    FloatParameter,
    StringParameter,
)
from lightly_studio.resolvers import (
    annotation_label_resolver,
    annotation_resolver,
    collection_resolver,
    evaluation_run_resolver,
    image_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from sqlmodel import Session

from pipeline.model_registry import DetectionModel
from pipeline.yolox_onnx import Detection, YoloxDetector

PARAM_SCORE_THRESHOLD = "score_threshold"
PARAM_ANNOTATION_SOURCE = "annotation_source"
PARAM_OVERWRITE = "overwrite"

DEFAULT_SCORE_THRESHOLD = 0.25
GROUND_TRUTH_SOURCE = "annotation"

_PAGE_SIZE = 100  # The upper bound Paginated allows.
_INSERT_BATCH_SIZE = 1000


@dataclass
class DetectOperator(BaseOperator):
    """Runs a detection model on image samples and stores the boxes as predictions."""

    model: DetectionModel | None = None
    name: str = "Detect"
    description: str = ""

    _detector: YoloxDetector | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Names the operator after its model."""
        if self.model is None:
            raise ValueError("DetectOperator needs a model.")
        self.name = f"Detect: {self.model.name}"
        self.description = (
            f"Runs {self.model.onnx_path.name} on the images in the current view and stores "
            f"the boxes in the annotation source '{self.model.default_annotation_source}'. "
            f"Classes: {', '.join(self.model.class_names)}."
        )

    @property
    def parameters(self) -> list[BaseParameter]:
        """Score threshold, target annotation source, and whether to redo done images."""
        assert self.model is not None
        return [
            FloatParameter(
                name=PARAM_SCORE_THRESHOLD,
                default=DEFAULT_SCORE_THRESHOLD,
                description="Minimum confidence for a kept detection (0-1).",
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                default=self.model.default_annotation_source,
                description="Annotation source the predictions are written to.",
            ),
            BoolParameter(
                name=PARAM_OVERWRITE,
                default=False,
                description=(
                    "Redo images in the current view that already have predictions in this "
                    "annotation source. Images outside the view are left alone."
                ),
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        """Image collections only."""
        return [OperatorScope.IMAGE]

    def startup(self) -> None:
        """Loads the ONNX model; a failure marks the operator as errored in the GUI."""
        assert self.model is not None
        try:
            self._detector = YoloxDetector(onnx_path=self.model.onnx_path)
            self.status = OperatorStatus.READY
        except Exception as exc:
            # Surfaced in the GUI as a failed operator instead of crashing the app.
            self.error_message = f"Failed to load {self.model.onnx_path.name}: {exc}"
            self.status = OperatorStatus.ERROR

    def shutdown(self) -> None:
        """Releases the model."""
        self._detector = None
        self.status = OperatorStatus.STOPPED

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Detects objects on the images in the current view and stores the boxes."""
        assert self.model is not None
        if self._detector is None:
            return OperatorResult(success=False, message=self.error_message or "Model not loaded.")
        try:
            settings = _parse_parameters(
                parameters=parameters, default_source=self.model.default_annotation_source
            )
        except ValueError as exc:
            return OperatorResult(success=False, message=str(exc))

        images = get_images_in_view(session=session, context=context)
        if not images:
            return OperatorResult(success=True, message="No images in the current view.")
        todo, already_done = _select_todo(
            session=session, context=context, images=images, settings=settings
        )
        label_ids = _get_or_create_label_ids(
            session=session,
            collection_id=context.collection_id,
            class_names=self.model.class_names,
        )

        buffer: list[AnnotationCreate] = []
        created = processed = unreadable = 0
        for image in todo:
            pixels = cv2.imread(image.file_path_abs)
            if pixels is None:
                unreadable += 1
                continue
            processed += 1
            detections = self._detector.predict(
                image=pixels, score_threshold=settings.score_threshold
            )
            buffer.extend(
                _to_annotation(detection=d, sample_id=image.sample_id, label_ids=label_ids)
                for d in detections
            )
            if len(buffer) >= _INSERT_BATCH_SIZE:
                created += _flush(
                    session=session,
                    collection_id=context.collection_id,
                    buffer=buffer,
                    annotation_source=settings.source,
                )
        created += _flush(
            session=session,
            collection_id=context.collection_id,
            buffer=buffer,
            annotation_source=settings.source,
        )
        if created:
            _mark_evaluations_stale(
                session=session, collection_id=context.collection_id, source=settings.source
            )

        parts = [f"{created} boxes on {processed} images into '{settings.source}'"]
        if already_done:
            parts.append(f"{already_done} already done (enable overwrite to redo)")
        if unreadable:
            parts.append(f"{unreadable} unreadable")
        return OperatorResult(success=True, message=", ".join(parts))


@dataclass(frozen=True)
class _Settings:
    score_threshold: float
    source: str
    overwrite: bool


def _parse_parameters(parameters: dict[str, Any], default_source: str) -> _Settings:
    """Reads and validates the GUI parameters.

    Raises:
        ValueError: With a message for the user when a value is out of range.
    """
    score_threshold = float(parameters.get(PARAM_SCORE_THRESHOLD, DEFAULT_SCORE_THRESHOLD))
    source = str(parameters.get(PARAM_ANNOTATION_SOURCE) or default_source).strip()
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    if source in ("", GROUND_TRUTH_SOURCE):
        raise ValueError(
            f"Pick an annotation_source other than '{GROUND_TRUTH_SOURCE}', so predictions "
            "never mix with the ground truth."
        )
    return _Settings(
        score_threshold=score_threshold,
        source=source,
        overwrite=bool(parameters.get(PARAM_OVERWRITE, False)),
    )


def _select_todo(
    session: Session, context: ExecutionContext, images: list[ImageTable], settings: _Settings
) -> tuple[list[ImageTable], int]:
    """Drops images that already have predictions, or deletes those predictions on overwrite.

    Returns:
        The images to process and how many were skipped as already done.
    """
    in_view = {image.sample_id for image in images}
    existing = _get_existing_predictions(
        session=session, annotation_source=settings.source, collection_id=context.collection_id
    )
    if settings.overwrite:
        stale_ids = [row.sample_id for row in existing if row.parent_sample_id in in_view]
        if stale_ids:
            annotation_resolver.delete_annotations_by_ids(session=session, annotation_ids=stale_ids)
        return images, 0
    done = {row.parent_sample_id for row in existing} & in_view
    return [image for image in images if image.sample_id not in done], len(done)


def _to_annotation(
    detection: Detection, sample_id: UUID, label_ids: dict[int, UUID]
) -> AnnotationCreate:
    return AnnotationCreate(
        annotation_label_id=label_ids[detection.class_index],
        annotation_type=AnnotationType.OBJECT_DETECTION,
        parent_sample_id=sample_id,
        x=round(detection.x1),
        y=round(detection.y1),
        width=max(1, round(detection.x2 - detection.x1)),
        height=max(1, round(detection.y2 - detection.y1)),
        confidence=detection.score,
    )


def _mark_evaluations_stale(session: Session, collection_id: UUID, source: str) -> None:
    annotation_collection_id = collection_resolver.get_or_create_child_collection(
        session=session,
        collection_id=collection_id,
        sample_type=SampleType.ANNOTATION,
        name=source,
    )
    evaluation_run_resolver.mark_stale_by_collection_id(
        session=session, collection_id=annotation_collection_id
    )


def get_images_in_view(session: Session, context: ExecutionContext) -> list[ImageTable]:
    """All images of the current view: the filtered grid, or the selection."""
    context_filter = context.context_filter
    if isinstance(context_filter, SampleFilter):
        image_filter: ImageFilter | None = ImageFilter(sample_filter=context_filter)
    elif isinstance(context_filter, ImageFilter):
        image_filter = context_filter
    else:
        image_filter = None

    images: list[ImageTable] = []
    offset = 0
    while True:
        result = image_resolver.get_all_by_collection_id(
            session=session,
            collection_id=context.collection_id,
            pagination=Paginated(offset=offset, limit=_PAGE_SIZE),
            filters=image_filter,
        )
        images.extend(result.samples)
        if result.next_cursor is None:
            return images
        offset = result.next_cursor


def _get_existing_predictions(
    session: Session, annotation_source: str, collection_id: UUID
) -> list[Any]:
    rows: list[Any] = []
    offset = 0
    while True:
        try:
            result = annotation_resolver.get_all_by_collection_name(
                session=session,
                collection_name=annotation_source,
                parent_collection_id=collection_id,
                pagination=Paginated(offset=offset, limit=_PAGE_SIZE),
            )
        except ValueError:
            # The annotation source does not exist yet.
            return rows
        rows.extend(result.annotations)
        if len(result.annotations) < _PAGE_SIZE:
            return rows
        offset += len(result.annotations)


def _get_or_create_label_ids(
    session: Session, collection_id: UUID, class_names: tuple[str, ...]
) -> dict[int, UUID]:
    """Maps class index to annotation label id, reusing labels that differ only in case."""
    collection = collection_resolver.get_by_id(session=session, collection_id=collection_id)
    if collection is None:
        raise ValueError(f"Collection {collection_id} does not exist.")
    existing = {
        label.annotation_label_name.lower(): label.annotation_label_id
        for label in annotation_label_resolver.get_all(
            session=session, dataset_id=collection.dataset_id
        )
    }
    label_ids: dict[int, UUID] = {}
    for class_index, class_name in enumerate(class_names):
        label_id = existing.get(class_name.lower())
        if label_id is None:
            label_id = annotation_label_resolver.create(
                session=session,
                label=AnnotationLabelCreate(
                    dataset_id=collection.dataset_id, annotation_label_name=class_name
                ),
            ).annotation_label_id
        label_ids[class_index] = label_id
    return label_ids


def _flush(
    session: Session, collection_id: UUID, buffer: list[AnnotationCreate], annotation_source: str
) -> int:
    """Inserts and clears the buffered annotations; returns how many were inserted."""
    if not buffer:
        return 0
    annotation_resolver.create_many(
        session=session,
        parent_collection_id=collection_id,
        annotations=buffer,
        collection_name=annotation_source,
    )
    count = len(buffer)
    buffer.clear()
    return count

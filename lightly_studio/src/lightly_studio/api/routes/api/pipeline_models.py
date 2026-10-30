"""API routes for the detection models in the models folder and their training history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from lightly_studio.api.routes.api.pipeline_training import EpochEvaluationView
from lightly_studio.dataset import env
from lightly_studio.pipeline_training import model_catalog
from lightly_studio.pipeline_training.config import TrainConfig
from lightly_studio.pipeline_training.model_catalog import ModelCard, ModelInfo

pipeline_models_router = APIRouter(tags=["pipeline_models"])


class ModelDatasetView(BaseModel):
    """The dataset a model was trained on, as it was at the start of the run."""

    name: str
    classes: list[str]
    images: int
    boxes: int
    splits: dict[str, dict[str, int]]
    annotation_source: str
    sources: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelDatasetView:
        """Builds the view from the card's dataset entry; older cards may lack counts."""
        return cls(
            name=str(data.get("name", "")),
            classes=list(data.get("classes", [])),
            images=int(data.get("images", 0)),
            boxes=int(data.get("boxes", 0)),
            splits={k: dict(v) for k, v in data.get("splits", {}).items()},
            annotation_source=str(data.get("annotation_source", "")),
            sources=list(data.get("sources", [])),
        )


class ModelCardView(BaseModel):
    """How a model was trained."""

    run_name: str
    dataset: ModelDatasetView
    baseline: str | None
    config: TrainConfig
    best_ap50_95: float | None
    evaluations: list[EpochEvaluationView]
    trained_epochs: int
    created_at: str
    finished_at: str

    @classmethod
    def from_card(cls, card: ModelCard) -> ModelCardView:
        """Builds the view from the stored card."""
        return cls(
            run_name=card.run_name,
            dataset=ModelDatasetView.from_dict(card.dataset),
            baseline=card.baseline,
            config=TrainConfig.model_validate(card.config),
            best_ap50_95=card.best_ap50_95,
            evaluations=[EpochEvaluationView.model_validate(e) for e in card.evaluations],
            trained_epochs=card.trained_epochs,
            created_at=card.created_at,
            finished_at=card.finished_at,
        )


class PipelineModelView(BaseModel):
    """One model of the models folder."""

    name: str
    classes: list[str]
    onnx_path: str
    has_checkpoint: bool
    size_bytes: int
    modified_at: str
    depth: float | None
    width: float | None
    card: ModelCardView | None

    @classmethod
    def from_info(cls, info: ModelInfo) -> PipelineModelView:
        """Builds the view from the catalog entry."""
        return cls(
            name=info.name,
            classes=info.classes,
            onnx_path=str(info.onnx_path),
            has_checkpoint=info.checkpoint_path is not None,
            size_bytes=info.size_bytes,
            modified_at=info.modified_at,
            depth=info.depth,
            width=info.width,
            card=ModelCardView.from_card(info.card) if info.card is not None else None,
        )


@pipeline_models_router.get("/pipeline_models", response_model=list[PipelineModelView])
def list_pipeline_models() -> list[PipelineModelView]:
    """The models in the models folder, newest first, with their training history."""
    return [
        PipelineModelView.from_info(info)
        for info in model_catalog.list_models(models_dir=env.LIGHTLY_STUDIO_MODELS_DIR)
    ]

"""A small, auditable quantitative research pipeline for learning."""

from .config import PipelineConfig
from .pipeline import PipelineResult, run_pipeline

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]


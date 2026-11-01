from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .pipeline import PipelineResult


def configure_logging(log_dir: str | Path) -> logging.Logger:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quant_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(directory / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def save_run_artifacts(result: PipelineResult, report_root: str | Path) -> Path:
    """Persist inputs and outputs so every research run can be inspected later."""

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(report_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    result.prices.to_csv(run_dir / "prices.csv")
    result.factors["composite"].to_csv(run_dir / "factor_scores.csv")
    result.target_weights.to_csv(run_dir / "target_weights.csv")
    result.backtest.equity.to_csv(run_dir / "equity.csv")
    result.backtest.net_returns.to_csv(run_dir / "returns.csv")
    result.portfolio.to_csv(run_dir / "latest_portfolio.csv")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, ensure_ascii=False, indent=2, allow_nan=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(result.config.__dict__, handle, ensure_ascii=False, indent=2)

    configure_logging(Path(report_root).parent / "logs").info("saved run %s", run_id)
    return run_dir


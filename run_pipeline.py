from pathlib import Path

from quant_pipeline import PipelineConfig, run_pipeline
from quant_pipeline.data import generate_demo_prices
from quant_pipeline.engineering import save_run_artifacts


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_pipeline(generate_demo_prices(), PipelineConfig())
    output = save_run_artifacts(result, root / "reports")
    print("Pipeline completed")
    print(f"Sharpe: {result.metrics['sharpe']:.3f}")
    print(f"Max drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Artifacts: {output}")


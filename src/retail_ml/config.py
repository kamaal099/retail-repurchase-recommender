"""Central configuration."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")
    artifact_dir: Path = Path("artifacts")   # trained models used by the API
    horizon_days: int = 90      # prediction window after the cutoff date
    top_k: int = 10             # recommendations per customer
    test_size: float = 0.2
    random_state: int = 42

"""Fixed project paths (see AGENT_SPEC §1.2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RAW = ROOT / "data" / "raw" / "osf"
RAW_DATA = RAW / "data"
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "external" / "analysis"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

for _p in (PROCESSED, RUNS, RESULTS, REPORTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

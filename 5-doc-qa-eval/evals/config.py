"""Settings for the eval suite. Paths are relative to the doc-qa project root."""
from rag.config import BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"
BASELINE_PATH = BASE_DIR / "evals" / "baseline.json"
DEFAULT_TOLERANCE = 0.05   # a metric may drop by up to 5 percentage points before we call it a regression

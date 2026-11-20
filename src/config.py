"""Central place for paths and fixed parameters, so nothing is hard-coded elsewhere."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Which dataset and which feature set a run uses. Both are read from the
# environment so that notebooks and tests keep working unchanged:
#   CITATION_DATASET  pubmed (default) | cora | citeseer
#   FEATURE_SET       full (default)   | no_closeness   (test branch for the closeness question)
DATASET = os.environ.get("CITATION_DATASET", "pubmed")
FEATURE_SET = os.environ.get("FEATURE_SET", "full")
EXCLUDED_FEATURES = {"full": set(), "no_closeness": {"closeness"}}[FEATURE_SET]

# The original PubMed run keeps the paths it always had; every other run gets its own folder.
RUN_NAME = "" if (DATASET, FEATURE_SET) == ("pubmed", "full") else (
    DATASET + ("" if FEATURE_SET == "full" else "_" + FEATURE_SET))

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / RUN_NAME
RESULTS_DIR = ROOT / "results" / RUN_NAME
DOCS_DIR = ROOT / "docs"

NODES_FILE = RAW_DIR / "Pubmed-Diabetes.NODE.paper.tab.txt"
EDGES_FILE = RAW_DIR / "Pubmed-Diabetes.DIRECTED.cites.tab"

SEED = 42

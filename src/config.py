"""Central place for paths and fixed parameters, so nothing is hard-coded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

NODES_FILE = RAW_DIR / "Pubmed-Diabetes.NODE.paper.tab.txt"
EDGES_FILE = RAW_DIR / "Pubmed-Diabetes.DIRECTED.cites.tab"

SEED = 42

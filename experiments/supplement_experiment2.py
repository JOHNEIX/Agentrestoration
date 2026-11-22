"""Supplement an existing experiment without restoration API calls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

from experiments.experiment2.supplement import main

if __name__ == "__main__":
    main()

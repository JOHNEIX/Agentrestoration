"""Reaggregate saved metrics after excluding trials where B returned the input."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

from experiments.experiment2.filtered import main

if __name__ == "__main__":
    main()

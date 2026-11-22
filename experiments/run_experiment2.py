"""Command-line entry point for experiment 2."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

from experiments.experiment2.runner import main

if __name__ == "__main__":
    main()

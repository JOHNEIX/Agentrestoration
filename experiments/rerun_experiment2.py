"""Rerun B/C while importing the original A analyses and first candidates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"src")]

from experiments.experiment2.rerun import main

if __name__ == "__main__":
    main()

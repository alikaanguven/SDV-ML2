from pathlib import Path
import user_scripts.ABCDiscoTEC_loss as ABCD
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "training"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))


print(__file__)
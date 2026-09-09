"""
pytest configuration for CreditWise tests.
Adds the project root to sys.path so src imports work from any directory.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

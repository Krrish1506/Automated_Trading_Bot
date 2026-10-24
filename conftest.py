# conftest.py — pytest configuration for the Trading project.
# Ensures the project root is on sys.path so `from forecast.xxx import ...` works.
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

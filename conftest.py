import sys
from pathlib import Path

# Add the project root to the path so `from src.arxiv_search import ...` works
# when pytest imports the test modules.
sys.path.insert(0, str(Path(__file__).parent))

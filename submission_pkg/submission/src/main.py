"""DrugCLIP Virtual Screening Agent - Thin Entry Point."""
import sys
from pathlib import Path
# Add src/ to path so we can import agent as a package
sys.path.insert(0, str(Path(__file__).parent))
# Import from the agent package directly (uses relative imports internally)
from agent import main

if __name__ == '__main__':
    main()
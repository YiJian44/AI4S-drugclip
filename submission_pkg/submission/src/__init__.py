"""DrugCLIP Virtual Screening Agent Package."""
import sys
from pathlib import Path

# Allow imports from agent/, models/, utils/ as subpackages
sys.path.insert(0, str(Path(__file__).parent))

# Import from subpackages using absolute imports (no leading dot)
from agent import DrugClipAgent, Evaluator, OptimizationHistory, opt_history, main
from models import Scorer, SEARCH_SPACE
from utils.io import extract_smiles_from_mol2, mol_to_fp, rrf_fusion, weighted_sum
from utils.logging import setup_logger, log_main, log_train, log_score

__version__ = '1.0.0'
__all__ = [
    'DrugClipAgent', 'Evaluator', 'OptimizationHistory', 'opt_history', 'main',
    'Scorer', 'SEARCH_SPACE',
    'setup_logger', 'log_main', 'log_train', 'log_score',
    'extract_smiles_from_mol2', 'mol_to_fp', 'rrf_fusion', 'weighted_sum',
]
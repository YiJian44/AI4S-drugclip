"""DrugCLIP utility modules."""
import sys
from pathlib import Path

# Add src to path for absolute imports
_src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_src_dir))

from utils import logging as _logging
from utils import io as _io

setup_logger = _logging.setup_logger
log_main  = _logging.log_main
log_train = _logging.log_train
log_score = _logging.log_score
LOG_DIR = _logging.LOG_DIR
ITER_LOG = _logging.ITER_LOG
OPT_LOG  = _logging.OPT_LOG

extract_smiles_from_mol2 = _io.extract_smiles_from_mol2
mol_to_fp = _io.mol_to_fp
rrf_fusion = _io.rrf_fusion
weighted_sum = _io.weighted_sum

__all__ = [
    'setup_logger', 'log_main', 'log_train', 'log_score',
    'LOG_DIR', 'ITER_LOG', 'OPT_LOG',
    'extract_smiles_from_mol2', 'mol_to_fp', 'rrf_fusion', 'weighted_sum',
]
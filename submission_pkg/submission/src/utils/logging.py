"""DrugCLIP utilities - logging module."""
import logging
import sys
from pathlib import Path

# Path discovery - relative to submission root (drugclip/submission_pkg/submission/)
_SUBMISSION_ROOT = Path(__file__).parent.parent.parent  # utils -> src -> submission
_PACKAGE_ROOT = _SUBMISSION_ROOT

LOG_DIR = _PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
ITER_LOG = LOG_DIR / 'iteration_log.jsonl'
OPT_LOG  = _PACKAGE_ROOT / 'optimization_history.json'

LOG_NAME_AGENT  = 'agent'
LOG_NAME_TRAIN  = 'train'
LOG_NAME_SCORE  = 'score'

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'

def setup_logger(name):
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    fh = logging.FileHandler(LOG_DIR / f'{name}.log', mode='a')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.addHandler(fh); lg.addHandler(ch)
    return lg

log_main  = setup_logger(LOG_NAME_AGENT)
log_train = setup_logger(LOG_NAME_TRAIN)
log_score = setup_logger(LOG_NAME_SCORE)
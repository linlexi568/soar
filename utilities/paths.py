import os
from datetime import datetime

"""Centralized path constants & helpers for results/artifact organization.

This module defines canonical directories so that all scripts avoid hardcoded
string literals. Use ensure_dir(...) before writing files.
"""

# Root directories
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
GLOBAL_RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')  # Only aggregated summaries & figures

# Global subdirectories (kept minimal & clean)
SUMMARY_DIR = os.path.join(GLOBAL_RESULTS_ROOT, 'summaries')
FIGURES_DIR = os.path.join(GLOBAL_RESULTS_ROOT, 'figures')

# Per-component results roots (component packages own their internal structure)
# Updated to new numbered folder layout; prefer 01_soar for the new package.
SOAR_RESULTS_ROOT = os.path.join(PROJECT_ROOT, '01_soar', 'results')  # new default
PILIGHT_RESULTS_ROOT = os.path.join(PROJECT_ROOT, '01_pi_light', 'results')  # legacy
BASELINE_RESULTS_ROOT = os.path.join(PROJECT_ROOT, '02_baseline', 'results')  # legacy: 'baseline/results'
CMAES_RESULTS_ROOT = os.path.join(PROJECT_ROOT, '03_CMA-ES', 'results')       # legacy: 'CMA-ES/results'
GSN_RESULTS_ROOT = os.path.join(PROJECT_ROOT, '04_nn_baselines', 'results')   # legacy: 'nn_baselines/results'

# Common subfolder names used inside component roots
EVAL_SUBDIR = 'eval'

def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def init_global_results_dirs():
    """Ensure global minimal dirs exist."""
    ensure_dir(SUMMARY_DIR)
    ensure_dir(FIGURES_DIR)
    # Ensure component roots exist lazily to reduce clutter
    try:
        ensure_dir(SOAR_RESULTS_ROOT)
    except Exception:
        pass

# Initialize on import (idempotent)
init_global_results_dirs()

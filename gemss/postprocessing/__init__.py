"""
This module aggregates all postprocessing and downstream modeling functionalities, including:
- Outlier detection and handling
- Extraction of solutions from the optimization run
- Simple regression analyses
"""

from .outliers import *
from .result_postprocessing import *
from .simple_regressions import *
from .tabpfn_evaluation import *

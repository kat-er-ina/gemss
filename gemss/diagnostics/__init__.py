"""
Diagnostics package for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This package contains modules for performance testing, optimization analysis,
and intelligent parameter recommendations.

Modules:
- performance_tests: Automated performance diagnostics and testing
- recommendations: Intelligent parameter recommendation system
- recommendation_messages: Message templates for recommendations
"""

from .performance_tests import *
from .recommendations import *
from .recommendation_messages import *
from .result_postprocessing import *

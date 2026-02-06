"""
Diagnostics package for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This package contains modules for performance testing, optimization analysis,
and intelligent parameter recommendations.

Modules:
- performance_tests: Automated performance diagnostics and testing
- recommendations: Intelligent parameter recommendation system
- recommendation_messages: Message templates for recommendations
"""

from gemss.diagnostics.performance_tests import PerformanceTests
from gemss.diagnostics.recommendation_messages import (
    RECOMMENDATION_MESSAGES,
    get_available_recommendation_keys,
    get_recommendation_message,
)
from gemss.diagnostics.recommendations import RecommendationEngine

"""
ecopystats
==========

Python package for ecological statistics.

"""

import logging
import os


logger = logging.getLogger(__name__)

#

_submodules = [
    "diversity",
]

# from .diversity import shannon_diversity, simpson_diversity
# from .distance import braycurtis_distance, jaccard_distance, sorensen_distance
# from .rarefaction import rarefaction_curve
# from .stats import ...


__version__ = "0.1.0"
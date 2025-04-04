"""
ecopystats
==========

Python package for ecological statistics.

"""

__author__ = """Pedro Pablo Silva Antilef"""
__email__ = "pedropablosilvaa@gmail.com"
__version__ = "0.1.0"


import logging
import os


logger = logging.getLogger(__name__)


from .diversity import diversity
# from .distance import braycurtis_distance, jaccard_distance, sorensen_distance
# from .rarefaction import rarefaction_curve
# from .stats import ...


__version__ = "0.1.0"
"""
Some utility functions.
"""

import logging
import re

logger = logging.getLogger(__name__)


def make_negation_name(column_name: str, suffix: str = "_x") -> str:
    """Make a new column name based for the negative value

    Returns
    -------
    negation_name : str
    """
    negation_name = re.sub("_\d\.\d$", "", column_name) + suffix
    if re.search("_\d\.\d$", column_name):
        negation_name += re.search("_\d\.\d$", column_name).group()
    return negation_name

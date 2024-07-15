import sys
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImputeGaps:
    """
    add documentation here
    """

    def __init__(
        self,
        records_df=None,
        variables=None,
        settings=None
    ):

        self.records_df = records_df,
        self.variables = variables,
        self.settings = settings

    def impute_gaps(self):
        """counter the number of evaluations per phase"""


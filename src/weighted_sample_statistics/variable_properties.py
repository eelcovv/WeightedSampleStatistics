"""
The variabele properties defined
"""
import logging

logger = logging.getLogger(__name__)


class VariableProperties:
    """
    Initializes the VariableProperties object

    Args:
        variables (DataFrame): DataFrame containing the variables.
        column (str): The name of the column containing the variables
        skip_columns (list): list with columns to skip
    """

    def __init__(self, variables, column, skip_columns=None):
        """
        Initialize the VariableProperties object.

        Parameters
        ----------
        variables : DataFrame
            DataFrame containing the variables.
        column : str
            The name of the column containing the variables.
        skip_columns : list, optional
            List of columns to skip.
            Default is None.
        """
        self.variables = variables
        self.column = column
        self.skip_columns = skip_columns
        self.type = None  # Variable type
        self.gewicht = None  # Weight of the variable
        self.filter = None  # Filter applied to the variable
        self.report_conditional = False  # Flag for conditional reporting
        self.report_number = False  # Flag for number reporting
        self.eval = None  # Evaluation expression
        self.negation_suffix = None  # Suffix for negated variables
        self.module_key = None  # Key for module association
        self.get_valid_var_type()  # Validate and set variable type

    def get_valid_var_type(self) -> None:
        """Check if the variable type is valid and return"""
        try:
            self.type = self.variables.loc[self.column, "type"]
        except KeyError:
            logger.debug(f"No valid var type in {self.column}")
        try:
            self.filter = self.variables.loc[self.column, "filter"]
        except KeyError:
            logger.debug(f"Variable {self.column} does not has a valid filter")
            self.type = None
        try:
            self.gewicht = self.variables.loc[self.column, "gewicht"]
        except KeyError:
            logger.debug(f"Variable {self.column} does not has a valid gewicht")
            self.type = None

        if self.type in ["index", "date", "str"]:
            logger.debug("Skipping imputing var type {}".format(self.type))
            self.type = None

        if self.skip_columns is not None and self.column in self.skip_columns:
            logger.debug("Skipping imputing variable {}".format(self.column))
            self.type = None

        try:
            self.report_conditional = self.variables.loc[
                self.column, "report_conditional"
            ]
        except KeyError:
            logger.debug(f"No valid var type in {self.column}")

        try:
            self.report_number = self.variables.loc[self.column, "report_number"]
        except KeyError:
            logger.debug(f"No valid report number type in {self.column}")

        try:
            self.eval = self.variables.loc[self.column, "eval"]
        except KeyError:
            logger.debug(f"No valid eval type in {self.column}")

        try:
            self.negation_suffix = self.variables.loc[self.column, "negation_suffix"]
        except KeyError:
            logger.debug(f"No valid negation_suffix in {self.column}")

        try:
            self.module_key = self.variables.loc[self.column, "module_key"]
        except KeyError:
            logger.debug(f"No valid module_key in {self.column}")

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series

from pandas import DataFrame

logger = logging.getLogger(__name__)


class ImputeGaps:
    """
    Initializes the ImputeGaps object.

    Arguments
    ---------
    records_df: DataFrame
        DataFrame containing variables with missing values.
    variables: dict
        Dictionary with information about the variables to impute.
    impute_settings: dict
        Dictionary with imputation settings.
    record_id: str
        Name of the variable by which a record is identified (e.g. be_id)

    Notes
    ----------
    *   De dictionary 'variables' is in principe de pd.DataFrame 'self.variables' uit de ICT analyser, geconverteerd
        naar een dictionary. Als preprocessing stap moet hierbij de kolom 'filter' zijn platgeslagen, oftewel: het mag
        geen dictionary meer zijn. De dictionary 'variables' moet ten minste de volgende kolommen bevatten:
        [["type", "no_impute", "filter"]], met optioneel: "impute_only".
    *   De dict 'impute_settings' is een nieuw kopje onder 'General' in de settingsfile. Een belangrijk subkopje is
        'group_by', wat er zo kan uitzien:
         group_by: "sbi_digit2, gk6_label; gk6_label"
        Dit betekent dat er eerst wordt geïmputeerd in strata o.b.v. sbi_digit2 en gk6_label. Als dat niet lukt, wordt
        alleen geïmputeerd o.b.v. gk6_label. Op dezelfde manier kunnen meer opties worden toegevoegd.
    """

    def __init__(
        self,
        records_df: DataFrame = None,
        variables: dict = None,
        impute_settings=None,
        impute_methods_per_type=None,
        record_id_key=None,
        group_by=None,
    ):

        self.group_by = group_by
        self.records_id_key = record_id_key

        self.original_indices = records_df.index.names

        # pass a copy of the records and reorganize the data
        self.records_df = self.prepare_data(records_df.copy())

        self.variables = variables
        self.impute_settings = impute_settings
        self.impute_methods_per_type = impute_methods_per_type

        self.impute_gaps()

    def prepare_data(self, records_df: DataFrame) -> DataFrame:
        """
        Make sure that all possible variables for group by are set as index
        """
        new_records_df = records_df.reset_index()
        new_index = self.group_by + [self.records_id_key]
        new_records_df = new_records_df.set_index(new_index, drop=True)

        return new_records_df

    def fill_missing_data(self, col, how) -> DataFrame:
        """Impute missing values for one variable of a particular stratum (subset)

        Parameters
        ----------
        col: pd.Series
            pd.Series with one column that contains missing values.
        how: String {"mean", "median", "pick", "nan", "pick1"},
            Method that should be used to fill the missing values;
            - mean: Impute with the mean
            - median: Impute with the median
            - pick: Impute with a random value (for categorical variables)
            - nan: Impute with the value 0
            - pick1: Impute with the value 1

        Returns
        -------
        imputed_col: pd.Series
            pd.Series with imputed values.
        """
        imputed_col = col.copy()

        # Create a mask with the size of a column with True for all missing values
        mask = imputed_col.isnull()

        # Skip if there are no missing values
        if not mask.any():
            return imputed_col

        # Fill missing values depending on which method to use
        if how == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                samples = np.full(mask.size, fill_value=imputed_col.mean())
        elif how == "median":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                samples = np.full(mask.size, fill_value=imputed_col.median())
        elif how == "mode":
            # Try to obtain mode
            try:
                fill_val = imputed_col.mode()[0]
            # Mode cannot be obtained when all values in that stratum are missing
            except KeyError:
                fill_val = np.nan
            samples = np.full(mask.size, fill_value=fill_val)
        elif how == "nan":
            samples = np.full(imputed_col.isnull().sum(), fill_value=0)
        elif how == "pick1":
            # Let pick1 work with values that are '1.0 and 0.0'.
            try:
                valid_values = imputed_col.cat.categories
                if "1.0" in valid_values:
                    samples = np.full(imputed_col.isnull().sum(), fill_value="1.0")
                elif valid_values.dtype == "object":
                    imputed_col = imputed_col.cat.add_categories("1.0")
                    samples = np.full(imputed_col.isnull().sum(), fill_value="1.0")
                elif 1 in valid_values:
                    samples = np.full(imputed_col.isnull().sum(), fill_value=1)
                else:
                    imputed_col = imputed_col.cat.add_categories(1)
                    samples = np.full(imputed_col.isnull().sum(), fill_value=1)
            except AttributeError:
                if "1.0" in imputed_col[~mask].values:
                    samples = np.full(imputed_col.isnull().sum(), fill_value="1.0")
                else:
                    samples = np.full(imputed_col.isnull().sum(), fill_value=1.0)
        elif how == "pick":
            if self.impute_settings["set_seed"] is not None:
                np.random.seed(self.impute_settings["set_seed"])
            number_of_nans = mask.sum()
            valid_values = imputed_col[~mask].values
            if valid_values.size == 0:
                return imputed_col
            else:
                samples = np.random.choice(
                    valid_values, size=number_of_nans, replace=True
                )
        else:
            raise ValueError("Not a valid choice for how {}.".format(how))
        # Fill the missing values with the values from samples
        if samples.size > 1:
            imputed_col[mask] = samples
        else:
            imputed_col.loc[mask] = samples
        return imputed_col

    def impute_gaps(self) -> None:
        """
        Impute all missing values in a dataframe.

        Returns
        -------
        None
        """

        # Iterate over variables
        for col_name in self.records_df.columns:
            # Check if there is information available about the variable
            try:
                var_type = self.variables[col_name]["type"]
            except KeyError as err:
                logger.info(f"Geen variabele info voor: {col_name}, {err}")
                continue

            # Check if the variable has a 'no_impute' flag or if its type should not be imputed
            if (
                self.variables[col_name]["no_impute"]
                or var_type in self.impute_settings["imputation_methods"]["skip"]
            ):
                logger.info(
                    "Skip imputing variable {} of var type {}".format(
                        col_name, var_type
                    )
                )
                continue

            # Variabele to impute
            col_to_impute = self.records_df[col_name]
            start_type = col_to_impute.dtype

            # Get filter(s) if provided
            if self.variables[col_name]["impute_only"] is None:
                var_filter = self.variables[col_name]["filter"]
            else:
                var_filter = self.variables[col_name]["impute_only"]

            # If a filter is provided, use it to filter the records
            if var_filter is not np.nan and var_filter is not None:
                # Apply filter with a variable that is also one of the index variables
                if var_filter in col_to_impute.index.names:
                    try:
                        col_to_impute = col_to_impute[
                            col_to_impute.index.get_level_values(var_filter) == 1
                        ]
                    except KeyError as err:
                        logger.warning(f"Failed to filter with {var_filter}, {err}")
                # Apply filter with a regular variable
                else:
                    try:
                        filter_mask = self.records_df[var_filter]
                    except KeyError as err:
                        logger.warning(f"Failed to filter with {var_filter}, {err}")
                    else:
                        col_to_impute = col_to_impute.loc[filter_mask == 1]

            # Compute number of missing values
            n_nans = n_nans1 = col_to_impute.isnull().sum()

            # Skip if there are no missing values
            if n_nans1 == 0:
                logger.debug(
                    "Skip imputing {}. It has no invalid numbers".format(col_name)
                )
                continue

            logger.info("Impute gaps {:20s} ({})".format(col_name, var_type))
            logger.debug("Imputing variable {}".format(col_name))
            logger.debug(
                "Filling {} with {} / {} nans ({:.1f} %)"
                "".format(
                    col_name,
                    n_nans1,
                    col_to_impute.size,
                    (n_nans1 / col_to_impute.size) * 100,
                )
            )

            # Get which imputing method to use
            imputation_dict = self.impute_settings["imputation_methods"]
            not_none = [
                i for i in imputation_dict.keys() if imputation_dict[i] is not None
            ]

            if not pd.isna(self.variables[col_name]["impute_method"]):
                how = self.variables[col_name]["impute_method"]
            else:
                for key, val in imputation_dict.items():
                    if key in not_none and var_type in imputation_dict[key]:
                        how = key
                        # Convert categorical (dict) variables to categorical
                        if var_type == "dict":
                            col_to_impute = col_to_impute.astype("category")
                        continue

            if how is None:
                logger.warning("Imputation method not found!")
            else:
                logger.debug(f"Fill gaps by taking the {how} of the valid values")

            def fill_gaps(col):
                """
                Impute missing values for one variable for a particular subset (aka stratum)

                Parameters
                ----------
                col: pd.Series
                    pd.Series with one column that contains missing values.

                Returns
                -------
                imputed_col: pd.Series
                    pd.Series with imputed values.
                """
                imputed_col = self.fill_missing_data(col, how=how)
                return imputed_col

            # Iterate over the variables in the group_by-list and try to impute until there are no more missing values
            n_iterations = 0
            while n_nans1 > 0 and n_iterations < len(self.group_by):
                group_keys = self.group_by[n_iterations].split(", ")
                df_grouped = col_to_impute.groupby(
                    group_keys, group_keys=False
                )  # Do group by
                col_to_impute = df_grouped.apply(fill_gaps)  # Impute missing values
                n_nans1 = col_to_impute.isnull().sum()  # Count remaining missing values
                n_iterations += 1
            else:
                if n_nans == n_nans1 and n_nans > 0:
                    logger.warning(
                        "Failed imputing any gap for {}: {} gaps imputed and {} gaps remaining."
                        "".format(col_name, n_nans - n_nans1, n_nans)
                    )
                elif n_nans1 > 0:
                    logger.warning(
                        "Failed imputing some gaps for {}: {} gaps imputed and {} gaps remaining. "
                        "".format(col_name, n_nans - n_nans1, n_nans1)
                    )
                elif n_nans1 == 0:
                    logger.warning(
                        "Successfully finished imputing all {}/{} = {} gaps for {}."
                        "".format(
                            n_nans,
                            col_to_impute.size,
                            n_nans / col_to_impute.size,
                            col_name,
                        )
                    )

            # Replace original column by imputed column
            self.records_df[col_name] = col_to_impute.astype(start_type)

        # Set original indices back
        self.records_df = self.records_df.copy()
        self.records_df.reset_index(inplace=True)

        if None not in self.original_indices:
            self.records_df.set_index(self.original_indices, inplace=True)
            logger.info("Set index {}.".format(self.original_indices))

        logger.info("Done")

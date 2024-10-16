import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataFrameType = Union["DataFrame", None]
DataFrameLikeType = Union["DataFrame", "Series", None]


def fill_missing_data(col, how) -> DataFrameLikeType:
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
    imputed_col: DataFrameLikeType
        Series with imputed values
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
        # Try to get mode
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
        number_of_nans = mask.sum()
        valid_values = imputed_col[~mask].values
        if valid_values.size == 0:
            return imputed_col
        else:
            samples = np.random.choice(valid_values, size=number_of_nans, replace=True)
    else:
        raise ValueError("Not a valid choice for how {}.".format(how))
    # Fill the missing values with the values from samples
    if samples.size > 1:
        imputed_col[mask] = samples
    else:
        imputed_col.loc[mask] = samples
    return imputed_col


class ImputeGaps:
    """
    Initializes the ImputeGaps object.

    Arguments
    ---------
    group_by: list | str
        List with the variables by which the records should be grouped.
        The first variable is the most important one.
    id_key: str
        Name of the variable by which a record is identified (e.g. be_id)
    variables: dict
        Dictionary with information about the variables to impute.
    imputation_methods: dict
        Dictionary with imputation methods per data type.
    seed: int
        Seed for random number generator.

    Notes
    ----------
    *   De dictionary 'variables' is in principe de pd.DataFrame 'self.variables' uit de ICT
        analyser, geconverteerd naar een dictionary.
        Als preprocessing stap moet hierbij de kolom 'filter' zijn platgeslagen, oftewel: het mag
        geen dictionary meer zijn. De dictionary 'variables' moet ten minste de volgende kolommen
        bevatten: [["type", "no_impute", "filter"]], met optioneel: "impute_only".
    *   De dict 'impute_settings' is een nieuw kopje onder 'General' in de settingsfile.
        Een belangrijk subkopje is 'group_by', wat er zo kan uitzien:
         group_by: "sbi_digit2, gk6_label; gk6_label"
        Dit betekent dat er eerst wordt geïmputeerd in strata o.b.v. sbi_digit2 en gk6_label.
        Als dat niet lukt, wordt alleen geïmputeerd o.b.v. gk6_label. Op dezelfde manier kunnen
        meer opties worden toegevoegd.
    """

    def __init__(
        self,
        group_by: list | str,
        id_key: str,
        imputation_methods: dict | None = None,
        variables: dict | None = None,
        seed: int = None,
    ):
        logger.debug("Start with debug")
        logger.info("Start with info")
        logger.warning("Start with warning")

        if isinstance(group_by, str):
            self.group_by: list = [group_by]
        else:
            self.group_by: list = group_by
        # self.group_by = self.impute_settings["group_by"].split("; ")
        self.imputation_methods = imputation_methods

        if seed is not None:
            # Set seed for random number generator. Only needs to be done one time
            np.random.seed(seed)

        self.variables = variables
        self.id_key = id_key

    def impute_gaps(self, records_df: DataFrameType) -> DataFrameType:
        """
        Impute all missing values in a dataframe for indices group_by.

        Parameters
        ----------
        records_df: DataFrameType
            DataFrame containing variables with missing values.

        Returns
        -------
        DataFrameType:
            DataFrame with imputed values.
        """

        original_indices = records_df.index.names
        number_of_dimensions = len(self.group_by)
        for group_dim in range(number_of_dimensions):
            max_dim = number_of_dimensions - group_dim
            if max_dim > 0:
                indices = self.group_by[:max_dim]
                new_index = [self.id_key] + indices
            else:
                new_index = self.id_key

            records_df = self.impute_gaps_for_dimensions(records_df, new_index)

        # Set original indices back
        records_df.reset_index(inplace=True)

        if None not in original_indices:
            records_df.set_index(original_indices, inplace=True)
            logger.debug("Set index {}.".format(original_indices))

        return records_df

    def impute_gaps_for_dimensions(
        self, records_df: DataFrameType, new_index: list
    ) -> DataFrameType:
        """
        Impute all missing values in a dataframe for a particular subset (aka stratum).

        Parameters
        ----------
        records_df: DataFrameType
            DataFrame containing variables with missing values.
        new_index: list
            List with the new index for the DataFrame.

        Returns
        -------
        DataFrameType:
            DataFrame with imputed values for indices new_index.
        """

        records_df = records_df.reset_index()
        records_df.set_index(new_index, inplace=True)
        how = None

        # Iterate over variables
        for col_name in records_df.columns:
            # Check if there is information available about the variable
            try:
                var_type = self.variables[col_name]["type"]
            except KeyError as err:
                logger.info(f"Geen variabele info voor: {col_name}, {err}")
                continue

            # Check if the variable has a 'no_impute' flag or if its type should not be imputed
            if self.variables[col_name]["no_impute"] or var_type in self.imputation_methods["skip"]:
                logger.info("Skip imputing variable {} of var type {}".format(col_name, var_type))
                continue

            # Variabele to impute
            col_to_impute = records_df[col_name]
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
                        filter_mask = records_df[var_filter]
                    except KeyError as err:
                        logger.warning(f"Failed to filter with {var_filter}, {err}")
                    else:
                        col_to_impute = col_to_impute.loc[filter_mask == 1]

            # Compute number of missing values
            number_of_nans_before = col_to_impute.isnull().sum()

            # Skip if there are no missing values
            if number_of_nans_before == 0:
                logger.debug(f"Skip imputing {col_name}. It has no invalid numbers")
                continue

            logger.info("Impute gaps {:20s} ({})".format(col_name, var_type))
            logger.debug("Imputing variable {}".format(col_name))
            col_size = col_to_impute.size
            percentage_to_replace = round(100 * number_of_nans_before / col_size, 1)
            logger.debug(
                f"Filling {col_name} with {number_of_nans_before} / {col_size} nans "
                f"({percentage_to_replace:.1f} %)"
            )

            # Get which imputing method to use
            imputation_dict = self.imputation_methods
            not_none = [i for i in imputation_dict.keys() if imputation_dict[i] is not None]

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
                imputed_col = fill_missing_data(col, how=how)
                return imputed_col

            # Iterate over the variables in the group_by-list and try to impute until there are no
            # more missing values
            df_grouped = col_to_impute.groupby(new_index, group_keys=False)  # Do group by
            col_to_impute = df_grouped.apply(fill_gaps)  # Impute missing values

            number_of_nans_after = col_to_impute.isnull().sum()

            number_of_removed_nans = number_of_nans_after - number_of_nans_before

            if number_of_removed_nans == 0 and number_of_nans_before > 0:
                logger.warning(
                    f"Failed imputing any gap for {col_name} for {new_index}: "
                    f"{number_of_removed_nans} gaps imputed / {number_of_nans_after} gaps remaining"
                )
            elif number_of_nans_after > 0:
                logger.warning(
                    f"Failed imputing some gap for {col_name} for {new_index}: "
                    f"{number_of_removed_nans} gaps imputed / {number_of_nans_after} gaps remaining"
                )
            elif number_of_nans_after == 0:
                col_size = col_to_impute.size
                percentage_replaced = round(100 * number_of_nans_before / col_size, 1)
                logger.info(
                    f"Successfully imputed all {number_of_nans_before}/{col_size} "
                    f"({percentage_replaced:.1f} %) gaps for {col_name}."
                )
            else:
                logger.warning("Something went wrong with imputing gaps for {}.".format(col_name))

            # Replace original column by imputed column
            records_df[col_name] = col_to_impute.astype(start_type)

        return records_df

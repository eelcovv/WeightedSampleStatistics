"""
Some utility functions.
"""

import logging
import re
from typing import Optional, Tuple, TypeVar, List, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)

DataFrameType = TypeVar("DataFrameType", bound=DataFrame)


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


def rename_variables(dataframe: DataFrameType, variables: DataFrameType) -> None:
    """
    Rename dataframe columns according to the variables DataFrame

    This function takes a dataframe and a variables dataframe as input and
    renames the columns in the dataframe according to the original names in
    the variables dataframe.

    If the original name is not defined for a variable, the variable will be
    skipped.
    If the original name is already defined for another variable, the
    variable will also be skipped.

    Parameters
    ----------
    dataframe : DataFrameType
        Dataframe to rename columns
    variables : DataFrameType
        Dataframe with new column names and their corresponding original names

    Returns
    -------
    None
    """
    rename_dict = {}
    for var_name, var_props in variables.iterrows():
        original_name = var_props.get("original_name")
        if original_name is None:
            logger.warning(f"No original name defined for {var_name}. Skipping")
            continue
        if original_name == var_name:
            logger.debug(f"Original name the same as varname: {var_name}. Skipping")
            continue
        if original_name in rename_dict:
            logger.warning(
                f"Original name {original_name} already defined for {rename_dict[original_name]}. Skipping"
            )
            continue
        rename_dict[original_name] = var_name
    dataframe.rename(columns=rename_dict, inplace=True)


def get_records_select(
    dataframe: DataFrameType,
    variables: DataFrameType,
    var_type: str,
    column: str,
    column_list: List[str],
    output_format: str,
    var_filter: Any,
    scaling_suffix: Optional[str] = None,
) -> Tuple[Optional[DataFrameType], Optional[List[str]]]:
    """Get records select

    Retrieves selected records from the dataframe based on a given column and
    variable type, and applies filtering and scaling as necessary.

    Parameters
    ----------
    dataframe : DataFrameType
        The input data frame containing data to be filtered.
    variables : DataFrameType
        A DataFrame containing variable properties.
    var_type : str
        The type of the variable (e.g., 'dict', 'float').
    column : str
        The column to be processed.
    column_list : List[str]
        List of columns to be included in the output.
    output_format : str
        The format of the output.
    var_filter : Any
        Filter applied to the data.
    scaling_suffix : Optional[str], optional
        Suffix used for scaling, by default None.

    Returns
    -------
    selected_records : Optional[DataFrameType]
        DataFrame with selected records, or None if the column is missing.
    updated_column_list : Optional[List[str]]
        Updated list of columns, or None if the column is missing.
    """
    selected_records = None
    ratio_units_key = (
        f"ratio_units_{scaling_suffix}" if scaling_suffix else "ratio_units"
    )

    if column in (ratio_units_key, "units"):
        try:
            selected_records = dataframe.loc[:, column_list]
        except KeyError as err:
            logger.warning(f"{err}\nMissing scaling ratio for columns: {column_list}")

    if selected_records is None:
        try:
            selected_records = get_filtered_data_column(
                dataframe=dataframe,
                column=column,
                var_filter=var_filter,
                output_format=output_format,
            )
        except KeyError:
            logger.warning(f"Missing column: {column}")
            return None, None

    if var_type == "dict" and selected_records is not None:
        df_dummies = pd.get_dummies(selected_records[column], prefix=column)
        renames = {
            col: f"{col}.0" for col in df_dummies.columns if re.search(r"_\d$", col)
        }
        if renames:
            df_dummies.rename(columns=renames, inplace=True)

        try:
            option_keys = variables.loc[column, "options"].keys()
        except AttributeError:
            option_keys = variables.loc[column, "translateopts"].values()

        col_exp = [f"{column}_{float(op):.1f}" for op in option_keys]
        missing_cols = set(col_exp).difference(df_dummies.columns)
        for col in missing_cols:
            df_dummies[col] = 0

        selected_records = selected_records.join(df_dummies).drop(column, axis=1)
        column_list = df_dummies.columns.to_list()

    elif var_type in ("float", "int", "unknown") and column not in (
        "ratio_units",
        "units",
    ):
        df_num = selected_records.astype(float).select_dtypes(include=[np.number])
        non_numeric_cols = set(selected_records.columns) - set(df_num.columns)
        if non_numeric_cols:
            logger.warning(f"Non-numerical columns found: {non_numeric_cols}")
        selected_records = df_num.copy()

    return selected_records, column_list


def prepare_df_for_statistics(
    dataframe, index_names, units_key="units", regional=None, region="nuts0"
) -> DataFrameType:
    """Prepare dataframe for the statistics calculation

    Args:
        dataframe (DataFrameType):
            the data frame with a normal index and columns containing at least the dimensions
        index_names (list): the index names to use for your statistical breakdown
        units_key (str, optional): name of the unity column.
            This column is added to your dataframe.
            Defaults to 'units'
        regional (dict, optional):
            The regional column names to use for your statistical breakdown.
        Defaults to None.
        region (str, options): The name of the region in case we want to select on regional data.
        Defaults to "nuts0",
            which is the whole country as a region

    Returns:
        DataFrameType: The new data frame with statistical breakdown on the index

    Notes:
        * This function modifies your dataframe in orde to set the breakdown on the index.
        * In case a region is passed, also a filter on the required region is applied.

    """
    if regional is None or regional == "nuts0":
        dataframe = dataframe.copy().reset_index()
    else:
        mask = dataframe[regional] == region
        dataframe = dataframe[mask].copy().reset_index()
    # the index which we are going to impose are the group_keys for this statistics
    # output
    # plus always the be_id if that was not yet added to the group_keys
    # make sure to copy group_keys
    # Add the index in tuples to the index names
    mi = [ll for ll in index_names]
    dataframe.set_index(mi, inplace=True, drop=True)
    # The index names now still have the tuples of *mi*.
    # Change that back to the normal names
    dataframe.index.rename(index_names, inplace=True)
    # gooi alle niet valide indices eruit
    dataframe = dataframe.reindex(dataframe.index.dropna())

    dataframe.sort_index(inplace=True)
    # deze toevoegen om straks bij get_statistics het gewicht voor units en wp op
    # dezelfde manier te kunnen doen
    dataframe[units_key] = 1
    return dataframe


def reorganise_stat_df(
    records_stats: DataFrameType,
    variables: DataFrameType,
    variable_key: str,
    use_original_names: bool = False,
    n_digits: Optional[int] = 3,
    sort_index: bool = True,
    module_key: str = "module",
    vraag_key: str = "vraag",
    optie_key: str = "optie",
) -> DataFrameType:
    """
    Reorganise the statistics data frame so that the variables and choice are on the index
    and the records are on the columns.

    Parameters
    ----------
    records_stats : DataFrameType
        The input data frame with statistics
    variables : DataFrameType
        The data frame with the variables
    variable_key : str
        The key of the variable column
    use_original_names : bool
        Use the original name of the variable
    n_digits : int, optional
        The number of digits to round the values to.
        Default to 3.
    sort_index : bool, optional
        Sort the index.
        Defaults to True.
    module_key : str, optional
        The key of the module column.
        Default to "module".
    vraag_key : str, optional
        The key of the vraag column.
        Defaults to "vraag".
    optie_key : str, optional
        The key of the optie column.
        Defaults to "optie".

    Returns
    -------
    DataFrameType
        The reorganised data frame with variables and choice on the index and records on the columns.
    """
    # First, select the columns we need
    columns = [module_key, vraag_key, optie_key, variable_key]
    # Then melt the data frame
    records_stats = records_stats[columns].melt(
        id_vars=[module_key, vraag_key, optie_key],
        value_vars=[variable_key],
        var_name=variable_key,
        value_name="value",
    )
    # Join the variables' data frame on the variable key
    records_stats = records_stats.merge(
        variables, how="left", on=variable_key, validate="m:1"
    )
    # If the boolean *use_original_names* is True, use the original name of the variable
    if use_original_names:
        records_stats[variable_key] = records_stats["original_name"]
    # Drop the original_name column
    records_stats.drop(columns=["original_name"], inplace=True)
    # Group the data frame by the module, vraag and optie keys
    records_stats = records_stats.groupby([module_key, vraag_key, optie_key])
    # Pivot the data frame so that the variables and choice are on the index
    records_stats = records_stats.pivot_table(
        index=[module_key, vraag_key, optie_key],
        columns=variable_key,
        values="value",
        aggfunc="mean",
    )
    # Round the values to n_digits
    records_stats = records_stats.round(n_digits)
    # Sort the index
    if sort_index:
        records_stats.sort_index(inplace=True)
    return records_stats


def get_filtered_data_column(dataframe, column, var_filter=None, output_format=None):
    """
    Retrieve the (filtered) column from the dataframe.

    Parameters
    ----------
    dataframe: DataFrameType
        All data
    column: str
        Name of the column to select
    var_filter: str or dict or None
        Optionally a filter if we want to filter on another column from the data frame.
        If given as a dict, it looks like::

            filter:
                statline: column_name_for_filter_statline_output
                eurostat: column_name_for_filter_euro_output

        If statline or eurostat is not provided, assume all data for that output (unfiltered)

    output_format: str
        Name of the current output format

    Returns
    -------
    DataFrameType:
        Data column to be used for the statistic
    """
    # Select the column data, optionally filtered if a filter variable is given
    if var_filter is None:
        # Without a filter, select all data
        records_selection = dataframe.loc[:, [column]]
    else:
        if isinstance(var_filter, dict):
            # Filter is a dict with different entries for eurostat and statline
            logger.debug(
                f"Trying to get filter variable for {column} with {var_filter}"
            )
            try:
                variable_filter: object = var_filter[output_format]
            except KeyError:
                # If the variable is not provided in the dict, select all data
                variable_filter = None
        else:
            # Filter is given as a key: value, so set it for all outputs
            variable_filter = var_filter

        if variable_filter is None:
            # If variable_filter is None, the dict had no entry for this output, select all data
            logger.debug(
                f"No valid entry for {var_filter} in {column} for {output_format}. Take all data"
            )
            records_selection = dataframe.loc[:, [column]]
        else:
            try:
                mask = dataframe[variable_filter] == 1
            except KeyError:
                # variable_filter is given, but this column does not exist. Warn and skip
                logger.warning(
                    f"KeyError for {column}: {variable_filter}. Provided filter column does not exist!"
                )
                logger.warning(
                    f"KeyError for {column}: {variable_filter}. Opgeven filter column bestaat "
                    f"niet!"
                )
                records_selection = None
            else:
                # We hebben een mask bepaald op basis van het filter. Verkrijg nu de gefilterde data
                records_selection = dataframe.loc[mask, [column]]

    return records_selection

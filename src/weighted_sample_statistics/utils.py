"""
Some utility functions.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

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


def rename_variables(dataframe: pd.DataFrame, variables: pd.DataFrame) -> None:
    """Rename dataframe columns according to the variables DataFrame

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to rename columns
    variables : pd.DataFrame
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
    dataframe,
    variables,
    var_type,
    column,
    column_list,
    output_format,
    var_filter,
    scaling_suffix=None,
):
    """Get records select

    Parameters
    ----------
    scaling_suffix
    dataframe
    variables
    var_type
    column
    column_list
    output_format
    var_filter

    Returns
    -------
    records_selection
    column_list
    """
    ratio_units_key = "ratio_units"
    records_selection = None
    if scaling_suffix is not None:
        ratio_units_key = "_".join([ratio_units_key, scaling_suffix])
        logger.debug(f"Adapted ratio units for scaling suffix to {ratio_units_key}")
    if column in (ratio_units_key, "units"):
        # We willen altijd units in de output. Nu wel expliciet weggooien
        logger.debug(f"creating eurostat specific column: {column}")
        try:
            records_selection = dataframe.loc[:, column_list]
        except KeyError as err:
            logger.warning(f"{err}\nYou are missing scaling ratio {column_list}")

    if records_selection is None:
        try:
            records_selection = get_filtered_data_column(
                dataframe=dataframe,
                column=column,
                var_filter=var_filter,
                output_format=output_format,
            )
        except KeyError as err:
            logger.warning(f"{err}\nYou are missing column {column}")
            return None, None

        if var_type == "dict" and records_selection is not None:
            newcols = None
            dfdummy = pd.get_dummies(records_selection[column], prefix=column)
            renames = dict()
            for col in dfdummy.columns:
                match = re.search(r"_\d$", col)
                if bool(match):
                    col_with_zero = col + ".0"
                    renames[col] = col_with_zero
            if renames:
                dfdummy.rename(columns=renames, inplace=True)
            try:
                optkeys = variables.loc[column, "options"].keys()
            except AttributeError as err:
                logger.info(err)
            else:
                # maak een lijst van name die je verwacht: download_1.0, download_2.0, etc
                try:
                    col_exp = [
                        "_".join([column, "{:.1f}".format(float(op))]) for op in optkeys
                    ]
                except ValueError:
                    optkeys = variables.loc[column, "translateopts"].values()
                    col_exp = [
                        "_".join([column, "{:.1f}".format(float(op))]) for op in optkeys
                    ]
                # Als een category niet voorkomt, dan wordt hij niet aangemaakt. Check
                # wat we missen en vul aan met 0
                missing = set(col_exp).difference(dfdummy.columns)
                try:
                    for col in list(missing):
                        dfdummy.loc[:, col] = 0
                except ValueError:
                    # fails for variance df, but we only need the expected column names only
                    newcols = col_exp

            if newcols is None:
                # new cols are still none, so this succeeded for the normal records_df. Fill in
                newcols = dfdummy.columns.to_list()
            # newcols.append(column)
            var_type = "bool"
            records_selection = records_selection.join(dfdummy)
            records_selection.drop(column, axis=1, inplace=True)
            column_list = list(newcols)
        else:
            column_list = list([column])

    if var_type in ("float", "int", "unknown") and column not in (
        "ratio_units",
        "units",
    ):

        df_num = records_selection.astype(float).select_dtypes(include=[np.number])
        diff = [
            cn
            for cn in records_selection.columns.values
            if cn not in df_num.columns.values
        ]
        if diff:
            logger.warning(
                "Non-numerical columns found in float/int columns:\n" "{}".format(diff)
            )
        # make a real copy of the numerical values to prevent changing the main group
        records_selection = df_num.copy()

    return records_selection, column_list


def prepare_df_for_statistics(
    dataframe, index_names, units_key="units", regional=None, region="nuts0"
) -> DataFrame:
    """Prepare dataframe for the statistics calculation

    Args:
        dataframe (DataFrame): the data frame with a normal index and columns containing at least the dimensions
        index_names (list): the index names to use for your statistical breakdown
        units_key (str, optional): name of the unity column. This column is added to your dataframe. Defaults to 'units'
        regional (dict, optional): the regional column names to use for your statistical breakdown. Defaults to None.
        region (str, options): The name of the region in case we want to select on regional data. Defaults to "nuts0",
            which is the whole country as a region

    Returns:
        DataFrame: The new data frame with statistical breakdown on the index

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
    records_stats: pd.DataFrame,
    variables: pd.DataFrame,
    variable_key: str,
    use_original_names: bool = False,
    n_digits: Optional[int] = 3,
    sort_index: bool = True,
    module_key: str = "module",
    vraag_key: str = "vraag",
    optie_key: str = "optie",
) -> pd.DataFrame:
    """
    Reorganise the statistics data frame so that the variables and choice are on the index
    and the records are on the columns.

    Parameters
    ----------
    records_stats : pd.DataFrame
        The input data frame with statistics
    variables : pd.DataFrame
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
    pd.DataFrame
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
    dataframe: pd.DataFrame
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
    pd.DataFrame:
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
                var_filt = var_filter[output_format]
            except KeyError:
                # If the variable is not provided in the dict, select all data
                var_filt = None
        else:
            # Filter is given as a key: value, so set it for all outputs
            var_filt = var_filter

        if var_filt is None:
            # If var_filt is None, the dict had no entry for this output, select all data
            logger.debug(
                f"No valid entry for {var_filter} in {column} for {output_format}. Take all data"
            )
            records_selection = dataframe.loc[:, [column]]
        else:
            try:
                mask = dataframe[var_filt] == 1
            except KeyError:
                # var_filt is given, but this column does not exist. Warn and skip
                logger.warning(
                    f"KeyError for {column}: {var_filt}. Provided filter column does not exist!"
                )
                # Er is een var_filt gegeven, maar deze kolom bestaat niet. Waarschuw en sla over
                logger.warning(
                    f"KeyError for {column}: {var_filt}. Opgeven filter column bestaat "
                    f"niet!"
                )
                records_selection = None
            else:
                # We hebben een mask bepaald op basis van het filter. Verkrijg nu de gefilterde data
                records_selection = dataframe.loc[mask, [column]]

    return records_selection

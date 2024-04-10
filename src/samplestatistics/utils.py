"""
Some utility functions.
"""
import logging
import re

logger = logging.getLogger(__name__)


def make_negation_name(column_name, suffix="_x") -> str:
    """Make a new column name based for the negative value

    Returns
    -------
    new_col : str
    """
    match = re.search("(_\d\.\d)$", column_name)
    if match:
        # column ends with _1.0, make the new name like _x_1.0
        new_col = "".join([re.sub("_\d\.\d$", "", column_name), suffix]) + match.group(
            1
        )
    else:
        new_col = "".join([column_name, suffix])
    return new_col


def rename_all_variables(dataframe, variables) -> None:
    """Rename all the columns of data as defined int the *variable* dictionary

    Parameters
    ----------
    dataframe: Dataframe
        Dataframe for which we want to rename the columns
    variables: DataFrame
        Dataframe of the variables with the index the name of the new variable

    Notes
    -----
    * This function checks if a variable is defined multiple time and raises a warning in that case
    * The dataframe is changed in place. Nothing is returned

    Returns
    -------
    None
    """
    new_names = dict()
    for var_name, var_props in variables.iterrows():
        try:
            original_name = var_props["original_name"]
        except KeyError:
            logger.warning(f"No original name defined for {var_name}. Skipping")
            # if original name is not defined just continue.
            continue
        # in case the name of the new variable is the same as the original, skip it
        if original_name == var_name:
            logger.debug(f"Original name the same as varname: {var_name}. Skipping")
            continue
        if original_name is None:
            logger.debug(f"Original name not defined for varname: {var_name}. Skipping")
            continue

        logger.debug(f"Renaming {original_name} -> {var_name}")

        # in the csv file, all dots are replaced with _. In DVZ not. Therefore sync here to _
        original_name_dash = original_name.replace(".", "_")

        if original_name_dash != original_name:
            logger.debug(f"Original name has dashed equivalent: {original_name_dash}")

        # now check if this column name was not already set before by another variable
        if original_name in new_names.keys():
            other = new_names[original_name]
        elif original_name_dash in new_names.keys():
            other = new_names[original_name_dash]
        else:
            other = None

        if other is not None:
            logger.warning(
                "Variable {} corresponds to original {} which was already defined by {}"
                "".format(var_name, original_name, other)
            )
            logger.warning("Assuming that we need to duplicate this columns")
            original_name_new = "_".join([original_name, var_name])
            dataframe[original_name_new] = dataframe[original_name]
            original_name = original_name_new

        # nope, it was not set before. Now also check if the original name as defined by
        # the variable is available in our data frame. If not, raise a warning
        if original_name in dataframe.columns:
            logger.debug("Rename variable {} -> {}".format(original_name, var_name))
            new_names[original_name] = var_name
        elif original_name_dash in dataframe.columns:
            logger.debug(
                "Rename variable {} -> {}".format(original_name_dash, var_name)
            )
            new_names[original_name_dash] = var_name
        else:
            logger.debug(
                "Variable {} as defined in {}  is not available"
                "".format(original_name, var_name)
            )
    # now rename all the variables
    dataframe.rename(columns=new_names, inplace=True)
    logger.debug("Done renaming")


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
        # Verkrijg de data van 'column' uit de dataframe. Pas eventueel een filter toe
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
            # In python 3.8 worden er geen nullen meer geplakt 0.0, 1.0, etc, maar 0, 1.
            # zorg dat er weer .0 achter geplakt wordt
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
                # new cols is still none, so this succeeded for the normal records_df. Fill in
                newcols = list(dfdummy.columns.values)
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

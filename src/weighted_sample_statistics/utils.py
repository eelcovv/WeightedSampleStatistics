"""
Some utility functions.
"""

import logging
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

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


def prepare_df_for_statistics(
    dataframe, index_names, units_key, regional=None, region=None
) -> DataFrame:
    """Prepare dataframe for statistics

    Args:
        dataframe (DataFrame): the data frame to reorganise
        index_names (list): the index names
        units_key (str): name of the units column
        regional (dict): the regional column
        region (str): the name of the region column

    Returns:
        dataframe: DataFrame
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
    # Therefore add the index in tuples to the index names
    mi = [ll for ll in index_names]
    dataframe.set_index(mi, inplace=True, drop=True)
    # the index names now still have the tuples of mi. Change that back to the normal
    # names
    dataframe.index.rename(index_names, inplace=True)
    # gooi alle niet valide indices eruit
    dataframe = dataframe.reindex(dataframe.index.dropna())

    dataframe.sort_index(inplace=True)
    # deze toevoegen om straks bij get_statistics het gewicht voor units en wp op
    # dezelfde manier te kunnen doen
    dataframe[units_key] = 1
    return dataframe


def reorganise_stat_df(
    records_stats,
    variables,
    variable_key,
    use_original_names=False,
    n_digits=3,
    sort_index=True,
    module_key="module",
    vraag_key="vraag",
    optie_key="optie",
):
    """
    We have a statistics data frame, but not yet all the information of the variables

    Parameters
    ----------
    use_original_names: bool
        Use the original name of the variable
    """

    logger.debug("Reorganising stat")
    # at the beginning, sbi and gk as multindex index, variabel/choice as mi-columns
    try:
        mask = records_stats.index.get_level_values(1) != ""
    except IndexError:
        mask = records_stats.index.get_level_values(0) != ""
    stat_df = records_stats.loc[mask].copy()
    # with unstack, the gk is put as an extra level to the columns. sbi is now a normal  index
    if len(stat_df.index.names) > 1:
        stat_df = stat_df.unstack()

    # transposing and resetting index puts the variables and choice + gk at the index,
    # reset index create columns out of them. The sbi codes are now at the columns
    temp_df = stat_df.transpose()
    stat_df = temp_df.reset_index()
    # stat_df = stat_df.T.reset_index()
    # onderstaande bij 1 index
    stat_df.rename(columns={"index": "variable"}, inplace=True)
    # onderstaand alleen bij unstack
    stat_df.rename(columns={"level_0": "variable"}, inplace=True)

    try:
        stat_df.drop([""], inplace=True, axis=1)
    except KeyError:
        pass

    # add new columns with the module type so we can reorganise the questions into modules
    stat_df[module_key] = "Unknown"
    stat_df["module_include"] = True
    stat_df[optie_key] = ""
    stat_df[vraag_key] = ""
    stat_df["check"] = False
    stat_df["od_key"] = None

    for var_name in stat_df[variable_key].unique():
        logger.debug("var varname {}".format(var_name))
        # copy the module key from the variables to the statistics data frame
        # get the mask to identify all variable in the stat_df equal to the current var_name
        # note that var_name in stat_df is not unique as each varname occurs multiple time
        # for other gk code, sbi groups etc. However, it is unique in the variabel dataframe

        mask = stat_df[variable_key] == var_name

        # tijdelijke oplossing categorien
        # import re
        # we hebben de naam nodig zonder nummertje erachter. De naam is nodig om gegevens
        # uit de yaml file te halen\
        match = re.search("_(\d)[\.0]*$", var_name)
        if bool(match):
            choice = int(match.group(1))
            var_name_clean = re.sub("_\d[\.0]*$", "", var_name)
        else:
            choice = None
            var_name_clean = var_name
        # var_name_clean = re.sub("\_x$", "", var_name_clean)
        try:
            module_key_key = variables.loc[var_name_clean, module_key]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, module_key] = module_key_key

        try:
            module_label = variables.loc[var_name_clean, "module_label"]
        except KeyError:
            pass
        else:
            if module_label is not None:
                stat_df.loc[mask, module_key] = module_label

        if use_original_names:
            try:
                label = variables.loc[var_name_clean, "original_name"]
            except KeyError:
                label = var_name
            else:
                if label in ("", None):
                    label = var_name
        else:
            try:
                label = variables.loc[var_name_clean, "label"]
            except KeyError:
                label = var_name
            else:
                if label in ("", None):
                    label = var_name

        stat_df.loc[mask, "vraag"] = label

        try:
            module_include = variables.loc[var_name_clean, "module_include"]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, "module_include"] = module_include

        try:
            check_vraag = variables.loc[var_name_clean, "check"]
        except KeyError:
            pass
        else:
            stat_df.loc[mask, "check"] = check_vraag

        try:
            options_dict = variables.loc[var_name_clean, "options"]
        except KeyError:
            pass
        else:
            if options_dict is not None and choice is not None:
                try:
                    option_label = options_dict[choice]
                except KeyError:
                    logger.warning(f"Invalid option {choice} for {var_name}")
                else:
                    stat_df.loc[mask, "optie"] = option_label

    # select only the module with the include flag to true
    stat_df = stat_df[stat_df["module_include"]]
    stat_df = stat_df[stat_df[module_key] != "Unknown"]
    stat_df.drop(["module_include", "check", "od_key"], axis=1, inplace=True)

    if n_digits is not None:
        stat_df = stat_df.round(decimals=n_digits)

    index_variables = [module_key, vraag_key, optie_key]
    if sort_index:
        stat_df.sort_values([module_key, vraag_key, variable_key], axis=0, inplace=True)
    stat_df.set_index(index_variables, inplace=True, drop=True)
    return stat_df


def get_filtered_data_column(dataframe, column, var_filter=None, output_format=None):
    """
    Verkrijg de (gefilterde) kolom uit de dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Alle data
    column: str
        Naam van de kolom die we willen selecteren
    var_filter: str of dict or None
        Eventueel een filter als we op een andere kolom uit data frame willen filteren.
        Als een dict gegeven is ziet het er zou uit::

            filter:
                statline: kolomnaam_voor_filter_voor_statline_output
                eurostat: kolomnaam_voor_filter_voor_euro_output

        Als statline of eurostat niet gegeven is aan nemen voor die output alle data (ongefilterd)

    output_format: str
        Naam van de huidige output format

    Returns
    -------
    pd.DataFrame:
        Datakolom die we willen gebruiken voor de statistiek

    """
    # hier wordt de colom data geselecteerd, eventueel gefiltered als een filter variable
    # gegeven is.
    if var_filter is None:
        # zonder filter nemen we alle data
        records_selection = dataframe.loc[:, [column]]
    else:

        if isinstance(var_filter, dict):
            # er is een filter in de vorm van een dict, dus een verschillende entry voor
            # eurostat en statline (keys zijn eurostat of statline). Probeer de filter variable
            # te krijgen
            logger.debug(
                f"Trying to get filter variabele for {column} with {var_filter}"
            )
            try:
                var_filt = var_filter[output_format]
            except KeyError as err:
                # Als de variable niet gegevens is in de dict nemen we alsnog alle data
                var_filt = None
        else:
            # het filter is als een key: value gegeven, dus zet het filter voor alle ouputs
            var_filt = var_filter

        if var_filt is None:
            # als var_filt hier None was hadden we een dict zonder entry voor deze output.
            # Dan nemen we dus ale data
            logger.debug(
                f"No valid entry for {var_filter} in {column} for {output_format}. "
                f"Take all data"
            )
            records_selection = dataframe.loc[:, [column]]
        else:
            try:
                mask = dataframe[var_filt] == 1
            except KeyError:
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

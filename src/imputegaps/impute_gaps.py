import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ImputeGaps:
    """
    add documentation here
    """

    def __init__(
        self,
        records_df=None,
        variables=None,
        impute_settings=None,
        sbi_key=None,
        gk6_label=None,
        be_id=None,
    ):

        self.records_df = records_df
        self.variables = variables
        self.impute_settings = impute_settings
        self.sbi_key = sbi_key
        self.gk6_label = gk6_label
        self.be_id = be_id

        self.impute_gaps()

    # TODO: Kopiëren uit ICT analyser?
    def create_multi_index(dataframe, index_label, mi_labels) -> pd.DataFrame:
        """
        Create a multindex for the dataframe based on the mi_labels

        Parameters
        ----------
        dataframe: Dataframe
            Dataframe for which the multiindex should be amde
        index_label: str
            label of the current index
        mi_labels: list
            List of strings with the new multi index labels

        """

        new_df = dataframe.reset_index().copy()
        new_df = new_df.rename(columns={"index": index_label})
        new_df = new_df.set_index(mi_labels, drop=True)
        new_df = new_df.sort_index(axis=0)
        new_df.index = new_df.index.rename(mi_labels)
        return new_df.copy()

    def fill_missing_data(self, df, how) -> pd.DataFrame:
        """The nan's  in the data frame are filled with random picks from the valid data in a column

        Parameters
        ----------
        dataframe: DataFrame
            Dataframe with the data to correct
        how: {"mean", "median", "pick"}
            How to fill the gaps.jjj

        Returns
        -------
        ds: pd.DataFrame
            Filled data frame
        """
        ds = df.copy()

        # create a mask with the size of a column with True at all nan locations
        mask = ds.isnull()

        if not mask.any():
            return ds

        if how == "mean":
            samples = np.full(mask.size, fill_value=ds.mean())
        elif how == "median":
            samples = np.full(mask.size, fill_value=ds.median())
        elif how == "nan":
            samples = np.full(ds.isnull().sum(), fill_value=0)
        elif how == "pick1":
            samples = np.full(ds.isnull().sum(), fill_value=1)
            try:
                if samples.size > 1:
                    ds[mask] = samples
                else:
                    ds.loc[mask] = samples
                return ds
            except TypeError:
                samples = np.full(ds.isnull().sum(), fill_value="1.0")
                if samples.size > 1:
                    ds[mask] = samples
                else:
                    ds.loc[mask] = samples
                return ds
        elif how == "pick":
            number_of_nans = mask.sum()
            valid_values = ds[~mask].values
            if valid_values.size == 0:
                # return ds met dus nog een lege, maar in de volgende stap deze zelfde ds weer proberen
                return ds
            samples = np.random.choice(valid_values, size=number_of_nans, replace=True)
        else:
            raise ValueError(
                "Not a valid choice for how {}. Should be mean, median or pick"
                "".format(how)
            )

        # copy the samples to the nan values in the column
        if samples.size > 1:
            ds[mask] = samples
        else:
            ds.loc[mask] = samples
        return ds

    def impute_gaps(self) -> None:

        self.records_df = self.records_df.reset_index().copy()

        # TODO: Grouping generieker maken
        self.records_df.set_index(
            [self.sbi_key, self.gk6_label, self.be_id], inplace=True
        )

        # Itereer over kolommen (variabelen)
        for col_name in self.records_df.columns:

            # Check of er informatie is over de variabele
            try:
                var_type = self.variables[col_name]["type"]
            except KeyError as err:
                logger.info(f"Geen variabele info voor: {col_name}, {err}")
                continue

            # Check of variabele op 'no_impute' staat
            skip_impute = self.variables[col_name]["no_impute"]
            if skip_impute:
                logger.info(
                    "Skipping imputing variable {} of var type {}".format(
                        col_name, var_type
                    )
                )
                continue

            # Check of variabele van een type is dat geskipt moet worden
            if var_type in self.impute_settings["skip"]:
                logger.debug(
                    "Skipping imputing variable {} of var type {}".format(
                        col_name, var_type
                    )
                )
                continue

            logger.info("Impute gaps {:20s} ({})".format(col_name, var_type))
            logger.debug("Imputing variable {}".format(col_name))

            # Variabele om te imputeren
            df = self.records_df[col_name]

            # For the dictionaries, first convert the columns to corresponding categories.
            if var_type in self.impute_settings["pick1"]:
                how = "pick1"
            elif var_type in self.impute_settings["nan"]:
                how = "nan"
            else:
                how = "mean"
            logger.debug(f"Fill gaps by taking the {how} of the valid values")

            def fill_gaps(x):
                return self.fill_missing_data(x, how=how)

            var_filter = self.variables[col_name]["filter"]
            # Indien gevuld selecteer dan alleen data achter filter
            if var_filter is not np.nan:
                try:
                    filter_mask = self.records_df[var_filter]
                except KeyError as err:
                    logger.warning(f"Failed to filter with {var_filter}, {err}")
                else:
                    df = df.loc[filter_mask == 1]

            n_nans1 = df.isnull().sum()

            if n_nans1 == 0:
                logger.debug(
                    "Skip imputing {}. It has no invalid numbers".format(col_name)
                )
                continue

            frc = n_nans1 / df.size
            logger.debug(
                "Filling {} with {} / {} nans ({:.1f} %)"
                "".format(col_name, n_nans1, df.size, frc * 100)
            )

            # TODO: Iets met deze keys
            grouped_sbi_gk = df.groupby(
                [self.sbi_key, self.gk6_label], group_keys=False
            )

            try:
                result_sbi_gk = grouped_sbi_gk.apply(fill_gaps)
            except (TypeError, ValueError):
                logger.warning(
                    "Failed imputing gaps for {} with apply. Try again per group"
                    "".format(col_name)
                )
                result_sbi_gk = None

            if result_sbi_gk is None or (
                result_sbi_gk is not None and result_sbi_gk.isnull().sum() > 0
            ):
                # failed to group. Try to group by size class only
                logger.debug(
                    "Could not fill completely based only on sbi/gk. "
                    "Fill the remaining gaps with gk6 means"
                )
                grouped_gk = df.groupby([self.gk6_label], group_keys=False)

                # om te voorkomen dat we de volledige imputatie van een breakdown op basis van sbi
                # doen, ook als er maar 1 NA is, is het beter om alleen de NA met een grovere
                # imputatie te vullen
                try:
                    result_gk = grouped_gk.apply(fill_gaps)
                # TODO: Uitzoeken waarom dit kan mislukken
                except (TypeError, ValueError):
                    logger.warning(
                        "Also failed imputing gaps by size class for {}".format(
                            col_name
                        )
                    )
                else:
                    # voor alle posities die we niet met de sbi/gk mean hebben kunnen vullen, vullen
                    # we nu met de gk6
                    if result_sbi_gk is None:
                        result_sbi_gk = result_gk
                    else:
                        # vul alleen de waardes van de grovere breakdown waar we eerde na hadden
                        mask = result_sbi_gk.isnull()
                        result_sbi_gk[mask] = result_gk[mask]

            self.records_df[col_name] = result_sbi_gk

            if result_sbi_gk is not None:
                n_nans2 = result_sbi_gk.isnull().sum()
                logger.debug(
                    "Variable {}: filled {} -> {}".format(col_name, n_nans1, n_nans2)
                )
            else:
                logger.warning(f"Failed filling for {col_name}")

        logger.info("create multiindex")
        self.records_df = self.create_multi_index(
            self.records_df, self.be_id, self.mi_labels
        )
        logger.info("Done")

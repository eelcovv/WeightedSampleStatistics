"""
Definition of SampleStatistics class to calculate weighted statistics
"""
import logging
import re
from typing import Union

import numpy as np
from pandas import DataFrame

from .utils import make_negation_name

DataFrameType = Union[DataFrame, None]

logger = logging.getLogger(__name__)


class SampleStatistics:
    """
    Calculate statistics for summations

    Parameters
    ----------
    group_keys: iterable
        The variables to use to group
    records_df_selection: DataFrame
        All the microdata including non-response
    weights_df: DataFrame
        The weights per unit
    all_records_df: DataFrame
        All the microdata including non-response
    column_list: iterable
        list of columns to calculate statistics
    scaling_factor_key: str
        Name of the weight variable
    var_type: str
        Type of the data
    add_inverse: bool
        Add the negated value as well for booleans
    report_numbers: bool
        Do not calculate the average, but the sum

    Attributes
    ----------
    records_sum: grouped
        The summation of the weighted values
    weights_sum: grouped
        The summation of the weights
    records_weighted_mean: grouped
        The sample mean estimate
    records_std grouped
        The sample standard deviation estimate
    number_samples_sqrt: grouped
        The square root of the sample size n
    standard_error: grouped
        The standard error of the mean estimate: std / n_sqrt
    """

    def __init__(
        self,
        group_keys,
        records_df_selection,
        weights_df,
        column_list=None,
        var_type=None,
        scaling_factor_key=None,
        units_scaling_factor_key=None,
        all_records_df=None,
        var_weight_key=None,
        variance_df_selection=None,
        records_df_unfilled=None,
        add_inverse=False,
        report_numbers=False,
        negation_suffix=None,
    ):
        self.group_keys = group_keys
        self.records_df_selection = records_df_selection
        self.records_norm_sel_df = None
        self.records_norm_pop_df = None
        self.variance_df_selection = variance_df_selection
        self.records_df_unfilled = records_df_unfilled
        self.records_df_selection_sqr = None
        self.records_std_df = None
        self.records_std_agg = None
        self.records_var_df = None
        self.records_var_agg = None
        self.weights_df = weights_df
        self.all_records_df = all_records_df
        self.column_list = column_list
        self.var_type = var_type
        self.var_weight_key = var_weight_key
        self.scaling_factor_key = scaling_factor_key
        self.units_scaling_factor_key = units_scaling_factor_key
        self.add_inverse = add_inverse
        self.report_numbers = report_numbers
        if negation_suffix is None:
            self.negation_suffix = "_x"
        else:
            self.negation_suffix = negation_suffix

        self.proportion_weighted_sel_df = None
        self.proportion_weighted_pop_df = None

        self.scale_variabele_pop_df = None
        self.scale_variabele_sel_df = None
        self.var_weight_pop_df = None
        self.var_weight_sel_df = None
        self.scale_variabele_pop_grp = None
        self.scale_variabele_sel_grp = None
        self.unweighted_means_grp = None
        self.var_weight_pop_grp = None
        self.weights_sel_grp = None
        self.weights_sel_sum_df = None
        self.weights_pop_sum_df = None
        self.var_weight_sel_grp = None
        self.records_df_valid = None
        self.var_weights_sel_sum_df = None
        self.var_weights_pop_sum_df = None
        self.var_weights_sel_sum_agg = None
        self.var_weights_pop_sum_agg = None
        self.records_sum = None
        self.response_count = None
        self.sample_count_initial = None
        self.valid_mask = None
        self.weights_sum = None
        self.mean_weight = None
        self.records_weighted_mean = None
        self.records_weighted_vars = None
        self.records_std = None
        self.records_var = None
        self.n_sample = None
        self.number_samples_sqrt = None
        self.unweighted_mean = None
        self.shifted_mean_df = None
        self.shifted_mean_grp = None
        self.standard_error = None
        self.response_proportion = None
        self.response_fraction: DataFrameType = None
        self.variances_df = None
        self.variances_grp = None
        self.records_weighted_sel_grp = None
        self.records_weighted_pop_grp = None
        self.records_weighted_sel_mean_df = None
        self.number_ratio = None
        self.records_weighted_pop_mean_df = None
        self.records_weighted_mean_agg = None
        self.records_weighted_conditional_mean_agg = None
        self.records_weighted_sel_df = None
        self.records_weighted_pop_df = None
        self.proportion_sel_df = None
        self.proportion_pop_df = None
        self.proportion_sel_grp = None
        self.proportion_pop_grp = None
        self.proportion_sel_mean_agg = None
        self.proportion_pop_mean_agg = None
        self.proportion_sel_mean_df = None
        self.proportion_pop_mean_df = None

        self.records_sel_grp = None
        self.variance_sel_grp = None
        self.records_valid_grp = None
        self.records_sel_sqr_grp = None
        self.all_records_grp = None
        self.weights_grp = None
        self.weights = None
        self.unit_weights_pop_df = None
        self.weights_sel_df = None
        self.unit_weights_sel_df = None
        self.unit_weights_sel_grp = None
        self.unit_weights_pop_grp = None
        self.unit_weights_sel_sum_df = None
        self.unit_weights_pop_sum_df = None
        self.records_weighted_mean_agg = None
        self.weights_sel_sum_agg = None
        self.weights_pop_sum_agg = None
        self.weights_sel_normalized_df = None
        self.weights_pop_normalized_df = None
        self.unit_weights_sel_sum_agg = None
        self.unit_weights_pop_sum_agg = None

        self.set_mask_valid_df()

        self.scale_variables()

        self.group_variables()

        self.calculate_weighted_means()
        if self.all_records_df is not None:
            self.calculate_response_fraction()
        self.calculate_proportions()
        self.calculate_standard_errors()

    def scale_variables(self):
        """scaling of variables"""

        logger.debug(f"Scaling variables with {self.scaling_factor_key}")
        self.weights = self.weights_df.loc[:, self.scaling_factor_key]
        self.unit_weights_pop_df = self.weights_df.loc[:, self.units_scaling_factor_key]
        fixed = set(list([self.var_weight_key, self.scaling_factor_key]))
        if set(self.column_list).intersection(fixed):
            # in case the variable to process (stored in the column list) is a scaling variable,
            # we do not scale it, so set the weights to 1
            self.weights.values[:] = 1.0

        self.weights_sel_df = self.weights.reindex(self.records_df_selection.index)
        self.unit_weights_sel_df = self.unit_weights_pop_df.reindex(
            self.records_df_selection.index
        )

        self.scale_variabele_pop_df = self.weights_df[self.var_weight_key]
        self.scale_variabele_sel_df = self.scale_variabele_pop_df.reindex(
            self.records_df_selection.index
        )

        # Nu we de teller hebben vermenigvuldigd moeten we ook de noemer doen (bij units is dit
        # 1 (maakt dus niet uit. Maar bij wp is dit wel een ander getal)
        # laat weight intact. In bereken je de populatie variable van het gewicht

        self.var_weight_pop_df = (
            self.weights_df[self.scaling_factor_key] * self.scale_variabele_pop_df
        )
        self.var_weight_sel_df = self.var_weight_pop_df.reindex(
            self.records_df_selection.index
        )

    def set_mask_valid_df(self):
        """Set mask valid df

        Returns
        -------
        None
        """
        if self.records_df_unfilled is not None:
            self.records_df_valid = ~self.records_df_unfilled.isna()
            try:
                self.records_df_valid = self.records_df_valid[self.column_list]
            except KeyError:
                col = re.sub(r"_\d\.\d", "", self.column_list[0])
                try:
                    self.records_df_valid = self.records_df_valid[col]
                except KeyError:
                    self.records_df_valid = None

    def group_variables(self):
        """make the groups for the variables"""
        logger.debug(f"Grouping variables with {self.group_keys}")
        self.records_sel_grp = self.records_df_selection.groupby(self.group_keys)
        if self.variance_df_selection is not None:
            self.variance_sel_grp = self.variance_df_selection.groupby(self.group_keys)

        if self.records_df_valid is not None:
            self.records_valid_grp = self.records_df_valid.groupby(self.group_keys)

        if self.all_records_df is not None:
            if self.var_weight_key not in self.all_records_df.columns:
                self.all_records_df[self.var_weight_key] = 1
            self.all_records_grp = self.all_records_df[self.var_weight_key].groupby(
                self.group_keys
            )
        self.weights_grp = self.weights.groupby(self.group_keys)
        self.weights_sel_grp = self.weights_sel_df.groupby(self.group_keys)

        self.unit_weights_sel_grp = self.unit_weights_sel_df.groupby(self.group_keys)
        self.unit_weights_pop_grp = self.unit_weights_pop_df.groupby(self.group_keys)

        self.var_weight_pop_grp = self.var_weight_pop_df.groupby(self.group_keys)
        self.var_weight_sel_grp = self.var_weight_sel_df.groupby(self.group_keys)
        self.scale_variabele_pop_grp = self.scale_variabele_pop_df.groupby(
            self.group_keys
        )
        self.scale_variabele_sel_grp = self.scale_variabele_sel_df.groupby(
            self.group_keys
        )

    def calculate_weighted_means(self):
        """Calculate summed statistics

        Returns
        -------
        None
        """
        logger.debug(f"Start calculation summed statistics for {self.column_list}")
        if "omzet_enq" in self.column_list:
            logger.debug("Stop hier")

        # for the rest: calculate the sums
        self.weights_sel_sum_df = self.weights_sel_grp.transform("sum")
        self.weights_pop_sum_df = self.weights_grp.transform("sum")
        self.weights_sel_sum_agg = self.weights_sel_grp.sum()
        self.weights_pop_sum_agg = self.weights_grp.sum()
        self.unit_weights_sel_sum_df = self.unit_weights_sel_grp.transform("sum")
        self.unit_weights_pop_sum_df = self.unit_weights_pop_grp.transform("sum")
        self.var_weights_sel_sum_df = self.var_weight_sel_grp.transform("sum")
        self.var_weights_pop_sum_df = self.var_weight_pop_grp.transform("sum")
        self.var_weights_sel_sum_agg = self.var_weight_sel_grp.sum()
        self.var_weights_pop_sum_agg = self.var_weight_pop_grp.sum()
        self.unit_weights_sel_sum_agg = self.unit_weights_sel_grp.sum()
        self.unit_weights_pop_sum_agg = self.unit_weights_pop_grp.sum()
        # deze regels veroorzaken een warning van multi.py 3587: the values are unorderable
        # is nu verholpen door het weghalen van de lege indices
        logger.debug(f"normalizing weights with sums in selection")
        self.weights_sel_normalized_df = self.weights.div(
            self.weights_sel_sum_df, axis="index"
        )
        logger.debug(f"normalizing weights with sums in population")
        self.weights_pop_normalized_df = self.weights.div(
            self.weights_pop_sum_df, axis="index"
        )

        logger.debug(f"applying weights to records")
        self.records_weighted_sel_df = self.records_df_selection.mul(
            self.weights_sel_normalized_df, axis="index"
        )
        self.records_weighted_pop_df = self.records_df_selection.mul(
            self.weights_pop_normalized_df, axis="index"
        )
        self.records_weighted_sel_grp = self.records_weighted_sel_df.groupby(
            self.group_keys
        )
        self.records_weighted_pop_grp = self.records_weighted_pop_df.groupby(
            self.group_keys
        )
        self.records_weighted_sel_mean_df = self.records_weighted_sel_grp.transform(
            "sum"
        )
        self.records_weighted_pop_mean_df = self.records_weighted_pop_grp.transform(
            "sum"
        )

        logger.debug(f"calculating weighted means")
        self.records_weighted_mean_agg = self.records_weighted_pop_grp.sum()
        self.records_weighted_conditional_mean_agg = self.records_weighted_sel_grp.sum()

        logger.debug(f"calculating conditional weighted means")
        self.records_sum = self.records_weighted_conditional_mean_agg.mul(
            self.weights_sel_sum_agg, axis="index"
        )

        # If your selection is empty because no company within the group meets the filter condition
        # is sufficient, then the sum is set to nan.
        # Now make that 0 again.
        logger.debug(f"Fill nan's with 0")
        self.records_sum = self.records_sum.astype(float).fillna(0)

        if self.add_inverse:
            for col_name in self.records_sum:
                new_col = make_negation_name(col_name, self.negation_suffix)
                logger.debug(f"Creating new negated column {new_col}")
                # Determine the sum of the population.
                # For an empty population, the index of the records_sum is interpolated on the population sum,
                # so that we have both items and set the nan to 0.
                # This also ensures that the negated variable always exists, and 0 is if the population is zero
                filter_sum = self.var_weights_sel_sum_agg.reindex(
                    self.records_sum.index
                ).fillna(0)
                self.records_sum[new_col] = filter_sum - self.records_sum[col_name]

        self.response_count = self.weights_grp.count()

        if self.var_type in ("bool", "dict"):
            # naar percentage omrekenen
            self.records_weighted_mean_agg *= 100
            self.records_weighted_conditional_mean_agg *= 100

    def calculate_response_fraction(self):
        """Calculate response fraction"""
        logger.debug("Calculating the response fractions")
        self.sample_count_initial = self.all_records_grp.count()
        if self.records_valid_grp is not None:
            valid_vals = self.records_valid_grp.sum()
        else:
            valid_vals = self.response_count
        response_series = 100 * valid_vals.div(self.sample_count_initial, axis="index")

        # turn the response series into a dataframe with the same number of columns and column
        # names as the mean
        for col_name in self.column_list:
            try:
                response_df: DataFrameType = response_series.to_frame(name=col_name)
            except AttributeError:
                logger.debug("Failed to transfer to frame. It is a frame already")
                response_df = response_series
            if self.response_fraction is None:
                self.response_fraction = response_df
            else:
                self.response_fraction = self.response_fraction.join(response_df)

    def calculate_proportions(self):
        """Calculate proportions"""
        logger.debug("Calculating the proportions")

        # normaliseer de records variable met de schaalfactor
        self.records_norm_pop_df = self.records_df_selection.div(
            self.scale_variabele_pop_df, axis="index"
        )
        self.records_norm_pop_df.clip(lower=0, upper=1, inplace=True)
        self.records_norm_sel_df = self.records_norm_pop_df.reindex(
            self.records_df_selection.index
        )

        sum_unit_weight = self.unit_weights_sel_sum_df
        sum_weight = self.var_weights_sel_sum_df
        self.number_ratio = sum_unit_weight.div(sum_weight, axis="index")

        self.proportion_sel_df = 100 * self.records_norm_sel_df
        self.proportion_pop_df = 100 * self.records_norm_pop_df

        self.proportion_weighted_sel_df = self.proportion_sel_df.mul(
            self.weights_sel_normalized_df, axis="index"
        )

        self.proportion_weighted_pop_df = self.proportion_pop_df.mul(
            self.weights_pop_normalized_df, axis="index"
        )

        self.proportion_sel_grp = self.proportion_weighted_sel_df.groupby(
            self.group_keys
        )
        self.proportion_pop_grp = self.proportion_weighted_pop_df.groupby(
            self.group_keys
        )

        # You can show that this proportion (calculated from the average of the fractions between
        # 0 and 100 %) is mathematically different from the sum of the elements divided by the sum of the
        # total.
        # To keep the output consistent, we simply print the last one.
        # But we still use the fraction to calculate the standard error
        self.proportion_sel_mean_agg = 100 * self.records_sum.div(
            self.var_weights_sel_sum_agg, axis="index"
        )
        self.proportion_pop_mean_agg = 100 * self.records_sum.div(
            self.var_weights_pop_sum_agg, axis="index"
        )

    def calculate_standard_errors(self):
        """Calculate standard errors"""
        logger.debug("Calculating the standard errors")

        if self.variance_df_selection is None:
            # for the first round (on the smallest strata, calculate the standard deviation based
            # on the microdata sum_i (w_i * (x_i - x_mean)**2)
            # where w_i are the normalized weight for which by definition: sum_i w_i = 1
            mean_proportion = self.proportion_pop_grp.transform("sum")
            proportion_minus_mean = self.proportion_pop_df - mean_proportion
            proportion_squared = np.square(proportion_minus_mean)
            proportion_squared_sel = proportion_squared.reindex(
                self.records_df_selection.index
            )
            records_var = proportion_squared_sel.mul(
                self.weights_pop_normalized_df, axis="index"
            )
        else:
            # for the compound breakdowns, us the variances from the first round and multiply
            # with w_i**2
            weights_sel_normalized_df_squared = np.square(
                self.weights_sel_normalized_df
            )
            records_square = self.variance_df_selection
            records_var = records_square.mul(
                weights_sel_normalized_df_squared, axis="index"
            )

        records_var_grp = records_var.groupby(self.group_keys)
        self.records_var_df = records_var_grp.transform("sum")
        self.records_var_agg = records_var_grp.sum()
        self.records_std_df = self.records_var_df.pow(0.5)
        self.records_std_agg = self.records_var_agg.pow(0.5)

        if self.variance_df_selection is None:
            # for the first round when we calculated the standard dev with the sum (x_i - xm)**2
            # you have to divide  by sqrt(n)  and multiply with the fpc
            self.n_sample = self.records_sel_grp.count()
            self.number_samples_sqrt = np.sqrt(self.n_sample)

            ratio = self.n_sample.div(self.weights_sel_sum_agg, axis="index")
            ratio[ratio > 1] = 1
            fpc = np.sqrt(1 - ratio)
            self.standard_error = self.records_std_agg.div(
                self.number_samples_sqrt, axis="index"
            )
            self.standard_error = self.standard_error.mul(fpc, axis="index")
        else:
            # We got the standard error from the compound standard deviations.
            # No need to divide by sqrt(n), as we used the w_i**2 terms already
            self.standard_error = self.records_std_agg

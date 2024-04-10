from weighted_sample_statistics import WeightedSampleStatistics
from weighted_sample_statistics import reorganise_stat_df

__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"


def test_class_import():
    weighted_sample_statistics = WeightedSampleStatistics(
        group_keys=None, records_df_selection=None, weights_df=None
    )
    assert weighted_sample_statistics.group_keys is None


def test_reorganise_stat_df():
    function = reorganise_stat_df
    assert function is not None

from weighted_sample_statistics.core import Weightedweighted_sample_statistics

__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    weighted_sample_statistics = Weightedweighted_sample_statistics(
        group_keys=None, records_df_selection=None, weights_df=None
    )
    assert weighted_sample_statistics.group_keys is None

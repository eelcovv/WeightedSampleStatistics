import pytest
from weighted_sample_statistics import WeightedSampleStatistics

from weighted_sample_statistics.main import main

__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    weighted_sample_statistics = WeightedSampleStatistics()

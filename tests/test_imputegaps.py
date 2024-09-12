import pytest
from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"


def test_failed_imputegaps():
    """API Tests"""
    with pytest.raises(AttributeError):
        test_obj = ImputeGaps()





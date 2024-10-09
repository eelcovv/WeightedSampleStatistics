import pytest
import pandas as pd
from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

AVAILABLE_IMPUTATION_METHODS = [
    "nan",
    "mean",
    "median",
    "pick",
    "pick1",
    "skip",
    "mode",
]
DEFAULT_IMPUTATION_METHOD_PER_TYPE = dict(
    bool="pick",
    int="nan",
    float="median",
    percentage="median",
    dict="pick",
    index="skip",
    str="skip",
    date="skip",
    undefined="skip",
)


def _get_variable():
    return {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }


def _get_record():
    return pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 12],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )


def add_impute_methods(impute_methods: dict, methods_to_add: dict):
    for impute_type, impute_method in methods_to_add.items():
        if impute_method not in AVAILABLE_IMPUTATION_METHODS:
            raise ValueError(f"Method  {impute_method} is not available")
        try:
            impute_methods[impute_method].append(impute_type)
        except KeyError:
            impute_methods[impute_method] = [impute_type]
    return impute_methods


def set_impute_method_per_type(impute_method_per_type):
    """
    Store the dictionary with the impute method per type in to a dictionary with types per methods

    Args:
        impute_method_per_type: dict
            metho per type

    Returns:
        dict: type per method
    """
    impute_methods = {}
    impute_methods = add_impute_methods(
        impute_methods, methods_to_add=impute_method_per_type
    )
    impute_methods = add_impute_methods(
        impute_methods, methods_to_add=DEFAULT_IMPUTATION_METHOD_PER_TYPE
    )

    for impute_method in AVAILABLE_IMPUTATION_METHODS:
        if impute_method not in impute_methods.keys():
            impute_methods[impute_method] = None

    return impute_methods


class ImputeSettings:
    def __init__(
        self,
        id_key="be_id",
        group_by=None,
        set_seed=1,
        **kwargs,
    ):
        """
        Store the impute settings

        Args:
            group_by: None of list of comma seperated column names. If None, the default is taken
            set_seed: int value of the seed to take for the random generator

        """

        self.id_key = id_key

        if group_by is None:
            self.group_by = ";".join(["sbi, gk", "gk", "internet"])
        else:
            self.group_by = ";".join(group_by)

        self.methods = set_impute_method_per_type(impute_method_per_type=kwargs)

        self.settings = {
            "group_by": self.group_by,
            "imputation_methods": self.methods,
            "set_seed": set_seed,
        }


def get_init_median():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": None,
            "pick": ["dict", "bool"],
            "mode": None,
            "median": ["float", "percentage"],
            "nan": ["int"],
            "skip": ["index", "str", "date", "undefined"],
            "mean": None,
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def get_init_mean():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": None,
            "pick": ["dict", "bool"],
            "mode": None,
            "median": None,
            "nan": ["int"],
            "skip": ["index", "str", "date", "undefined"],
            "mean": ["float", "percentage"],
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def get_init_mode():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": None,
            "pick": ["dict", "bool"],
            "mode": ["float", "percentage"],
            "median": None,
            "nan": ["int"],
            "skip": ["index", "str", "date", "undefined"],
            "mean": None,
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def get_init_pick1():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": ["float", "percentage", "dict", "bool"],
            "pick": None,
            "mode": None,
            "median": None,
            "nan": ["int"],
            "skip": ["index", "str", "date", "undefined"],
            "mean": None,
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def get_init_nan():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": None,
            "pick": None,
            "mode": None,
            "median": None,
            "nan": ["int", "float", "percentage", "dict", "bool"],
            "skip": ["index", "str", "date", "undefined"],
            "mean": None,
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def get_init_pick():
    impute_settings = {
        "group_by": "sbi, gk; gk; internet",
        "imputation_methods": {
            "pick1": None,
            "pick": ["int", "float", "percentage", "dict", "bool"],
            "mode": None,
            "median": None,
            "nan": None,
            "skip": ["index", "str", "date", "undefined"],
            "mean": None,
        },
        "set_seed": 1,
    }
    id_key = "be_id"
    return impute_settings, id_key


def test_failed_imputegaps():
    """API Tests"""
    with pytest.raises(AttributeError):
        test_obj = ImputeGaps()


def test_dict_pick():
    """
     var_type: float
     missing: 1
     imputatie: o.b.v. sbi x gk
     methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame([
            [1, 1, 'A', '10', 1], [2, 1, 'A', '10', 1], [3, 1, 'A', '10', 1], [4, 1, 'A', '10', 1],
            [1, 1, 'A', '10', 2], [2, 1, 'A', '10', 2], [3, 1, 'A', '10', 2], [4, 1, 'A', '10', 2],
            [1, 1, 'A', '10', None], [2, 1, 'A', '10', None], [3, 1, 'A', '10', None], [4, 1, 'A', '10', None]
        ],
        columns=['be_id', 'internet', 'sbi', 'gk', 'telewerkers'])
    variables = {'telewerkers': {'type': 'dict', 'no_impute': False, 'no_impute': None, 'filter': None, 'impute_only': None}}

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([float(1),1,1,1,2,2,2,2,2,1,2,1], copy=False, name='telewerkers')

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df['telewerkers'], expected)

def test_float_median_p1():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data
    # impute = ImputeSettings()

    # records_df = _get_record()
    # variables = _get_variable()
    impute_settings, id_key = get_init_median()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 12],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(
        records_df=records_df,
        variables=variables,
        impute_settings=impute_settings,
        id_key=id_key,
    )

    # Maak verwacht
    expected = pd.Series([float(10), 20, 12, 12], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_median_p2():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. gk
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_median()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 12],
            [4, 1, "A", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([float(10), 20, 12, 12], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_median_p3():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. internet
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_median()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 12],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([float(10), 20, 12, 12], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_median_p4():
    """
    var_type: float
    missing: 1
    imputatie: None
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_median()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mean_p1():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_mean()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10.1],
            [2, 1, "A", "10", 20.3],
            [3, 1, "C", "20", 30.4],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([10.1, 20.3, 30.4, 15.2], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mean_p2():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. gk
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_mean()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 5.5],
            [3, 1, "A", "10", 8.7],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([1, 5.5, 8.7, 5.06667], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mean_p3():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. internet
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_mean()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 5.5],
            [3, 1, "C", "20", None],
            [4, 0, "C", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([1, 5.5, 3.25, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mean_p4():
    """
    var_type: float
    missing: 1
    imputatie: None
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_mean()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mode_p1():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: modus
    """
    # Maak data
    impute_settings, id_key = get_init_mode()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 10],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([float(10), 20, 10, 10], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mode_p2():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. gk
    methode: modus (er is geen modus) -> in dat geval wordt het kleinste getal geimputeerd
    """
    # Maak data
    impute_settings, id_key = get_init_mode()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 12],
            [4, 1, "A", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([float(10), 20, 12, 10], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_mode_p4():
    """
    var_type: float
    missing: 1
    imputatie: None
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_mode()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_pick1_p1():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10.1],
            [2, 1, "A", "10", 20.3],
            [3, 1, "C", "20", 30.4],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([10.1, 20.3, 30.4, 1.0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_pick1_p2():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. gk
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 0],
            [3, 1, "A", "10", 1],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([1, 0, 1, float(1)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_pick1_p4():
    """
    var_type: float
    missing: 1
    imputatie: None
    methode: median
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series(
        [float(1), 1, 1, 1], dtype="object", copy=False, name="telewerkers"
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_float_nan_p1():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: mean
    """
    # Maak data
    impute_settings, id_key = get_init_nan()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10.1],
            [2, 1, "A", "10", 20.3],
            [3, 1, "C", "20", 30.4],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([10.1, 20.3, 30.4, 0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_nan_p1():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_nan()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 0],
            [2, 1, "A", "10", 1],
            [3, 1, "C", "20", 1],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0, 1, 1, float(0)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_pick1_p1():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 0],
            [2, 1, "A", "10", 1],
            [3, 1, "C", "20", 1],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0, 1, 1, float(1)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_pick_p1():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 0],
            [2, 1, "A", "10", 1],
            [3, 1, "C", "20", 1],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0, 1, 1, float(1)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_pick_p2():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 0],
            [2, 1, "A", "10", 0],
            [3, 1, "A", "10", 0],
            [4, 1, "B", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0, 0, 0, float(0)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_pick_p3():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", "0.0"],
            [2, 1, "A", "10", "1.0"],
            [3, 1, "A", "10", "1.0"],
            [4, 1, "B", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series(["0.0", "1.0", "1.0", "1.0"], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_bool_pick_p4():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick1_p1a():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.0],
            [2, 1, "A", "10", 2.0],
            [3, 1, "C", "20", 3.0],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([1.0, 2.0, 3.0, float(1)], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick1_p1b():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", "1.0"],
            [2, 1, "A", "10", "2.0"],
            [3, 1, "C", "20", "3.0"],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series(["1.0", "2.0", "3.0", "1.0"], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick1_p1c():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 4.0],
            [2, 1, "A", "10", 2.0],
            [3, 1, "C", "20", 3.0],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([4.0, 2.0, 3.0, 1.0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick1_p1d():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 0.0],
            [2, 1, "A", "10", 2.0],
            [3, 1, "C", "20", 3.0],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0.0, 2.0, 3.0, 1.0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick1_p1e():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick1()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", "4.0"],
            [2, 1, "A", "10", "2.0"],
            [3, 1, "C", "20", "3.0"],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series(["4.0", "2.0", "3.0", "1.0"], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick_p1():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.0],
            [2, 1, "A", "10", 2.0],
            [3, 1, "A", "10", 2.0],
            [4, 1, "A", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([1.0, 2.0, 2.0, 2.0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick_p2():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", "1.0"],
            [2, 1, "A", "10", "2.0"],
            [3, 1, "A", "10", "2.0"],
            [4, 1, "B", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series(["1.0", "2.0", "2.0", "2.0"], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_dict_pick_p4():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "dict",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)


def test_index_pick_p1():
    """
    var_type: bool
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [2, 1, "A", "10", None],
            [3, 1, "A", "10", None],
            [4, 1, "B", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "index",
            "no_impute": False,
            "no_impute": None,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df["telewerkers"], expected)

def test_impute_method():
    """
     var_type: bool
     missing: 1
     imputatie: o.b.v. sbi x gk
     methode: nan
    """
    # Maak data
    impute_settings, id_key = get_init_pick()
    records_df = pd.DataFrame(
        [[1, 1, 'A', '10', None], [2, 1, 'A', '10', None], [3, 1, 'A', '10', None], [4, 1, 'B', '10', None]],
        columns=['be_id', 'internet', 'sbi', 'gk', 'telewerkers'])
    variables = {
        'telewerkers': {'type': 'float', 'no_impute': None, 'filter': None,
                        'impute_only': None, 'impute_method': 'nan'}}

    # Run ImputeGaps
    test = ImputeGaps(records_df, variables, impute_settings, id_key)

    # Maak verwacht
    expected = pd.Series([0,0,0,0], copy=False, name='telewerkers')

    # Test uitvoeren
    pd.testing.assert_series_equal(test.records_df['telewerkers'], expected)

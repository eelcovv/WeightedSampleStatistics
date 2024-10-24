from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if the project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .core import WeightedSampleStatistics as WeightedSampleStatistics
from .variable_properties import VariableProperties as VariableProperties
from .utils import make_negation_name as make_negation_name
from .utils import get_records_select as get_records_select
from .utils import prepare_df_for_statistics as prepare_df_for_statistics
from .utils import reorganise_stat_df as reorganise_stat_df

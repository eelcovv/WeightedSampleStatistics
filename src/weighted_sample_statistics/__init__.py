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

import argparse
import logging
import sys
import pandas as pd

from impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=None)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """
    doc here
    """
    args = parse_args(args)
    # setup_logging(args.loglevel)
    _logger.debug("Starting class ImputeGaps.")

    records_df = pd.DataFrame()

    ImputeGaps(records_df)

    # use eval class here
    _logger.info("ImputeGaps has finished.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

import argparse
import codecs
import logging
import sys

import pandas as pd
import yaml

from imputegaps.impute_gaps import ImputeGaps

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
    parser.add_argument("records_df", default=None)
    parser.add_argument("--variables", default=None)
    parser.add_argument("--impute_settings", default=None)
    parser.add_argument("--group_by", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--loglevel", default=None)
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

    _logger.debug("Starting class ImputeGaps.")

    # Get command line arguments and set up logging
    args = parse_args(args)
    setup_logging(args.loglevel)

    # Read input files
    records_df = pd.read_csv(args.records_df, sep=";")
    variables = pd.read_csv(args.variables, sep=";")
    id_key = args.id

    # Read settings file
    with codecs.open(args.impute_settings, "r", encoding="UTF-8") as stream:
        impute_settings = yaml.load(stream=stream, Loader=yaml.Loader)["general"][
            "imputation"
        ]

    # Convert variables to dictionary
    # variables.set_index("naam", inplace=True)
    variables = variables.to_dict("index")

    # Start class ImputeGaps
    ImputeGaps(records_df, variables, impute_settings, id_key=id_key)

    _logger.info("Class ImputeGaps has finished.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

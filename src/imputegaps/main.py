import argparse
import logging
import sys
import pandas as pd
import yaml
import codecs
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
    parser.add_argument("--records_df", default=None)
    parser.add_argument("--variables", default=None)
    parser.add_argument("--impute_settings", default=None)
    parser.add_argument("--sbi_key", default=None)
    parser.add_argument("--gk_key", default=None)
    parser.add_argument("--be_id", default=None)
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

    # Bestanden inlezen
    records_df = pd.read_csv(args.records_df)
    lst = []
    zzp = [
        "zzp_zijn",
        "zzp_openbare_wifi",
        "zzp_zelfde_apparaten",
        "zzp_uur",
        "zzp_omzet",
        "zzp_voorzieningen_opdrachtgever",
        "zzp_tijd_voorzieningen_opdrachtgever",
        "zzp_aantal_opdrachtgever",
        "zzp_wie_ict_veiligheid",
        "zzp_omzet_helft_opdrachtgever",
    ]
    for col in records_df.columns:
        df = records_df[col]
        if df.isnull().sum() > 0 and col not in zzp:
            lst = lst + [col]

    records_df = records_df[[args.sbi_key] + [args.gk_key] + [args.be_id] + lst]

    variables = pd.read_csv(args.variables)
    with codecs.open(args.impute_settings, "r", encoding="UTF-8") as stream:
        impute_settings = yaml.load(stream=stream, Loader=yaml.Loader)["general"][
            "impute_options"
        ]

    # Variables opschonen (als dict, kies 'statline')
    for i in range(variables.shape[0]):
        filter = variables.loc[i, "filter"]
        try:
            filter = eval(filter)["statline"]
        except NameError:
            filter = filter
        except TypeError:
            continue
        if filter == "nan":
            filter = None
        variables.loc[i, "filter"] = filter

    variables = variables[["Unnamed: 0", "type", "no_impute", "filter"]]
    variables.set_index("Unnamed: 0", inplace=True)
    variables = variables.to_dict("index")

    ImputeGaps(
        records_df,
        variables,
        impute_settings,
        sbi_key=args.sbi_key,
        gk6_label=args.gk_key,
        be_id=args.be_id,
    )

    _logger.info("ImputeGaps has finished.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

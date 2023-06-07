import argparse
import molli.lib_gen.test_aso.alt_aso as aa

parser = argparse.ArgumentParser(
    "molli descriptors",
    description="Generates aso descriptor data for a conformer library"
)

parser.add_argument(
    "input",
    action="store",
    required=True,
    type=str,
    metavar='<fpath>',
    help="ConformerLibrary file to be parsed"
)

parser.add_argument(
    '--output',
    '-o',
    action='store',
    type=str,
    default=None,
    metavar='<fpath>',
    help='File path for directory to write to. Defaults to "aso.h5" in same directory as alt_aso.py'
)

def molli_main(args, config=None, output=None, **kwargs):
    args = parser.parse_args(args)

    clib = args.file
    output = args.output

    aa.aso_description(clib, output)
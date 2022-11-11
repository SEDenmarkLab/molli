import molli as ml
from argparse import ArgumentParser
from pprint import pprint
from importlib import import_module
import yaml
from . import scripts
from warnings import warn

# KNOWN_CMDS = ["info", "split", "optimize", "whatever"]
KNOWN_CMDS = ["list", *scripts.__all__]

arg_parser = ArgumentParser(
    "molli",
    description=f"[molli {ml.__version__} description] MOLLI package is an API that intends to create a concise and easy-to-use syntax that encompasses the needs of cheminformatics (especially so, but not limited to the workflows developed and used in the Denmark laboratory.",
    add_help=False,
)

arg_parser.add_argument(
    "COMMAND",
    choices=KNOWN_CMDS,
    # nargs=1,
    help="This is main command that invokes a specific standalone routine in MOLLI. To get full explanation of available commands, run `molli list`",
)

arg_parser.add_argument(
    "-O",
    "--OUTPUT",
    action="store",
    metavar="<file.log>",
    help="Sets the file that molli output will be printed into",
)

arg_parser.add_argument(
    "-C",
    "--CONFIG",
    action="store",
    metavar="<file.yml>",
    help="Sets the file from which molli configuration will be read from",
)

arg_parser.add_argument(
    "-H",
    "--HELP",
    action="help",
)

arg_parser.add_argument("-V", "--VERSION", action="version", version=ml.__version__,)


def main():
    parsed, unk_args = arg_parser.parse_known_args()

    cmd = parsed.COMMAND

    match cmd:
        # cases can override default behavior, which is to import the module from standalone
        case "list":

            for m in scripts.__all__:
                try:
                    requested_module = import_module(f"molli.scripts.{m}")
                    requested_module.molli_main
                except:
                    with ml.aux.ForeColor("ltred"):
                        print(f"molli {m}:\nNOT IMPLEMENTED\n")
                else:
                    with ml.aux.ForeColor("green"):
                        print(f"molli {m}")
                    if isinstance(doc := requested_module.__doc__, str):
                        print(doc.strip())
                    else:
                        with ml.aux.ForeColor("ltred"):
                            print("No documentation available")

                    if hasattr(requested_module, "arg_parser"):
                        print(requested_module.arg_parser.format_usage())
                    else:
                        with ml.aux.ForeColor("ltred"):
                            print("No documentation available")

        case _:
            if parsed.CONFIG is not None:
                warn(
                    "CONFIG file option is implemented as EXPERIMENTAL. Do not rely on its results."
                )
            try:
                requested_module = import_module(f"molli.scripts.{cmd}")
            except:
                raise NotImplementedError(
                    f"Requested module <{cmd}> does not seem to be implemented. Check with the developers!"
                )
            else:
                requested_module.molli_main(
                    unk_args,
                    config=parsed.CONFIG,
                    output=parsed.OUTPUT,
                )


if __name__ == "__main__":
    main()

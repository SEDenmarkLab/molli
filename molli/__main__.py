from argparse import ArgumentParser
from pprint import pprint
from importlib import import_module
import yaml
from . import scripts
from . import config
from warnings import warn
import logging
import sys
import molli as ml
from uuid import uuid1 # Needed for a lock file
import os
from mpi4py import MPI
from socket import gethostname

KNOWN_CMDS = ["list", *scripts.__all__]

arg_parser = ArgumentParser(
    "molli",
    description=f"MOLLI package is an API that intends to create a concise and easy-to-use syntax that encompasses the needs of cheminformatics (especially so, but not limited to the workflows developed and used in the Denmark laboratory.",
    add_help=False,
)

arg_parser.add_argument(
    "COMMAND",
    choices=KNOWN_CMDS,
    # nargs=1,
    help="This is main command that invokes a specific standalone routine in MOLLI. To get full explanation of available commands, run `molli list`",
)

arg_parser.add_argument(
    "-C",
    "--CONFIG",
    action="store",
    metavar="<file.yml>",
    default=None,
    help="Sets the file from which molli configuration will be read from",
)

arg_parser.add_argument(
    "-L",
    "--LOG",
    action="store",
    metavar="<file.log>",
    default=None,
    help="Sets the file that will contain the output of molli routines.",
)

arg_parser.add_argument(
    "-V",
    "--VERBOSITY",
    action="store",
    metavar="0..5",
    default=3,
    type=int,
    help="Sets the level of verbosity for molli output. Negative numbers will remove all output. Defaults to 0.",
)

arg_parser.add_argument(
    "-H", "--HELP", action="help", help="show help message and exit"
)

arg_parser.add_argument(
    "--VERSION",
    action="version",
    version=config.VERSION,
)


def main():
    comm = MPI.COMM_WORLD

    parsed, unk_args = arg_parser.parse_known_args()
    cmd = parsed.COMMAND

    #########################################
    #TODO Set up the logger HERE!
    # This will make sure that all molli stuff is now fully captured.
    logging.basicConfig(level=50-parsed.VERBOSITY*10, handlers=[logging.NullHandler()])
    logger = logging.getLogger('molli')
    logger.setLevel(50-parsed.VERBOSITY*10)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(50-parsed.VERBOSITY*10)

    # create formatter
    # formatter = logging.Formatter('%(asctime)s (%(relativeCreated)d) - %(name)s - %(module)s (%(pathname)s) - %(levelname)s - %(message)s')
    host = gethostname()
    rank = comm.Get_rank()
    formatter = logging.Formatter('{levelname:s}: {message:s} ({name:s}:{lineno} at %s:%d {asctime:s})' % (host, rank), style="{")
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    #########################################

    # This thing allows to 
    if parsed.CONFIG is not None:
        with open(parsed.CONFIG) as f:
            _config_f = yaml.safe_load(f)
    else:
        _config_f = None
    ml.config.configure(_config_f)
 
    match cmd:
        # cases can override default behavior, which is to import the module from standalone
        case "list":

            for m in scripts.__all__:
                try:
                    requested_module = import_module(f"molli.scripts.{m}")
                    requested_module.molli_main
                except Exception as xc:
                    with ml.aux.ForeColor("ltred"):
                        print(f"molli {m}:\nERROR: {xc}\n")
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
            try:
                requested_module = import_module(f"molli.scripts.{cmd}")
            except:
                raise NotImplementedError(
                    f"Requested module <{cmd}> does not seem to be implemented. Check with the developers!"
                )
            else:
                # This may need to be revised. Not sure if parent creation is a great idea.
                
                if rank == 0:
                    ml.config.SHARED_DIR.mkdir(parents=True, exist_ok=True)
                    ml.config.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

                    shared_lkfile = ml.config.SHARED_DIR / f"molli-{uuid1().hex}.lock"
                    scratch_lkfile = ml.config.SCRATCH_DIR / f"molli-{uuid1().hex}.lock"
                else:
                    shared_lkfile = None
                    scratch_lkfile = None
                
                shared_lkfile = comm.bcast(shared_lkfile, 0)
                scratch_lkfile = comm.bcast(scratch_lkfile, 0)

                try:                
                    _code = requested_module.molli_main(
                        unk_args,
                        shared_lkfile=shared_lkfile,
                        scratch_lkfile=scratch_lkfile,
                    )
                except Exception as xc:
                    logger.exception(xc)
                    _code = 1 # Maybe change this later
                
                # This should free up these keys in case it still exists
                if shared_lkfile.is_file():
                    os.remove(shared_lkfile)
                
                if scratch_lkfile.is_file():
                    os.remove(scratch_lkfile)
                
                return _code

if __name__ == "__main__":
    exit(main())

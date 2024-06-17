# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
This is the main file that dispatches `molli xxx` commands.
"""

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
from uuid import uuid1  # Needed for a lock file
import os
from socket import gethostname
from datetime import datetime
import shlex

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
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Sets the level of verbosity for molli output.",
)

arg_parser.add_argument(
    "-H",
    "--HELP",
    action="help",
    help="show help message and exit",
)

arg_parser.add_argument(
    "-V",
    "--VERSION",
    action="version",
    version=config.VERSION,
)


def main():
    parsed, unk_args = arg_parser.parse_known_args()
    cmd = parsed.COMMAND

    log_lvl = 30 - max(parsed.verbose or 0, 3) * 10

    #########################################
    # TODO Set up the logger HERE!
    # This will make sure that all molli stuff is now fully captured.
    logging.basicConfig(level=log_lvl, handlers=[logging.NullHandler()])
    logger = logging.getLogger("molli")
    logger.setLevel(log_lvl)

    if parsed.LOG is None:
        ch = logging.StreamHandler()
    else:
        with open(parsed.LOG, "at") as f:
            f.write(ml.config.SPLASH)
        ch = logging.FileHandler(parsed.LOG)

    ch.setLevel(log_lvl)

    host = gethostname()
    if parsed.verbose >= 3:
        formatter = logging.Formatter(
            "{levelname:s}: {message:s} ({name:s}:{lineno} {asctime:s})",
            style="{",
        )
    else:
        formatter = logging.Formatter(
            "{message:s}",
            style="{",
        )
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

    _code = 0

    match cmd:
        # cases can override default behavior, which is to import the module from standalone
        case "list":
            for m in scripts.__all__:
                try:
                    requested_module = import_module(f"molli.scripts.{m}")
                    requested_module.molli_main
                except Exception as xc:
                    if parsed.verbose >= 2:
                        with ml.aux.ForeColor("ltred"):
                            print(f"molli {m}:\nERROR: {xc}\n")
                else:
                    with ml.aux.ForeColor("green"):
                        print(f"molli {m}")
                    if isinstance(doc := requested_module.__doc__, str):
                        if parsed.verbose >= 1:
                            print(doc.strip() + "\n")
                    else:
                        with ml.aux.ForeColor("ltred"):
                            print("No documentation available")

                    if hasattr(requested_module, "arg_parser"):
                        if parsed.verbose >= 2:
                            print(requested_module.arg_parser.format_usage())
                    else:
                        with ml.aux.ForeColor("ltred"):
                            print("No documentation available")

        case _ as _cmd:
            try:
                requested_module = import_module(f"molli.scripts.{cmd}")
            except:
                raise NotImplementedError(
                    f"Requested module <{cmd}> does not seem to be implemented. Check with the developers!"
                )
            else:
                # This may need to be revised. Not sure if parent creation is a great idea.

                try:
                    _code = (
                        requested_module.molli_main(unk_args, verbosity=parsed.verbose)
                        or 0
                    )
                except KeyboardInterrupt:
                    logger.error("Keyboard interrupt")
                    _code = 1
                except Exception as xc:
                    logger.exception(xc)
                    _code = 1  # Maybe change this later

    return _code


if __name__ == "__main__":
    exit(main())

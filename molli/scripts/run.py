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
Run a custom script that defines `molli_main(args, config, output, **)`
"""

raise NotImplementedError
import molli as ml
from pprint import pprint
from argparse import ArgumentParser

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli run",
    description=__doc__,
    add_help=False,
)

arg_parser.add_argument("script", action="store", type=str, metavar="<script.py>")


def molli_main(args,  **kwargs):
    parsed, unknown = arg_parser.parse_known_args(args)

    with ml.aux.ForeColor("yellow"):
        print(
            "WARNING: this routine can execute arbitrary code. Do not feed untrusted .py files into it!"
        )

    extm = ml.aux.load_external_module(parsed.script, "extm")
    extm.molli_main(unknown, config=config, output=output, **kwargs)

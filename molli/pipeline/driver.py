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
This describes the foundational class for all drivers.
"""

import shutil


class DriverBase:
    def __init__(
        self,
        executable: str = None,
        nprocs: int = 1,
        memory: int = None,
        envars: dict = None,
        check_exe: bool = True,
        find: bool = True,
    ) -> None:
        self.executable = executable
        if hasattr(self, "default_executable"):
            self.executable = self.executable or self.default_executable
        self.nprocs = nprocs
        self.envars = envars
        self.memory = memory

        if check_exe and not (which_exe := self.which()):
            raise FileNotFoundError(
                f"Requested executable {self.executable!r} for {self.__class__.__name__!r} is not reachable."
            )
        elif find:
            self.executable = which_exe

    def which(self):
        return shutil.which(self.executable)

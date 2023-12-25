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
This function provides a convenience `LineReader` class that comes very handy in the mol2 file parsing
"""

from io import StringIO
from typing import Callable
from collections import deque


class LineReader:
    def __init__(
        self,
        io: StringIO,
        post: Callable[[str], str] = None,
    ):
        self._io = io
        self._current_line = 0
        self._post = post
        self._extra_lines = deque()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._extra_lines):
            _line = self._extra_lines.popleft()
        else:
            _line = next(self._io)

        if self._post is None:
            line = _line
        else:
            line = self._post(_line)

        self._current_line += 1

        return line

    def next_noexcept(self) -> str | None:
        try:
            line = next(self)
        except StopIteration:
            return None
        except:
            raise
        else:
            return line

    def put_back(self, line: str):
        self._current_line -= 1
        self._extra_lines.append(line)

    @property
    def pos(self):
        return self._current_line

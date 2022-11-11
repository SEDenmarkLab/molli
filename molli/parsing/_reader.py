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

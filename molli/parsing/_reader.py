from io import StringIO
from typing import Callable


class LineReader:
    def __init__(
        self,
        io: StringIO,
        post: Callable[[str], str] = None,
    ):
        self._io = io
        self._current_line = 0
        self._post = post

    def __iter__(self):
        return self

    def __next__(self):
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
        """Return the line back into the iterator"""
        self._q.appendleft(line)

    @property
    def pos(self):
        return self._current_line

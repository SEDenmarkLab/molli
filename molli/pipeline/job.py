from ..storage import _Library
from typing import TypeVar, Generic, Callable, Any, Unpack, Iterable

JInT = TypeVar("JInT")
JOutT = TypeVar("JOutT")



class Job(Generic[JInT, JOutT]):

    def __init__(
        self,
        target: Callable,
        args: tuple[Any] = None,
        kwds: dict[str, Any] = None,
        cache: _Library = None,
    ):
        self.target = target
        self.args = args
        self.kwds = kwds
    
    def setup(self):
        pass

    def run(self, arg: JInT) -> JOutT:
        return self.target(arg, *self.args, **self.kwds)
    
    def persist(self, output: JOutT):
        pass

    def serialize(self, output: JOutT):
        pass
        
    def __call__(self, arg: Iterable[JInT]) -> JOutT:
        pass

# molli submit a.job -t mylib.mlib -b 64 --sge
from molli import config
from fasteners import InterProcessReaderWriterLock
from hashlib import sha3_512
from pathlib import Path
from base64 import urlsafe_b64encode


def rwlock(path: str | Path):
    """
    This is a convenience function that returns a reader/writer lock based on a file path.
    """
    abspath = Path(path).resolve()
    key = abspath.as_posix().encode()
    h = sha3_512(key).digest()
    (lkdir := config.SHARED_DIR / "lock").mkdir(exist_ok=True, parents=True)
    return lkdir / f"{urlsafe_b64encode(h).decode()}.lock"

# molli makes use of the following variables
# MOLLI_HOME : default= ~/.molli
# MOLLI_DATA_DIR : default= $MOLLI_HOME / data
# MOLLI_BACKUP_DIR : default= $MOLLI_HOME / backup
# MOLLI_SCRATCH_DIR : default= $MOLLI_HOME / scratch
# MOLLI_LOG_DIR : default= $MOLLI_HOME / log
# MOLLI_LOG_VERBOSITY : default= 3
import os, sys
from pathlib import Path
import logging

VERSION = "1.0.0b1"

HOME: Path = Path("~/.molli").expanduser()
USER_DATA_DIR: Path = HOME / "user_data"
BACKUP_DIR: Path = HOME / "backup"
SCRATCH_DIR: Path = HOME / "scratch"
SHARED_DIR: Path = HOME / "shared"

SPLASH = f"""
         __    __     ______     __         __         __    
        /\ "-./  \   /\  __ \   /\ \       /\ \       /\ \   
        \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \ \____  \ \ \  
         \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_____\  \ \_\ 
          \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/ 

                --- version {VERSION} ---
"""


def configure(config_from_file: dict[str, str] = None, **kwds):
    """
    This function populates the values of molli configuration
    """
    global HOME
    # logger = logging.getLogger("molli.config")

    requested_config: dict[str, str] = (config_from_file or {}) | kwds

    if "MOLLI_HOME" in os.environ:
        HOME = Path(os.environ["MOLLI_HOME"])
    elif requested_config is not None and "HOME" in requested_config:
        HOME = Path(requested_config["HOME"])
    else:
        HOME = Path("~/.molli").expanduser()

    os.environ["MOLLI_HOME"] = str(HOME)

    # tuple[type, default_value]
    _VAR_DEFAULTS = {
        "DATA_DIR": (Path, HOME / "data"),
        "BACKUP_DIR": (Path, HOME / "backup"),
        "SCRATCH_DIR": (Path, HOME / "scratch"),
        "SHARED_DIR": (Path, HOME / "shared"),
    }

    for varname, (vartype, vardefault) in _VAR_DEFAULTS.items():
        if (envarname := f"MOLLI_{varname}") in os.environ:
            value = vartype(os.environ[envarname])
        # elif varname in kwds:
        #     value = vartype(requested_config[varname])
        elif requested_config is not None and varname in requested_config:
            value = vartype(requested_config[varname])
        else:
            value = vartype(vardefault)

        globals()[varname] = value

    for k, v in requested_config.items():
        if k.startswith("ENV_"):
            envar = k.removeprefix("ENV_")
            value = os.path.expandvars(v)
            os.environ[envar] = value


# This happens if a configuration file exists
if (default_path := HOME / "config.yaml").is_file():
    import yaml

    with open(default_path, "rt") as f:
        cfg = yaml.safe_load(f)

    configure(cfg)

# This is to patch messagepack so that numpy arrays can be serialized
import msgpack_numpy

msgpack_numpy.patch()

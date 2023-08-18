# molli makes use of the following variables
# MOLLI_HOME : default= ~/.molli
# MOLLI_DATA_DIR : default= $MOLLI_HOME / data
# MOLLI_BACKUP_DIR : default= $MOLLI_HOME / backup
# MOLLI_SCRATCH_DIR : default= $MOLLI_HOME / scratch
# MOLLI_LOG_DIR : default= $MOLLI_HOME / log
# MOLLI_LOG_VERBOSITY : default= 3
import os, sys
from pathlib import Path
from . import __version__ as VERSION

HOME: Path = Path("~/.molli").expanduser()
USER_DATA_DIR: Path  = HOME / "user_data"
BACKUP_DIR: Path  = HOME / "backup"
SCRATCH_DIR: Path  = HOME / "scratch"
LOG_DIR: Path  = HOME / "log"
LOG_VERBOSITY: int  = 3

SPLASH = f"""
         __    __     ______     __         __         __    
        /\ "-./  \   /\  __ \   /\ \       /\ \       /\ \   
        \ \ \-./\ \  \ \ \/\ \  \ \ \____  \ \ \____  \ \ \  
         \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_____\  \ \_\ 
          \/_/  \/_/   \/_____/   \/_____/   \/_____/   \/_/ 

             --- version {VERSION} ---
"""

def configure(config_from_file: dict[str, str]=None):
    """
    This function populates the values of molli configuration
    """
    global HOME
    
    if "MOLLI_HOME" in os.environ:
        HOME = Path(os.environ["MOLLI_HOME"])
    elif config_from_file is not None and "HOME" in config_from_file:
        HOME = config_from_file["HOME"]
    else:
        HOME = Path("~/.molli").expanduser()
    
    # tuple[type, default_value]
    _VAR_DEFAULTS = {
        "HOME" : (Path, HOME),
        "DATA_DIR" : (Path, HOME / "data"),
        "BACKUP_DIR" : (Path, HOME / "backup"),
        "SCRATCH_DIR" : (Path, HOME / "scratch"),
        "LOG_DIR" : (Path, HOME / "log"),
        # Note that in molli's logic verbosity (v) 0-5 corresponds to 50 - 10*v logger level
        "LOG_VERBOSITY" : (int, 3), 
    }

    for varname, (vartype, vardefault) in _VAR_DEFAULTS.items():
        if (envarname := f"MOLLI_{varname}") in os.environ:
            value = vartype(os.environ[envarname])
        elif config_from_file is not None and varname in config_from_file:
            value = vartype(config_from_file[varname])
        else:
            value = vartype(vardefault)

        # Log something 
        globals()[varname] = value
    


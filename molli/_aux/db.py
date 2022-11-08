import requests
from requests import exceptions
from random import randint


def pdb_get(pdb_id: str) -> bytes:
    """
    # `retrieve_pdb`
    This function downloads a pdb file from the Protein Data Bank (https://files.rcsb.org/download)

    For more details, go to https://wwpdb.org


    ## Parameters

    `pdb_id: str`
        PDB identifier. Example: `'1I10'` for HUMAN MUSCLE L-LACTATE DEHYDROGENASE

    ## Returns

    `bytes`
        _description_
    """
    raise NotImplementedError


class ZINC15GetError(exceptions.HTTPError):
    """Raised when"""


def zinc15_get(zinc15_id: int | str, suffix: str = ".mol2"):
    _suffix = suffix or ""
    _zinc15_id = f"ZINC{zinc15_id:0>15}" if isinstance(zinc15_id, int) else zinc15_id

    req = requests.get(f"https://zinc15.docking.org/substances/{_zinc15_id}{_suffix}")

    if req.status_code == 200:
        return req.text
    else:
        raise ZINC15GetError(
            f"Request failed with code {req.status_code}: {req.reason}"
        )


def zinc15_yield(*zinc15_idx: int | str, suffix: str = ".mol2"):
    for zid in zinc15_idx:
        yield zinc15_get(zid, suffix=suffix)

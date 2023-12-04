from molli.config import VERSION
from .misc import ForeColor
from packaging.version import parse as parse_version


current_version = parse_version(VERSION)


def assert_molli_version_min(vmin: str):
    """
    # `assert_molli_version_min`
    Asserts the current version of molli is above the minimum requirement

    This is commonly encountered when scripts were written to use a very specific version of the API.
    This is a control step to ensure

    ## Parameters

    `vmin: str`
        Minimum required version. For instance, `'1.2.3b4'`

    ## Raises

    `AssertionError`
        If the test fails, it raises an error.
    """
    if current_version < (_vmin := parse_version(vmin)):
        raise AssertionError(
            f"Current molli version {current_version} is below the minimum requirement of {_vmin}"
        )


def assert_molli_version_max(vmax: str):
    """
    # `assert_molli_version_max`
    Asserts the current version of molli is below the maximum requirement

    This is commonly encountered when scripts were written to use a very specific version of the API.
    This ensures a control step to ensure compatibility.

    ## Parameters

    `vmax: str`
        Maximum required version. For instance, `'1.2.3b4'`

    ## Raises

    `AssertionError`
        If the test fails, it raises an error.
    """
    if current_version > (_vmax := parse_version(vmax)):
        raise AssertionError(
            f"Current molli version {current_version} is above the maximum requirement of {_vmax}"
        )


def assert_molli_version_in_range(vmin: str, vmax: str):
    """
    # `assert_molli_version_in_range`
    Asserts the current version of molli to be within the specified range

    This is commonly encountered when scripts were written to use a very specific version of the API.
    This ensures a control step to ensure compatibility.

    ## Parameters

    `vmin: str`
        _description_
    `vmax: str`
        _description_
    """
    assert_molli_version_min(vmin)
    assert_molli_version_max(vmax)

# This file can be left empty. It serves to indicate a valid submodule.
# The respective sub-submodules are meant to be loaded "lazily" so as not to invoke
# unnecessary dependencies before they might become necessary.

class RDKitException(Exception):
    "Raised when RDKit fails during Molecule Creation"
    pass

class RDKitKekulizationException(RDKitException):
    "Raised when RDKit fails to kekulize during Molecule Creation"
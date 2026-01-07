def detect_mpi_launch():
    """
    This function determines if the current mpi world has more than one process
    """
    try:
        import mpi4py
    except ImportError:
        return False
    else:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_size() > 1

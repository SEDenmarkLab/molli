import shutil


class DriverBase:
    def __init__(
        self,
        executable: str = None,
        nprocs: int = 1,
        envars: dict = None,
        check_exe: bool = True,
    ) -> None:
        self.executable = executable
        if hasattr(self, "default_executable"):
            self.executable = self.executable or self.default_executable
        self.nprocs = nprocs
        self.envars = envars

        if check_exe and not self.which():
            raise FileNotFoundError(
                f"Requested executable {self.executable!r} for {self.__class__.__name__!r} is not reachable."
            )

    def which(self):
        return shutil.which(self.executable)
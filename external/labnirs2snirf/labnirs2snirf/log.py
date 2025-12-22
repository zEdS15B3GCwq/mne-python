"""
Logging configuration utilities for the labnirs2snirf package.
"""

from logging import (
    CRITICAL,
    DEBUG,
    INFO,
    WARNING,
    FileHandler,
    Formatter,
    StreamHandler,
    getLogger,
)

LOGFILE_NAME = "labnirs2snirf.log"


def config_logger(file_logging: bool = False, verbosity_level: int = 0) -> None:
    """
    Configure the root logger for the application.

    Parameters
    ----------
    file_logging : bool, default = False
        If True, log messages will be written to a file. If False, log messages will
        be printed to the console.
    verbosity_level : int, default = 0
        The verbosity level of the log messages. Must be between 0 and 3.
        Possible values:
        - 0: disabled (no logging)
        - 1: WARNING (warnings and above)
        - 2: INFO (informational messages and above)
        - 3: DEBUG (debugging messages and above)

    Notes
    -----
    This function should be called only once, at the start of the application,
    and only if it's run as a script, not as a library. Most modules should invoke
    `getLogger(__name__)` to get a module-specific logger, which will inherit the
    configuration set here.
    """

    # verbosity_level -> loglevel
    match verbosity_level:
        case 0:
            loglevel = CRITICAL + 1
        case 1:
            loglevel = WARNING
        case 2:
            loglevel = INFO
        case 3:
            loglevel = DEBUG
        case _:
            raise ValueError("verbosity_level must be between 0 and 3")

    # file or stream handler with corresponding formats
    if file_logging:
        logfmt = Formatter(
            # fmt="%(asctime)s %(levelname)-8s [%(module)s.%(funcName)s:%(lineno)d] %(message)s",
            fmt="%(asctime)s %(levelname)-8s [%(name)s %(funcName)s:%(lineno)d] %(message)s",
            # datefmt="%Y.%m.%d %H:%M:%S",
            datefmt="%H:%M:%S",
        )
        # types indicated to satisfy mypy
        handler: FileHandler | StreamHandler = FileHandler(LOGFILE_NAME, mode="a")
    else:
        logfmt = Formatter(
            # fmt="%(levelname)-8s [%(module)s] %(message)s",
            fmt="%(levelname)-8s [%(name)s] %(message)s",
            # datefmt="%Y.%m.%d %H:%M:%S",
            datefmt="%H:%M:%S",
        )
        handler = StreamHandler()

    # Don't set level on handler, as it will make it impossible for pytest to capture logs.
    # For whatever reason...
    # handler.setLevel(loglevel)
    handler.setFormatter(logfmt)

    # assign handler to root logger
    log = getLogger()
    log.setLevel(loglevel)
    log.addHandler(handler)

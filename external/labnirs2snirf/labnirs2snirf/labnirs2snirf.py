"""
Main entrypoint for LABNIRS to SNIRF conversion when run as a script.
"""

import logging
import sys
from pprint import pformat

from .args import ArgumentError, Arguments
from .error import Labnirs2SnirfError
from .labnirs import read_labnirs
from .layout import read_layout, update_layout
from .log import config_logger
from .snirf import write_snirf


def main() -> int:
    """
    LABNIRS to SNIRF conversion script.

    This function coordinates the full conversion workflow when
    run as `python -m labnirs2snirf` on the command line:

    1. Parse command-line arguments
    2. Configure logging based on user preferences
    3. Read and parse LABNIRS data file
    4. Optionally add probe position information
    5. Write output in SNIRF format

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure.

    Notes
    -----
    Exception handling is designed with the intention to reduce end-user exposure
    to stack traces and technical details. Most exceptions are hidden from end users,
    providing only a concise error message. Detailed error information can be
    obtained by increasing verbosity with -v flags and/or using --log to write
    to a log file.
    """
    log = None
    args = None
    try:
        # 1. Read and check arguments
        args = Arguments().parse(sys.argv[1:])

        # Since we're not a library, we will configure logging here.
        config_logger(file_logging=args.log, verbosity_level=args.verbosity)
        log = logging.getLogger(__name__)
        log.info("Logger configured")

        log.debug("Parsed arguments: %s", pformat(args, indent=2))

        # 2. Read in labNIRS file
        log.info("Reading labNIRS data")
        data = read_labnirs(
            data_file=args.source_file,
            keep_category=args.type,
            drop_subtype=args.drop,
        )
        # raise RuntimeError("Test exception")

        # 3. Add probe positions if provided
        if args.locations is not None:
            log.info("Reading probe layout from file")
            layout = read_layout(args.locations)
            update_layout(data, layout)

        # 4. Export SNIRF data
        log.info("Writing SNIRF file")
        write_snirf(data, args.target_file)

        log.info("Successfully completed conversion")
        return 0

    # Hide all exceptions from end users as they aren't necessarily developers.
    # Provide error message, explain how to enable logging and get more information about the issue.
    except ArgumentError as e:
        print(f"Argument error: {e}")
        return 1
    except Labnirs2SnirfError as e:
        if log is not None:
            log.exception("%s", e)
        print(
            f"Conversion failed: {e}\n"
            "Increase verbosity (-v, -vv, -vvv) for more details. Use --log to log messages to a file.",
        )
        return 1
    except Exception as e:  # pylint: disable=W0718
        print(
            "Something went wrong. Increase verbosity (-v, -vv, -vvv) for more details. Use --log to log messages to a file.",
        )
        if log is not None:
            log.exception("Exception received. Error message: %s", e)
        else:
            print("Logging not configured, dumping exception:")
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())

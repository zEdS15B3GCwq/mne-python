"""
Module for handling command-line arguments.
"""

import argparse
from pathlib import Path
from typing import Self

from .error import Labnirs2SnirfError


class ArgumentError(Labnirs2SnirfError):
    """Error indicating invalid command-line arguments."""


def _file_must_exist(path_str: str) -> Path:
    """
    Validate that a file path exists.

    This function is used as an argparse type validator to ensure that
    a provided path points to an existing file.

    Parameters
    ----------
    path_str : str
        String representation of the file path to validate.

    Returns
    -------
    Path
        Validated Path object if the file exists.

    Raises
    ------
    ArgumentError
        If the path is empty, does not exist, or is not a file.
    """
    if not path_str:
        raise ArgumentError("Path must not be empty.")
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        raise ArgumentError(f"File '{path_str}' does not exist.")
    return path


def _file_must_not_exist(path_str: str) -> Path:
    """
    Validate that a file path does not already exist.

    This function is used as an argparse type validator to ensure that
    a provided path does not point to an existing file, preventing accidental
    overwrites.

    Parameters
    ----------
    path_str : str
        String representation of the file path to validate.

    Returns
    -------
    Path
        Validated Path object if the path does not exist.

    Raises
    ------
    ArgumentError
        If the path is empty or already exists.
    """
    if not path_str:
        raise ArgumentError("Path must not be empty.")
    path = Path(path_str)
    if path.exists():
        raise ArgumentError(f"Path '{path_str}' already exists.")
    return path


def _validate_drop_value(value: str) -> str:
    """
    Validate a single drop value for the --drop argument.

    Parameters
    ----------
    value : str
        Drop value to validate. Can be 'HbT', 'HbO', 'HbR' (case insensitive),
        or a positive integer indicating wavelength.

    Returns
    -------
    str
        Validated and normalized (lowercase) drop value.

    Raises
    ------
    ArgumentError
        If the value is not a valid drop type.
    """
    value = value.lower().strip()
    if value.isdecimal() and int(value) > 0:
        return value
    if value in {"hbt", "hbo", "hbr"}:
        return value
    raise ArgumentError(
        f"Invalid drop type '{value}'. Must be 'HbT', 'HbO', 'HbR' "
        f"(case insensitive) or a positive non-zero integer indicating wavelength.",
    )


# class _NotImplementedAction(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         raise ArgumentError(f"'{option_string}' option is not implemented yet.")


class Arguments:
    """
    Class to handle configuration and parsing of command-line arguments.

    Parameters
    ----------
    progname : str or None, default=__package__
        Program name to display in help message. If None, defaults to package name.
    """

    parser: argparse.ArgumentParser
    source_file: Path
    target_file: Path
    type: str
    log: bool
    verbosity: int
    locations: Path | None
    drop: set[str] | None

    def __init__(self, progname: str | None = __package__):
        """
        Initialize the Arguments parser.

        Parameters
        ----------
        progname : str or None, default=__package__
            Program name to display in help message. If None, defaults to package name.
        """
        parser = argparse.ArgumentParser(
            description="Convert LabNIRS data to SNIRF format.",
            allow_abbrev=False,
            prog=progname if progname else "labnirs2snirf",
        )
        parser.add_argument(
            "source_file",
            help="path to LabNIRS data file (*.txt)",
            type=_file_must_exist,
        )
        parser.add_argument(
            "target_file",
            help="path to output file (*.snirf); if not specified, output is written to the current directory as <out.snirf>",
            nargs="?",
            default="out.snirf",
            type=_file_must_not_exist,
        )
        parser.add_argument(
            "--locations",
            help="Path to file holding probe location data. "
            "Location files are expected to follow the .sfp format, i.e. "
            "tab-separated text file with columns: optode name, x, y, and z, "
            "where x, y, an z are the 3D coordinates of the optode. "
            "Conflicts with -g.",
            type=_file_must_exist,
        )
        parser.add_argument(
            "--type",
            help="Include specific data category only. "
            "'Raw' includes raw voltage, 'Hb' includes heamoglobin' data, 'all' (default) includes both.",
            choices=["hb", "raw", "all"],
            type=str.lower,
            default="all",
        )
        parser.add_argument(
            "--drop",
            help="Drop specific data types. "
            "Possible values: HbT, HbO, HbR, or wavelength integers (e.g. 780). Can be used multiple times.",
            action="append",
            type=_validate_drop_value,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            help="Increase verbosity of output, can be used multiple times. One -v for ERROR/WARNING level, "
            "-vv for INFO level, -vvv for DEBUG level. Combine with --log to redirect log output to file.",
            default=0,
            dest="verbosity",
        )
        parser.add_argument(
            "--log",
            action="store_true",
            help="Redirects logging to file labnirs2snirf.log in the current directory. "
            "Logging level is controlled by -v/--verbose. Specifying --log implies -v."
            "Log messages written to file contain additional information about where messages occurred.",
        )
        # parser.add_argument(
        #     "-csv",
        #     "--tasks",
        #     help="(NOT IMPLEMENTED) path to .csv file containing task timings",
        #     type=Path,
        #     action=NotImplementedAction,
        #     # action=FileMustExistAction,
        # )
        # parser.add_argument(
        #     "-pat",
        #     "--patient",
        #     help="(NOT IMPLEMENTED) path to .pat file containing patient and study metadata",
        #     type=Path,
        #     action=NotImplementedAction,
        #     # action=FileMustExistAction,
        # )

        self.parser = parser

    def parse(self, args: list[str]) -> Self:
        """
        Parse command-line arguments and populate the Arguments object.

        Parameters
        ----------
        args : list[str]
            List of command-line arguments to parse. If empty, shows help.

        Returns
        -------
        Self
            Self with parsed argument values set as attributes.

        Notes
        -----
        If --log is specified, verbosity is automatically set to at least 1.
        Drop values are converted to a set to avoid duplicates.
        """
        parser = self.parser
        del self.parser
        parser.parse_args(args=args or ["-h"], namespace=self)

        # If --log is specified, ensure verbosity is at least 1
        if self.log:
            self.verbosity = max(self.verbosity, 1)

        # Convert drop list to set to avoid duplicates
        if self.drop is not None:
            self.drop = set(self.drop)

        return self

    def __str__(self) -> str:
        """
        Return a string representation of the Arguments object.

        Returns
        -------
        str
            String showing arguments stored.
        """
        return f"Arguments({str(self.__dict__)})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Arguments object.

        Returns
        -------
        str
            String showing arguments stored.
        """
        return f"Arguments({repr(self.__dict__)})"

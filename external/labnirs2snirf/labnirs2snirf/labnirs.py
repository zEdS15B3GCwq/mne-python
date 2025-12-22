"""
Functions related to reading data from LabNIRS files.
"""

import logging
import re
from collections.abc import Collection
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from . import model
from .error import Labnirs2SnirfError

log = logging.getLogger(__name__)


class LabNirsReadError(Labnirs2SnirfError):
    """Custom error class for LabNIRS reading errors."""


# Constants

# (tab-separated) data table starts on line 36 (counting from 1)
DATA_START_LINE: Final[int] = 36

# regexp patterns to extract fields from the header & to verify correct format
LINE_PATTERNS: Final[dict[str, str]] = {
    "top_line": rf"^ \[File Information\]\s*\[Data Line\]\t(?P<data_start_line>{DATA_START_LINE})\s*$",
    "version": r"^[^\t]*\t[^\t]*\tVersion\t11\.0$",
    "headertype": r"^[^\t]*\t[^\t]*\t\[HeaderType\]\t11\.0/11\.0$",
    "id": r"^ID\t(?P<id>[^\t]*)\t.*$",
    "measurement_datetime": r"^Measured Date\t(?P<date>[\d/]+) (?P<time>[\d:]+)\s*$",
    "name": r"^Name\t(?P<name>[^\t]*)\t.*$",
    "comment": r"^Comment\t(?P<comment>.*)$",
    "channel_pairs": r"^(?P<channel_pairs>(?>\(\d+,\d+\))+)$",
}


def read_labnirs(
    data_file: Path,
    keep_category: str = "all",
    drop_subtype: Collection[str] | None = None,
) -> model.Nirs:
    """
    Read and process a LabNIRS data file and returns a NIRS data model.

    Parameters
    ----------
    data_file : Path
        Path to the LabNIRS data file.
        File is expected to be in the format exported by the LabNIRS software,
        with 35 lines of header and a version number/header type of 11.0.
    keep_category : "hb"| "raw" | "all", default="all"
        Data category to include in the output. "Raw" means raw voltage data,
        "hb" means haemoglobin data. If "all", both categories are included.
    drop_subtype : Collection[str] | None, default=None
        Set or list of data types and/or wavelengths to drop from the data.
        Hb data types are: "hbr", "hbo" and "hbt".
        Wavelengths should be convertible to integer.
        All included if None.

    Returns
    -------
    model.Nirs
        A NIRS object containing most data required by the SNIRF specification.

    Notes
    -----
    This function reads experiment data and metadata from .txt files exported by
    the LabNIRS software. It expects a 35-line header with specific formatting
    (and version 11.0). Only the top line (containing header length), the
    presence of channel pairs on line 33, and the presence of data columns are
    enforced. Other validation failures only raise a warning, but errors may
    still occur.

    LabNIRS can export files with both raw and Hb data, depending on the options
    selected. The ``keep_category`` parameter controls which of the two is
    retained in the output. The ``drop_subtype`` parameter can be used to
    further exclude specific wavelengths or Hb data types from the output.

    By default, all data is included, which may not be desirable when the goal
    is to import the .snirf file to a NIRS analyser tool such as MNE, as these
    tools may not support files with both raw and Hb data present, may not need
    HbT, or may not be able to handle more than 2 wavelengths. For MNE, for
    example, it would be sensible to either include ``raw`` and drop one
    wavelength of the 3, or to include ``hb`` and drop ``HbT``.

    For reasons of compatibility with other software, a list of wavelengths is
    preserved even for Hb data. Dropped wavelengths are not included in the
    list. For Hb data, the wavelength indices are set to 0 for each data
    channel. NB that this is an invalid index.

    Since the labNIRS files don't store coordinates, probe positions are all set
    to (0, 0, 0). Positions can be read from file using the ``--layout`` option.
    Probe labels are based on actual numbers in the source file, however,
    the position matrices are contiguous and skip over any missing probe numbers.
    E.g. if there are sources 1 and 3, then the source position matrix will have
    2 rows, with source 1 at index 0 and source 3 at index 1, and the labels
    will be S1 and S3 respectively.
    """

    ###########################
    # Validate input parameters
    ###########################

    log.info("Validating input parameters")
    log.debug(
        "Parameters: data_file=%s, keep_category=%s, drop_subtype=%s",
        data_file,
        keep_category,
        drop_subtype,
    )
    if not isinstance(keep_category, str):
        raise LabNirsReadError("Invalid parameters: 'keep_category' must be a string.")
    keep_category = keep_category.lower()
    if keep_category not in ("hb", "raw", "all"):
        raise LabNirsReadError(
            f"Invalid parameters: 'keep_category': must be one of 'hb', 'raw', or 'all', got {keep_category}.",
        )
    if drop_subtype is not None:
        if not (
            isinstance(drop_subtype, Collection)
            and all(isinstance(x, str) for x in drop_subtype)
        ):
            raise LabNirsReadError(
                "Invalid parameters: 'drop_subtype' must be a collection of strings or None.",
            )
        drop_subtype = {x.lower() for x in drop_subtype}
        if not all(x in {"hbo", "hbr", "hbt"} or x.isdigit() for x in drop_subtype):
            raise LabNirsReadError(
                "Invalid parameters: 'drop_subtype' can only contain 'hbo', 'hbr', 'hbt', or wavelength integers.",
            )
    if not data_file.exists():
        log.error("Data file not found: %s", data_file)
        raise LabNirsReadError(f"Data file not found: {data_file}")

    ##########################
    # Read & verify the header
    ##########################

    log.info("Reading and validating header")
    header = _read_header(data_file)

    #########################
    # Parse channels & probes
    #########################

    log.info("Parsing channel pairs and probe information")

    # parse channel pairs
    channels = (
        pl.DataFrame(
            data=[
                (int(x), int(y)) for x, y in re.findall(r"\((\d+),(\d+)\)", header[32])
            ],
            schema=[("source", pl.UInt32), ("detector", pl.UInt32)],
            orient="row",
        )
        .with_row_index(name="channel", offset=1)
        # add probe indices; order is closest to probe numbers in file; missing probes are skipped over
        .with_columns(
            pl.col("source").rank(method="dense").alias("source_index"),
            pl.col("detector").rank(method="dense").alias("detector_index"),
        )
    )
    log.debug(
        "Channel pairs: %s",
        [
            f"{row['source']}-{row['detector']}"
            for row in channels.iter_rows(named=True)
        ],
    )

    # Extract source and detector indices, add labels (Si, Di)
    sources = (
        channels.select(
            pl.col("source").alias("number"),
            pl.col("source_index").alias("index"),
        )
        .unique("number")
        .with_columns(
            # pl.lit("source").alias("type").cast(pl.Categorical()),
            pl.concat_str(pl.lit("S"), pl.col("number")).alias("label"),
        )
        .drop("number")
        .sort("index")
    )
    log.debug("Sources: %s", sources["label"].to_list())

    detectors = (
        channels.select(
            pl.col("detector").alias("number"),
            pl.col("detector_index").alias("index"),
        )
        .unique("number")
        .with_columns(
            # pl.lit("detector").alias("type").cast(pl.Categorical()),
            pl.concat_str(pl.lit("D"), pl.col("number")).alias("label"),
        )
        .drop("number")
        .sort("index")
    )
    log.debug("Detectors: %s", detectors["label"].to_list())

    log.info(
        "Found %d channels, %d sources, %d detectors",
        len(channels),
        len(sources),
        len(detectors),
    )

    #######################
    # Parse column metadata
    #######################

    log.info("Parsing column metadata and data structure")

    # parse and transform column names to conform with naming in model
    column_names_line1 = (
        header[33]
        .lower()
        .replace(" ", "")
        .replace("ch-", "")
        .replace("\n", "")
        .split("\t")
    )
    column_names_line2 = (
        header[34]
        .lower()
        .replace(" ", "")
        .replace("time(sec)", "time")
        .replace("deoxyhb", "hbr")
        .replace("oxyhb", "hbo")
        .replace("totalhb", "hbt")
        .replace("abs", "")
        .replace("nm", "")
        .replace("\n", "")
        .split("\t")
    )

    # name, type, etc. information about all data columns in the experiment file
    columns = (
        pl.DataFrame(
            data=[
                [
                    # column name (e.g. time, mark, 1-hbr, 2-870)
                    f"{int(a)}-{b}" if a else b,
                    # channel number (1, 2, ...), None for non-channel metadata like time
                    int(a) if a.isdigit() else None,
                    # data category (meta, raw (voltage), hb)
                    "meta" if a == "" else "raw" if b.isdigit() else "hb",
                    # subtype (hbr, hbo, hbt, wavelength)
                    b if a != "" else None,
                    # wavelength as string (e.g. 870, 830), None for non-wavelength or meta columns
                    # b if b.isdigit() else None,
                    # wavelength as integer (e.g. 870, 830), None for non-wavelength or meta columns
                    int(b) if b.isdigit() else None,
                ]
                for a, b in zip(column_names_line1, column_names_line2)
            ],
            schema=pl.Schema(
                [
                    ("name", pl.String),
                    ("channel", pl.Int32),
                    ("category", pl.Enum(["meta", "raw", "hb"])),
                    ("subtype", pl.Categorical()),
                    # ("wavelength_str", pl.String),
                    ("wavelength", pl.UInt32),
                ],
            ),
            orient="row",
        )
        # index required later for excluding dropped columns from data table
        .with_row_index(name="column")
        # join with source and detector indexes
        .join(
            channels.select(["channel", "source_index", "detector_index"]),
            on="channel",
            how="left",
        )
    )

    column_names = columns["name"].to_list()

    log.debug("Parsed %d columns: %s", len(columns), column_names)

    # Drop user-specified columns based on subtype (e.g. hbo, hbr, a specific wavelength, etc.).
    # This needs to happen before the list of wavelengths is extracted, as dropped wavelengths are
    # not to be included. Things might break if all wavelengths are dropped, but that's up to the
    # user to decide...
    if drop_subtype is not None and len(drop_subtype) > 0:
        log.info("Dropping columns based on subtype filter: %s", drop_subtype)
        initial_count = len(columns)
        columns = columns.filter(
            # parentheses are necessary otherwise Polars thinks that "meta" is a column name
            (pl.col("category") == "meta") | (~pl.col("subtype").is_in(drop_subtype)),
        )
        log.debug(
            "Dropped %d columns based on subtype filter %s. Remaining columns after filtering: %s",
            initial_count - len(columns),
            drop_subtype,
            columns.to_dict(as_series=False),
        )
    else:
        log.info("No column subtypes specified for dropping")

    # Extract set of remaining unique wavelengths, create wavelength indices
    log.info("Extracting unique wavelengths from data columns")
    wavelengths = (
        columns.select("wavelength")
        .drop_nulls()
        .unique()
        .sort("wavelength")
        .with_row_index(name="wavelength_index", offset=1)
    )

    if wavelengths.height == 0:
        log.debug(
            "No wavelengths found in data columns, creating dummy wavelength entry",
        )
        wavelengths = pl.DataFrame({"wavelength": [0], "wavelength_index": [1]})

    wavelength_list = wavelengths["wavelength"].to_list()
    log.debug("Identified %d wavelengths: %s nm", len(wavelength_list), wavelength_list)
    log.debug("Wavelength mapping: %s", wavelengths.to_dict(as_series=False))

    # Add wavelength indices and data types to columns
    log.info("Adding wavelength indices and data types to column metadata")
    columns = columns.join(wavelengths, on="wavelength", how="left").with_columns(
        # Hb data also needs a wavelength index, even if meaningless, so assign 0 to those rows
        pl.when(pl.col("category") == "hb")
        .then(pl.col("wavelength_index").replace(None, 0))
        .otherwise(pl.col("wavelength_index"))
        .alias("wavelength_index"),
        # continuous wave datatype = 1, processed = 99999 according to SNIRF specifications
        pl.when(pl.col("category") == "hb")
        .then(pl.lit(99999))
        .when(pl.col("category") == "raw")
        .then(pl.lit(1))
        .otherwise(None)
        .alias("datatype"),
    )

    # Finally, if the user only wants to keep one data category, keep only that and meta columns.
    # Discarding "raw" destroys wavelength information, that's why it had to be extracted earlier.
    if keep_category != "all":
        log.info("Filtering to keep only '%s' data category", keep_category)
        initial_count = len(columns)
        columns = columns.filter(pl.col("category").is_in(["meta", keep_category]))
        log.debug(
            "Filtered to keep only '%s' and required meta categories: %d columns retained from %d",
            keep_category,
            len(columns),
            initial_count,
        )
    else:
        log.info("Keeping all data categories (keep_category=%s)", keep_category)
    log.debug("Dropping channel and wavelength columns")
    columns = columns.drop(["channel", "wavelength"])
    log.debug("Final columns: %s", columns["name"].to_list())

    ###############################
    # Read experiment data from CSV
    ###############################

    log.info("Reading experiment data from file")

    # read the data table from the experiment file, formatted as CSV
    # keep only time, task, mark, and selected data columns
    data_table = (
        pl.scan_csv(
            data_file,
            has_header=False,
            skip_lines=DATA_START_LINE - 1,
            separator="\t",
            schema=pl.Schema(
                zip(
                    column_names,
                    [pl.String] * len(column_names),
                ),
            ),
        )
        # select only needed columns
        .select(columns["name"].to_list())
        # drop count metadata
        .drop("count")
        # remove whitespace around values
        .select(pl.col(pl.String).str.strip_chars())
        # convert mark to enum, task to uint, and the rest to float
        .cast({"mark": pl.Enum(["0Z", "0", "1"]), "task": pl.UInt32})
        .cast(
            {pl.String: pl.Float64},
        )
        # scan_csv is lazy, need to collect
        .collect()
    )

    log.info(
        "Successfully read data table with %d rows and %d columns",
        len(data_table),
        len(data_table.columns),
    )

    ###########################################
    # Extract information needed for NIRS model
    ###########################################

    log.info("Extracting metadata, data, stimuli, and probe information")

    return model.Nirs(
        metadata=_extract_metadata(header),
        data=[_extract_data(data_table, columns)],
        stim=_extract_stims(data_table),
        probe=_extract_probes(sources, detectors, wavelengths),
    )


def read_probe_pairs(data_file: Path) -> str:  # noqa: F841
    """
    Read the header line containing probe pairs.

    Parameters
    ----------
    data_file : Path
        Path to the LabNIRS data file.
        File is expected to be in the format exported by the LabNIRS software,
        with 35 lines of header and a version number/header type of 11.0.

    Returns
    -------
    str
        String containing probe pairs without leading and trailing whitespace.
        For example: "(1,1)(2,1)...".
    """
    log.info("Reading probe pairs from file: %s", data_file)
    if not data_file.exists():
        raise LabNirsReadError(f"Data file not found: {data_file}")

    header = _read_header(data_file)
    pairs_str = header[32].strip()
    log.debug("Found probe pairs string: %s", pairs_str)
    return pairs_str


def _extract_data(data: pl.DataFrame, columns: pl.DataFrame) -> model.Data:
    """
    Compile data into a model.Data object.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing experimental time series data.
    columns : pl.DataFrame
        DataFrame with column metadata including category, subtype, source/detector indices.

    Returns
    -------
    model.Data
        Data object containing time, data time series, and measurement list.

    Raises
    ------
    LabNirsReadError
        If no data columns are found after filtering.
    """

    def get_label(subtype: str) -> str | None:
        """
        Map subtype to data type label.

        Parameters
        ----------
        subtype : str
            Data subtype identifier (e.g., "hbo", "hbr", "hbt").

        Returns
        -------
        str or None
            Corresponding label ("HbO", "HbR", "HbT") or None if no match.
        """
        match subtype:
            case "hbo":
                return "HbO"
            case "hbr":
                return "HbR"
            case "hbt":
                return "HbT"
            case _:
                return None

    log.info("Extracting experimental data")
    measurementList = [
        model.Measurement(
            sourceIndex=row["source_index"],
            detectorIndex=row["detector_index"],
            dataType=row["datatype"],
            dataTypeIndex=0,
            dataTypeLabel=(get_label(row["subtype"])),
            wavelengthIndex=row["wavelength_index"],
        )
        for row in columns.rows(named=True)
        if row["category"] != "meta"
    ]
    data_columns = columns.filter(pl.col("category") != "meta")["name"].to_list()
    if len(data_columns) == 0:
        raise LabNirsReadError(
            "No data columns found after filtering; cannot extract data.",
        )
    extracted_data = model.Data(
        time=data["time"].to_numpy(),
        dataTimeSeries=data.select(data_columns).to_numpy(),
        measurementList=measurementList,
    )
    log.debug(
        "Extracted data has %d time points (range %.3f - %.3f), %d data channels, and %d MeasurementList entries",
        len(extracted_data.time),
        extracted_data.time[0],
        extracted_data.time[-1],
        extracted_data.dataTimeSeries.shape[1],
        len(extracted_data.measurementList),
    )
    log.debug(
        "Unique data type labels: %s, wavelength indices: %s",
        {m.dataTypeLabel for m in extracted_data.measurementList},
        {m.wavelengthIndex for m in extracted_data.measurementList},
    )
    return extracted_data


def _extract_probes(
    sources: pl.DataFrame,
    detectors: pl.DataFrame,
    wavelengths: pl.DataFrame,
) -> model.Probe:
    """
    Compile probe information into a model.Probe object.

    Parameters
    ----------
    sources : pl.DataFrame
        DataFrame with source indices and labels.
    detectors : pl.DataFrame
        DataFrame with detector indices and labels.
    wavelengths : pl.DataFrame
        DataFrame with wavelength values.

    Returns
    -------
    model.Probe
        Probe object with wavelengths, positions (initialized to zero), and labels.

    Raises
    ------
    LabNirsReadError
        If any of the input dataframes are empty.

    Notes
    -----
    - All positions are set to 0. Locations can be read from file or guessed elsewhere.
    - Probe labels are set according to Si and Di (source, detector respectively),
      where the numbers are the same as the probe numbers in the labNIRS file.
    - Position matrices skip over missing probe numbers, so make sure you use the
      labels to associate actual positions with probes.
    """
    log.info("Extracting probe information")
    if wavelengths.height == 0 or sources.height == 0 or detectors.height == 0:
        raise LabNirsReadError(
            "Cannot extract probe information: wavelength, source, or detector list is empty.",
        )
    probe = model.Probe(
        wavelengths=wavelengths["wavelength"].to_numpy().astype(np.float64),
        sourcePos3D=np.zeros((sources.height, 3), dtype=np.float64),
        detectorPos3D=np.zeros((detectors.height, 3), dtype=np.float64),
        sourceLabels=sources["label"].to_list(),
        detectorLabels=detectors["label"].to_list(),
    )
    log.debug(
        "Extracted probe information: %d wavelengths, %d sources, and %d detectors, %d source labels, %d detector labels",
        len(probe.wavelengths),
        probe.sourcePos3D.shape[0],
        probe.detectorPos3D.shape[0],
        len(probe.sourceLabels) if probe.sourceLabels is not None else 0,
        len(probe.detectorLabels) if probe.detectorLabels is not None else 0,
    )
    return probe


def _extract_metadata(header: list[str]) -> model.Metadata:
    """
    Compile metadata into a model.Metadata object.

    Parameters
    ----------
    header : list[str]
        List of header lines from the LabNIRS file.

    Returns
    -------
    model.Metadata
        Metadata object with subject ID, measurement date/time, and additional fields.

    Raises
    ------
    LabNirsReadError
        If date or time format in header is invalid.

    Notes
    -----
    - Additional patient and study metadata are also stored in a .pat file,
      which is not exported by labNIRS. For now, reading this file is not supported.
    """
    # extract snirf metadata fields from the header
    log.info("Extracting metadata from header")
    # ID may be missing, in which case return empty string
    subject_id = _match_line(LINE_PATTERNS["id"], header).get("id", "")
    measurement_datetime = _match_line(LINE_PATTERNS["measurement_datetime"], header)
    date = measurement_datetime["date"].split("/")
    if len(date) != 3:
        raise LabNirsReadError(
            f"Invalid measurement date format in header: {measurement_datetime['date']}",
        )
    measurement_date = f"{date[0]}-{date[1]:>02}-{date[2]:>02}"
    time = measurement_datetime["time"].split(":")
    if len(time) != 3:
        raise LabNirsReadError(
            f"Invalid measurement time format in header: {measurement_datetime['time']}",
        )
    measurement_time = f"{time[0]:>02}:{time[1]:>02}:{time[2]:>02}"
    additional_fields = dict()
    if (
        len(subject_name := _match_line(LINE_PATTERNS["name"], header).get("name", ""))
        > 0
    ):
        additional_fields["SubjectName"] = subject_name
    if (
        len(comment := _match_line(LINE_PATTERNS["comment"], header).get("comment", ""))
        > 0
    ):
        additional_fields["comment"] = comment
    metadata = model.Metadata(
        SubjectID=subject_id,
        MeasurementDate=measurement_date,
        MeasurementTime=measurement_time,
        additional_fields=additional_fields,
    )
    log.debug(
        "Extracted metadata has subject ID: %s, has date: %s, has time: %s, and has additional fields: %s",
        metadata.SubjectID is not None and metadata.SubjectID != "",
        metadata.MeasurementDate is not None and metadata.MeasurementDate != "",
        metadata.MeasurementTime is not None and metadata.MeasurementTime != "",
        (
            metadata.additional_fields.keys()
            if len(metadata.additional_fields) > 0
            else "none"
        ),
    )
    return metadata


def _extract_stims(data: pl.DataFrame) -> list[model.Stim]:
    """
    Extract stimulus information into a list of model.Stim objects.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing time, task, and mark columns.

    Returns
    -------
    list[model.Stim]
        List of Stim objects, one for each unique task/stimulus type.

    Notes
    -----
    - In case of event-marked tasks, mark is 1 for the event and task contains the task number.
      Event 0Z marks zeroing to baseline.
    - In the output, task name is a string, Z for zeroing and the task number for others.
    - Event-marked operation allows task 0 to be used as a normal event, whereas the .csv file
      saved by the labnirs software doesn't contain information for task 0; the timings are also
      different: timings in the .csv are 1 sample later than in the .txt.
    - I'm uncertain how tasks are marked in other modus operandi, e.g. when tasks are generated
      by the labnirs software.
    - LabNIRS also stores stim information in a .csv file (not exported), which includes duration,
      pre-rest and post-rest periods. For now, reading that file is not supported. This function
      only extracts event onsets from the .txt file.
    """
    log.info("Extracting stimulus information from data")
    task_df = (
        data.lazy()
        .select(["time", "task", "mark"])
        .filter(pl.col("mark") != "0")
        .with_columns(
            pl.when(pl.col("mark") == "0Z")
            .then(pl.lit("Z"))
            .otherwise(pl.col("task").cast(pl.String))
            .alias("task_name"),
        )
        .select(["time", "task_name"])
        .collect()
    )
    log.debug(
        "Extracted task dataframe has %d rows and %d columns",
        task_df.shape[0],
        task_df.shape[1],
    )
    stims = [
        model.Stim(
            name=task,
            data=task_df["time"].filter(task_df["task_name"] == task).to_numpy(),
        )
        for task in task_df["task_name"].unique().sort()
    ]
    log.debug("Found %d stimulus types", len(stims))
    for stim in stims:
        log.debug("Stimulus type '%s' has %d events", stim.name, len(stim.data))
    return stims


def _match_line(pattern: str, lines: list[str]) -> dict[str, str]:
    """
    Match a regexp pattern against each line until a match is found.

    Parameters
    ----------
    pattern : str
        Regular expression pattern with named capture groups.
    lines : list[str]
        List of lines to search through.

    Returns
    -------
    dict[str, str]
        Dictionary of matched groups (empty if no match found).
    """
    log.debug("Matching pattern '%s' against header lines", pattern)
    pat = re.compile(pattern)
    for line in lines:
        m = pat.match(line)
        if m is not None:
            log.debug("Found pattern in line: %s", line.strip())
            return m.groupdict()
    log.debug("Pattern not found in header")
    return dict()


def _read_header(data_file: Path) -> list[str]:
    """
    Read header lines from a LabNIRS file.

    Parameters
    ----------
    data_file : Path
        Path to the LabNIRS data file.

    Returns
    -------
    list[str]
        List of header lines (35 lines expected).

    Raises
    ------
    LabNirsReadError
        If an error occurs while reading the file or if header format is invalid.
    """
    log.info("Reading header lines from file %s", data_file)
    try:
        with open(data_file, encoding="ASCII") as f:
            header = [f.readline() for _ in range(DATA_START_LINE - 1)]
        log.debug(
            "Read header lines: requested %d, read %d lines",
            DATA_START_LINE - 1,
            len(header),
        )
    except Exception as e:
        log.exception("Error reading the header of %s: %s", data_file, e)
        raise LabNirsReadError(f"Error reading the header of {data_file}") from e
    _verify_header_format(header)

    return header


def _verify_header_format(header: list[str]) -> None:
    """
    Verify that the header conforms to expected LabNIRS format.

    Parameters
    ----------
    header : list[str]
        List of header lines to verify.

    Raises
    ------
    LabNirsReadError
        If critical format errors are found (invalid top line or missing channel pairs).

    Notes
    -----
    - Critical errors (top line format, channel pairs) raise exceptions
    - Non-critical issues (version, metadata fields) only log warnings
    """

    log.info("Verifying header format with %d lines", len(header))

    # Critical errors
    # Check exact top line format
    log.debug("Checking for critical header format errors")
    if re.match(LINE_PATTERNS["top_line"], header[0]) is None:
        raise LabNirsReadError(
            f"Critical header format error: invalid top line in header: {header[0].strip()}",
        )
    # Channel pairs are on line 33
    if re.match(LINE_PATTERNS["channel_pairs"], header[32]) is None:
        raise LabNirsReadError(
            f"Critical header format error: channel pairs not found in line 33: {header[32].strip()}. "
            "Expected format: (source,detector)(source,detector)...",
        )

    # Non-critical warnings (may produce errors later)
    # Version number and header type should be "11.0"
    if re.match(LINE_PATTERNS["version"], header[2]) is None:
        log.warning(
            "Version number in line 3 must be '11.0'. Current: %s. Errors may occur.",
            header[2].strip(),
        )
    if re.match(LINE_PATTERNS["headertype"], header[3]) is None:
        log.warning(
            "HeaderType in line 4 must be '11.0/11.0'. Current: %s. Errors may occur.",
            header[3].strip(),
        )
    if re.match(LINE_PATTERNS["id"], header[2]) is None:
        log.warning("Missing ID metadata in line 3: %s", header[2].strip())
    if re.match(LINE_PATTERNS["measurement_datetime"], header[1]) is None:
        log.warning(
            "Missing measurement datetime metadata in line 2: %s",
            header[1].strip(),
        )
    if re.match(LINE_PATTERNS["name"], header[3]) is None:
        log.warning("Missing subject name metadata in line 4: %s", header[3].strip())
    if re.match(LINE_PATTERNS["comment"], header[4]) is None:
        log.warning("Missing comment metadata in line 5: %s", header[4].strip())

    log.debug("Header format verification completed")

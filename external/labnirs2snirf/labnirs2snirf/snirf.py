"""
Functions related to writing NIRS data to SNIRF files using HDF5.
"""

import logging
from pathlib import Path

import h5py  # type: ignore
import numpy as np

from . import model
from .error import Labnirs2SnirfError


class SnirfWriteError(Labnirs2SnirfError):
    """Custom error class for SNIRF writing errors."""


log = logging.getLogger(__name__)


def write_snirf(nirs: model.Nirs, output_file: Path) -> None:
    """
    Write the NIRS data to a SNIRF file.

    Parameters
    ----------
    nirs : model.Nirs
        NIRS data model containing metadata, data, stim and probe information.
    output_file : Path
        Path to the output SNIRF file. A warning is shown if the file does not
        have a ".snirf" extension.

    Notes
    -----
    - Stimulus onsets are available, but durations, pre, and post rest periods
      are not stored in the data exported by LABNIRS. As such, onsets are written
      correctly, durations are set to 0, and rest periods are not included.
      These timings are stored in a separate .csv file by LABNIRS, reading which
      may be implemented in the future.
    - While the SNIRF specification allows for both indexed and non-indexed nirs
      groups (e.g. /nirs, /nirs1, /nirs2, ...), some tools (e.g. MNE) only accept
      a single non-indexed /nirs group. Therefore, we always write it as such.
    - These tools (may) have other restrictions as well, such as expecting only
      a single data group (/nirs/data1), but as far as I am aware, this is not an
      issue for LABNIRS data.
    """

    SPECS_FORMAT_VERSION = "1.1"

    # Warn about incorrect file extension, but continue
    if output_file.suffix != ".snirf":
        log.warning("Output file doesn't have the .snirf extension: %s", output_file)

    log.debug("Writing SNIRF file: %s", output_file)
    with h5py.File(output_file, "w") as f:
        log.debug("Created HDF5 file, writing SNIRF structure")
        # While the SNIRF specification expects the nirs group to be indexed (/nirs1),
        # the use of a non-indexed entry is also allowed if there's only one.
        # Some tools (e.g. MNE) only accept a non-indexed /nirs.
        nirs_group = f.create_group("/nirs")
        f.create_dataset("formatVersion", data=_str_encode(SPECS_FORMAT_VERSION))
        _write_metadata_group(nirs.metadata, nirs_group.create_group("metaDataTags"))
        _write_data_group(nirs.data[0], nirs_group.create_group("data1"))
        _write_probe_group(nirs.probe, nirs_group.create_group("probe"))
        if nirs.stim is not None:
            _write_stim_group(nirs.stim, nirs_group)
        else:
            log.debug("No stimulus data to write")
    log.debug("SNIRF file write completed")


def _str_encode(s: str) -> bytes:
    """
    Encode a string to bytes using UTF-8 encoding.

    Parameters
    ----------
    s : str
        String to encode.

    Returns
    -------
    bytes
        UTF-8 encoded bytes representation of the string.
    """
    return s.encode("utf-8")


def _write_metadata_group(metadata: model.Metadata, group: h5py.Group) -> None:
    """
    Write metadata to a HDF5 group following SNIRF specification.

    Parameters
    ----------
    metadata : model.Metadata
        Metadata object containing subject ID, measurement date/time, units, and additional fields.
    group : h5py.Group
        HDF5 group where metadata will be written (must be /nirs/metaDataTags).

    Raises
    ------
    SnirfWriteError
        If the group path is not /nirs/metaDataTags.
    """
    log.info("Writing metadata entries into %s", group.name)
    if group.name != "/nirs/metaDataTags":
        log.error("Metadata group must be at /nirs/metaDataTags, got %s", group.name)
        raise SnirfWriteError("Metadata group must be at /nirs/metaDataTags")

    group.create_dataset("SubjectID", data=_str_encode(metadata.SubjectID))
    group.create_dataset("MeasurementDate", data=_str_encode(metadata.MeasurementDate))
    group.create_dataset("MeasurementTime", data=_str_encode(metadata.MeasurementTime))
    group.create_dataset("LengthUnit", data=_str_encode(metadata.LengthUnit))
    group.create_dataset("TimeUnit", data=_str_encode(metadata.TimeUnit))
    group.create_dataset("FrequencyUnit", data=_str_encode(metadata.FrequencyUnit))
    for field, text in metadata.additional_fields.items():
        group.create_dataset(field, data=_str_encode(text))


def _write_data_group(data: model.Data, group: h5py.Group) -> None:
    """
    Write experimental data to a HDF5 group following SNIRF specification.

    Parameters
    ----------
    data : model.Data
        Data object containing time, data time series, and measurement list.
    group : h5py.Group
        HDF5 group where data will be written (must be /nirs/data1).

    Raises
    ------
    SnirfWriteError
        If the group path is not /nirs/data1.
    """
    log.info("Writing data entries into %s", group.name)
    if group.name != "/nirs/data1":
        log.error("Data group must be at /nirs/data1, got %s", group.name)
        raise SnirfWriteError("Data group must be at /nirs/data1")

    log.debug("Writing time with %d points", len(data.time))
    group.create_dataset("time", data=data.time, compression="gzip")
    log.debug("Writing dataTimeSeries with shape %s", data.dataTimeSeries.shape)
    group.create_dataset("dataTimeSeries", data=data.dataTimeSeries, compression="gzip")
    log.debug("Writing measurementList with %d entries", len(data.measurementList))
    for i, row in enumerate(data.measurementList, start=1):
        ml = group.create_group(f"measurementList{i}")
        ml.create_dataset("sourceIndex", data=row.sourceIndex, dtype="int32")
        ml.create_dataset("detectorIndex", data=row.detectorIndex, dtype="int32")
        ml.create_dataset("wavelengthIndex", data=row.wavelengthIndex, dtype="int32")
        ml.create_dataset("dataType", data=row.dataType, dtype="int32")
        ml.create_dataset("dataTypeIndex", data=row.dataTypeIndex, dtype="int32")
        if row.dataTypeLabel is not None:
            ml.create_dataset("dataTypeLabel", data=_str_encode(row.dataTypeLabel))


def _write_stim_group(stims: list[model.Stim], group: h5py.Group) -> None:
    """
    Write stimulus information to HDF5 groups following SNIRF specification.

    Parameters
    ----------
    stims : list[model.Stim]
        List of Stim objects containing stimulus names and onset times.
    group : h5py.Group
        HDF5 group where stimuli will be written (must be /nirs).

    Raises
    ------
    SnirfWriteError
        If the group path is not /nirs.

    Notes
    -----
    Stimulus data is written as Nx3 arrays where N is the number of events.
    Column 0 contains onset times, column 2 contains amplitude (set to 1),
    and column 1 (duration) is set to 0 as duration information is not available.
    """
    log.info("Writing stimulus entries into %s", group.name)
    if group.name != "/nirs":
        log.error("Stimulus group must be at /nirs, got %s", group.name)
        raise SnirfWriteError("Stimulus group must be at /nirs")

    for i, stim in enumerate(stims, start=1):
        st = group.create_group(f"stim{i}")
        log.debug(
            "Writing stimulus %02d/%02d: %s with %d data points into %s",
            i,
            len(stims),
            stim.name,
            len(stim.data),
            st.name,
        )
        st.create_dataset("name", data=_str_encode(stim.name))
        d = np.zeros((len(stim.data), 3), dtype=stim.data.dtype)
        d[:, 0] = stim.data
        d[:, 2] = 1
        st.create_dataset("data", data=d, compression="gzip")


def _write_probe_group(probe: model.Probe, group: h5py.Group) -> None:
    """
    Write probe information to a HDF5 group following SNIRF specification.

    Parameters
    ----------
    probe : model.Probe
        Probe object containing wavelengths, source/detector positions, and labels.
    group : h5py.Group
        HDF5 group where probe information will be written (must be /nirs/probe).

    Raises
    ------
    SnirfWriteError
        If the group path is not /nirs/probe.
    """
    log.info("Writing probe information into %s", group.name)
    if group.name != "/nirs/probe":
        log.error("Probe group must be at /nirs/probe, got %s", group.name)
        raise SnirfWriteError("Probe group must be at /nirs/probe")

    log.debug("Writing %d wavelengths", len(probe.wavelengths))
    group.create_dataset("wavelengths", data=probe.wavelengths)

    log.debug("Writing source positions with shape %s", probe.sourcePos3D.shape)
    group.create_dataset("sourcePos3D", data=probe.sourcePos3D, compression="gzip")

    log.debug("Writing detector positions with shape %s", probe.detectorPos3D.shape)
    group.create_dataset("detectorPos3D", data=probe.detectorPos3D, compression="gzip")

    if probe.sourceLabels is not None:
        log.debug("Writing %d source labels", len(probe.sourceLabels))
        group.create_dataset(
            "sourceLabels",
            data=np.array(
                probe.sourceLabels,
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )
    else:
        log.debug("No source labels to write")
    if probe.detectorLabels is not None:
        log.debug("Writing %d detector labels", len(probe.detectorLabels))
        group.create_dataset(
            "detectorLabels",
            data=np.array(
                probe.detectorLabels,
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )
    else:
        log.debug("No detector labels to write")

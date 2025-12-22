"""
NIRS data model definitions for Labnirs --> SNIRF conversion.

Based on the SNIRF specification v1.1.
See https://github.com/fNIRS/snirf/blob/v1.1/snirf_specification.md for details.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True, frozen=True)
class Metadata:
    """
    Metadata container for NIRS measurements following SNIRF specification.

    Attributes
    ----------
    SubjectID : str
        Subject identifier string.
    MeasurementDate : str
        Date of measurement in YYYY-MM-DD format.
    MeasurementTime : str
        Time of measurement in HH:MM:SS format.
    LengthUnit : str, default="m"
        Unit of length measurements (meters).
    TimeUnit : str, default="s"
        Unit of time measurements (seconds).
    FrequencyUnit : str, default="Hz"
        Unit of frequency measurements (Hertz).
    additional_fields : dict[str, str], default=empty dict
        Additional optional metadata fields as key-value pairs.
    """

    SubjectID: str
    MeasurementDate: str
    MeasurementTime: str
    LengthUnit: str = "m"
    TimeUnit: str = "s"
    FrequencyUnit: str = "Hz"
    additional_fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Measurement:
    """
    Measurement channel specification following SNIRF specification.

    Each Measurement describes a single data channel with information about
    the source-detector pair, wavelength, and data type.

    Attributes
    ----------
    sourceIndex : int
        Index of the source optode (1-based).
    detectorIndex : int
        Index of the detector optode (1-based).
    dataType : int
        Type of data: 1 for continuous wave, 99999 for processed (e.g., HbO, HbR).
    dataTypeIndex : int
        Index for grouping measurements of the same type (typically 0).
    wavelengthIndex : int
        Index into the wavelengths array, indicating the wavelength used.
    dataTypeLabel : str or None, default=None
        Optional label for processed data types (e.g., "HbO", "HbR", "HbT").
    """

    sourceIndex: int
    detectorIndex: int
    dataType: int
    dataTypeIndex: int
    wavelengthIndex: int
    dataTypeLabel: str | None = None


@dataclass(slots=True, frozen=True)
class Data:
    """
    Experimental time series data following SNIRF specification.

    Attributes
    ----------
    time : np.ndarray
        1D array of time points in seconds.
        Possible dtypes: float64 (recommended), float32.
    dataTimeSeries : np.ndarray
        2D array (time x channels) of measurement values.
        Possible dtypes: float64 (recommended), float32.
    measurementList : list[Measurement]
        List of Measurement objects describing each data channel.
    """

    time: np.ndarray
    dataTimeSeries: np.ndarray
    measurementList: list[Measurement]


@dataclass(slots=True, frozen=True)
class Probe:
    """
    Probe geometry and configuration following SNIRF specification.

    Attributes
    ----------
    wavelengths : np.ndarray
        1D array of wavelengths in nanometers.
    sourcePos3D : np.ndarray
        2D array (n_sources x 3) of source 3D coordinates [x, y, z].
    detectorPos3D : np.ndarray
        2D array (n_detectors x 3) of detector 3D coordinates [x, y, z].
    sourceLabels : list[str] or None, default=None
        Optional list of source labels (e.g., ["S1", "S2", ...]).
    detectorLabels : list[str] or None, default=None
        Optional list of detector labels (e.g., ["D1", "D2", ...]).
    """

    wavelengths: np.ndarray
    sourcePos3D: np.ndarray
    detectorPos3D: np.ndarray
    sourceLabels: list[str] | None = None
    detectorLabels: list[str] | None = None


@dataclass(slots=True, frozen=True)
class Stim:
    """
    Stimulus/event information following SNIRF specification.

    Attributes
    ----------
    name : str
        Name or identifier for the stimulus condition.
    data : np.ndarray
        1D array of stimulus onset times in seconds.
    """

    name: str
    data: np.ndarray
    # The following field may become necessary in the future is this tool
    # is extended to support .csv files that include pre- and post-rest periods.
    # dataLabels: list[str] | None = None


@dataclass(slots=True, frozen=True)
class Nirs:
    """
    Complete NIRS dataset following SNIRF specification.

    This is the top-level container for a complete NIRS measurement,
    including metadata, experimental data, probe geometry, and stimuli.

    Attributes
    ----------
    metadata : Metadata
        Metadata about the subject and measurement.
    data : list[Data]
        List of Data objects (in this package, it always contains one Data object).
    probe : Probe
        Probe geometry and configuration information.
    stim : list[Stim] or None, default=None
        Optional list of stimulus/event information.
    """

    metadata: Metadata
    data: list[Data]
    probe: Probe
    stim: list[Stim] | None = None

# type: ignore
# pylint: disable=E1101
import logging

import h5py
import numpy as np
import pytest

from labnirs2snirf import model
from labnirs2snirf.snirf import (
    SnirfWriteError,
    _write_data_group,
    _write_metadata_group,
    _write_probe_group,
    _write_stim_group,
    write_snirf,
)

# Test Data Fixtures


@pytest.fixture(name="minimal_nirs")
def fixture_minimal_nirs():
    """Minimal NIRS dataset with only required fields and single data points."""
    metadata = model.Metadata(
        SubjectID="S001",
        MeasurementDate="2024-01-15",
        MeasurementTime="14:30:00",
    )

    time = np.array([0.0], dtype=np.float64)
    data_series = np.array([[1.0]], dtype=np.float64)
    measurement_list = [
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
        ),
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([760.0], dtype=np.float64)
    sourcePos3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    detectorPos3D = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
    )

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=None)


@pytest.fixture(name="full_nirs")
def fixture_full_nirs():
    """Full NIRS dataset with all optional fields populated."""
    metadata = model.Metadata(
        SubjectID="S002",
        MeasurementDate="2024-02-20",
        MeasurementTime="09:15:30",
        LengthUnit="mm",
        TimeUnit="ms",
        FrequencyUnit="MHz",
        additional_fields={
            "StudyName": "Test Study",
            "InstitutionName": "Test Lab",
        },
    )

    time = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    data_series = np.array(
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]],
        dtype=np.float64,
    )
    measurement_list = [
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
            dataTypeLabel="HbO",
        ),
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=2,
            dataType=1,
            dataTypeIndex=1,
            dataTypeLabel="HbR",
        ),
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([760.0, 850.0], dtype=np.float64)
    sourcePos3D = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    detectorPos3D = np.array([[0.5, 1.0, 0.0], [1.5, 1.0, 0.0]], dtype=np.float64)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
        sourceLabels=["S1", "S2"],
        detectorLabels=["D1", "D2"],
    )

    stims = [
        model.Stim(name="TaskA", data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)),
        model.Stim(name="TaskB", data=np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)),
    ]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="large_nirs")
def fixture_large_nirs():
    """Large NIRS dataset with realistic array sizes and long strings."""
    rng = np.random.default_rng(42)

    # Long strings for metadata
    long_subject_id = "S" + "X" * 100
    metadata = model.Metadata(
        SubjectID=long_subject_id,
        MeasurementDate="2024-03-15",
        MeasurementTime="10:00:00",
        additional_fields={
            "StudyName": "A" * 500,
            "Description": "Long description " * 50,
        },
    )

    # Large data arrays: 6000 timepoints, 48 channels
    n_timepoints = 6000
    n_channels = 48
    time = np.linspace(0, 600, n_timepoints, dtype=np.float64)
    data_series = rng.random((n_timepoints, n_channels), dtype=np.float64)
    measurement_list = [
        model.Measurement(
            sourceIndex=(i % 16) + 1,
            detectorIndex=(i // 16) + 1,
            wavelengthIndex=(i % 2) + 1,
            dataType=1,
            dataTypeIndex=1,
            dataTypeLabel=f"Channel{i:03d}",
        )
        for i in range(n_channels)
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    # Large probe arrays: 16 sources, 3 detectors
    wavelengths = np.array([760.0, 850.0], dtype=np.float64)
    sourcePos3D = rng.random((16, 3), dtype=np.float64) * 10
    detectorPos3D = rng.random((3, 3), dtype=np.float64) * 10
    sourceLabels = [f"Source_{i:03d}" for i in range(16)]
    detectorLabels = [f"Detector_{i:03d}" for i in range(3)]
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
        sourceLabels=sourceLabels,
        detectorLabels=detectorLabels,
    )

    # Multiple stimuli with many events
    stims = [
        model.Stim(
            name=f"Task{i}",
            data=np.linspace(i * 100, i * 100 + 500, 50, dtype=np.float64),
        )
        for i in range(5)
    ]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="unicode_nirs")
def fixture_unicode_nirs(minimal_nirs):
    """NIRS dataset with unicode and special characters in string fields."""
    metadata = model.Metadata(
        SubjectID="Sübject™-006_Test",
        MeasurementDate=minimal_nirs.metadata.MeasurementDate,
        MeasurementTime=minimal_nirs.metadata.MeasurementTime,
        additional_fields={
            "StudyName": "fNÍRS Study™ with émojis 🧠",
            "Comment": "Temperature: 22°C, Location: Room #3",
        },
    )

    measurement_list = [
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
            dataTypeLabel="HbÖ",
        ),
    ]
    data = model.Data(
        time=minimal_nirs.data[0].time,
        dataTimeSeries=minimal_nirs.data[0].dataTimeSeries,
        measurementList=measurement_list,
    )

    probe = model.Probe(
        wavelengths=minimal_nirs.probe.wavelengths,
        sourcePos3D=minimal_nirs.probe.sourcePos3D,
        detectorPos3D=minimal_nirs.probe.detectorPos3D,
        sourceLabels=["S-1™"],
        detectorLabels=["Det-α"],
    )

    stims = [model.Stim(name="Täsk™ 第一", data=np.array([1.0], dtype=np.float64))]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="mixed_types_nirs")
def fixture_mixed_types_nirs(large_nirs):
    """NIRS dataset with mixed dataType values and various field combinations."""
    # Reuse large_nirs metadata with custom units
    metadata = model.Metadata(
        SubjectID=large_nirs.metadata.SubjectID,
        MeasurementDate=large_nirs.metadata.MeasurementDate,
        MeasurementTime=large_nirs.metadata.MeasurementTime,
        LengthUnit="cm",
        TimeUnit="ms",
        FrequencyUnit="kHz",
        additional_fields=large_nirs.metadata.additional_fields,
    )

    # Mixed dataType and dataTypeIndex values
    rng = np.random.default_rng(43)
    time = np.linspace(0, 10, 100, dtype=np.float64)
    n_channels = 12
    data_series = rng.random((100, n_channels), dtype=np.float64)
    measurement_list = [
        # Raw data (dataType=1)
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
        ),
        model.Measurement(
            sourceIndex=1,
            detectorIndex=2,
            wavelengthIndex=2,
            dataType=1,
            dataTypeIndex=1,
        ),
        model.Measurement(
            sourceIndex=2,
            detectorIndex=2,
            wavelengthIndex=0,
            dataType=99999,
            dataTypeIndex=1,
            dataTypeLabel="HbO",
        ),
        model.Measurement(
            sourceIndex=2,
            detectorIndex=2,
            wavelengthIndex=0,
            dataType=99999,
            dataTypeIndex=2,
            dataTypeLabel="HbR",
        ),
        model.Measurement(
            sourceIndex=2,
            detectorIndex=2,
            wavelengthIndex=0,
            dataType=99999,
            dataTypeIndex=3,
            dataTypeLabel="HbT",
        ),
        # More raw data
        model.Measurement(
            sourceIndex=3,
            detectorIndex=3,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
        ),
        model.Measurement(
            sourceIndex=3,
            detectorIndex=3,
            wavelengthIndex=2,
            dataType=1,
            dataTypeIndex=1,
        ),
        # Additional processed types
        model.Measurement(
            sourceIndex=4,
            detectorIndex=4,
            wavelengthIndex=0,
            dataType=99999,
            dataTypeIndex=4,
            dataTypeLabel="StO2",
        ),
        model.Measurement(
            sourceIndex=5,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=2,
            dataTypeIndex=1,
        ),
        model.Measurement(
            sourceIndex=5,
            detectorIndex=2,
            wavelengthIndex=2,
            dataType=2,
            dataTypeIndex=1,
        ),
        model.Measurement(
            sourceIndex=6,
            detectorIndex=3,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=2,
            dataTypeLabel="alternate",
        ),
        model.Measurement(
            sourceIndex=6,
            detectorIndex=4,
            wavelengthIndex=2,
            dataType=1,
            dataTypeIndex=2,
        ),
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    # Reuse large probe data
    probe = large_nirs.probe

    # Reuse large stim data
    stims = large_nirs.stim

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="float32_nirs")
def fixture_float32_nirs(minimal_nirs):
    """NIRS dataset with float32 data types in numeric fields."""
    metadata = minimal_nirs.metadata

    time = np.array([0.0], dtype=np.float32)
    data_series = np.array([[1.0]], dtype=np.float32)
    measurement_list = minimal_nirs.data[0].measurementList
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([760.0], dtype=np.float32)
    sourcePos3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    detectorPos3D = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
    )

    stims = [model.Stim(name="Task", data=np.array([1.0], dtype=np.float32))]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="empty_nirs")
def fixture_empty_nirs():
    """NIRS dataset with empty values where possible."""
    metadata = model.Metadata(
        SubjectID="",
        MeasurementDate="",
        MeasurementTime="",
        LengthUnit="",
        TimeUnit="",
        FrequencyUnit="",
        additional_fields={"EmptyField": "", "AnotherEmpty": ""},
    )

    time = np.array([], dtype=np.float64)
    data_series = np.empty((0, 0), dtype=np.float64)
    measurement_list = []
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([], dtype=np.float64)
    sourcePos3D = np.empty((0, 3), dtype=np.float64)
    detectorPos3D = np.empty((0, 3), dtype=np.float64)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
        sourceLabels=[],
        detectorLabels=[],
    )

    stims = []

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="negative_nirs")
def fixture_negative_nirs():
    """NIRS dataset with negative values in numeric fields."""
    metadata = model.Metadata(
        SubjectID="S001",
        MeasurementDate="2024-01-15",
        MeasurementTime="14:30:00",
    )

    time = np.array([-1.0, -0.5, 0.0], dtype=np.float64)
    data_series = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float64)
    measurement_list = [
        model.Measurement(
            sourceIndex=-1,
            detectorIndex=-2,
            wavelengthIndex=-3,
            dataType=-4,
            dataTypeIndex=-5,
        ),
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([-760.0, -850.0], dtype=np.float64)
    sourcePos3D = np.array([[-1.0, -2.0, -3.0]], dtype=np.float64)
    detectorPos3D = np.array([[-0.5, -1.5, -2.5]], dtype=np.float64)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
    )

    stims = [
        model.Stim(name="Task", data=np.array([-1.0, -2.0, -3.0], dtype=np.float64)),
    ]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="unsorted_nirs")
def fixture_unsorted_nirs(minimal_nirs):
    """NIRS dataset with unsorted timepoints."""
    metadata = minimal_nirs.metadata

    time = np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float64)
    data_series = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    measurement_list = minimal_nirs.data[0].measurementList
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    probe = minimal_nirs.probe

    stims = [
        model.Stim(name="Task", data=np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float64)),
    ]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="zero_nirs")
def fixture_zero_nirs():
    """NIRS dataset with zero values in numeric fields."""
    metadata = model.Metadata(
        SubjectID="S001",
        MeasurementDate="2024-01-15",
        MeasurementTime="14:30:00",
    )

    time = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    data_series = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    measurement_list = [
        model.Measurement(
            sourceIndex=0,
            detectorIndex=0,
            wavelengthIndex=0,
            dataType=0,
            dataTypeIndex=0,
        ),
    ]
    data = model.Data(
        time=time,
        dataTimeSeries=data_series,
        measurementList=measurement_list,
    )

    wavelengths = np.array([0.0], dtype=np.float64)
    sourcePos3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    detectorPos3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    probe = model.Probe(
        wavelengths=wavelengths,
        sourcePos3D=sourcePos3D,
        detectorPos3D=detectorPos3D,
    )

    stims = [model.Stim(name="Task", data=np.array([0.0, 0.0], dtype=np.float64))]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


@pytest.fixture(name="whitespace_nirs")
def fixture_whitespace_nirs(minimal_nirs):
    """NIRS dataset with whitespace in string fields."""
    metadata = model.Metadata(
        SubjectID="  S001  ",
        MeasurementDate="  2024-01-15  ",
        MeasurementTime="  14:30:00  ",
        additional_fields={
            "MultiLine": "Line1\nLine2\nLine3",
            "Tabs": "Column1\tColumn2\tColumn3",
            "Spaces": "   Leading and trailing   ",
        },
    )

    measurement_list = [
        model.Measurement(
            sourceIndex=1,
            detectorIndex=1,
            wavelengthIndex=1,
            dataType=1,
            dataTypeIndex=1,
            dataTypeLabel="  HbO  ",
        ),
    ]
    data = model.Data(
        time=minimal_nirs.data[0].time,
        dataTimeSeries=minimal_nirs.data[0].dataTimeSeries,
        measurementList=measurement_list,
    )

    probe = model.Probe(
        wavelengths=minimal_nirs.probe.wavelengths,
        sourcePos3D=minimal_nirs.probe.sourcePos3D,
        detectorPos3D=minimal_nirs.probe.detectorPos3D,
        sourceLabels=["  S1  "],
        detectorLabels=["  D1  "],
    )

    stims = [model.Stim(name="  Task  ", data=minimal_nirs.data[0].time)]

    return model.Nirs(metadata=metadata, data=[data], probe=probe, stim=stims)


# Validator Helper Functions


def validate_metadata_group(metadata: model.Metadata, file_handle: h5py.File) -> None:
    """Validate that metadata in file matches the source metadata object."""
    meta_group = file_handle["/nirs/metaDataTags"]

    # Check required fields
    assert "SubjectID" in meta_group
    assert "MeasurementDate" in meta_group
    assert "MeasurementTime" in meta_group
    assert "LengthUnit" in meta_group
    assert "TimeUnit" in meta_group
    assert "FrequencyUnit" in meta_group
    assert meta_group["SubjectID"][()].decode("utf-8") == metadata.SubjectID
    assert meta_group["MeasurementDate"][()].decode("utf-8") == metadata.MeasurementDate
    assert meta_group["MeasurementTime"][()].decode("utf-8") == metadata.MeasurementTime
    assert meta_group["LengthUnit"][()].decode("utf-8") == metadata.LengthUnit
    assert meta_group["TimeUnit"][()].decode("utf-8") == metadata.TimeUnit
    assert meta_group["FrequencyUnit"][()].decode("utf-8") == metadata.FrequencyUnit

    # Check additional fields if present
    if metadata.additional_fields:
        for key, value in metadata.additional_fields.items():
            assert key in meta_group
            assert meta_group[key][()].decode("utf-8") == value

    # Count expected fields: 6 required + additional
    expected_count = 6 + len(metadata.additional_fields)
    assert len(meta_group.keys()) == expected_count


def validate_data_group(data: model.Data, file_handle: h5py.File) -> None:
    """Validate that data in file matches the source data object."""
    data_group = file_handle["/nirs/data1"]

    # Check time array
    assert "time" in data_group
    assert data_group["time"].dtype == data.time.dtype
    assert np.allclose(data_group["time"][:], data.time, atol=1e-9)
    if len(data.time) > 0:
        assert data_group["time"].compression == "gzip"

    # Check dataTimeSeries
    assert "dataTimeSeries" in data_group
    assert data_group["dataTimeSeries"].shape == data.dataTimeSeries.shape
    assert data_group["dataTimeSeries"].dtype == data.dataTimeSeries.dtype
    assert np.allclose(data_group["dataTimeSeries"][:], data.dataTimeSeries, atol=1e-9)
    if len(data.dataTimeSeries) > 0:
        assert data_group["dataTimeSeries"].compression == "gzip"

    # Check measurement list
    ml_count = len([k for k in data_group.keys() if k.startswith("measurementList")])
    assert ml_count == len(data.measurementList)

    for i, measurement in enumerate(data.measurementList, start=1):
        ml_key = f"measurementList{i}"
        assert ml_key in data_group
        ml_group = data_group[ml_key]

        # Check required measurement fields
        assert ml_group["sourceIndex"][()] == measurement.sourceIndex
        assert ml_group["detectorIndex"][()] == measurement.detectorIndex
        assert ml_group["wavelengthIndex"][()] == measurement.wavelengthIndex
        assert ml_group["dataType"][()] == measurement.dataType
        assert ml_group["dataTypeIndex"][()] == measurement.dataTypeIndex

        # Check data types are int32
        assert ml_group["sourceIndex"].dtype == np.dtype("int32")
        assert ml_group["detectorIndex"].dtype == np.dtype("int32")
        assert ml_group["wavelengthIndex"].dtype == np.dtype("int32")
        assert ml_group["dataType"].dtype == np.dtype("int32")
        assert ml_group["dataTypeIndex"].dtype == np.dtype("int32")

        # Check optional dataTypeLabel
        if measurement.dataTypeLabel is not None:
            assert "dataTypeLabel" in ml_group
            assert (
                ml_group["dataTypeLabel"][()].decode("utf-8")
                == measurement.dataTypeLabel
            )
        else:
            assert "dataTypeLabel" not in ml_group


def validate_probe_group(probe: model.Probe, file_handle: h5py.File) -> None:
    """Validate that probe in file matches the source probe object."""
    probe_group = file_handle["/nirs/probe"]

    # Check wavelengths
    assert "wavelengths" in probe_group
    assert probe_group["wavelengths"].shape == probe.wavelengths.shape
    assert probe_group["wavelengths"].dtype == probe.wavelengths.dtype
    assert np.allclose(probe_group["wavelengths"][:], probe.wavelengths, atol=1e-9)

    # Check source positions
    assert "sourcePos3D" in probe_group
    assert probe_group["sourcePos3D"].shape == probe.sourcePos3D.shape
    assert probe_group["sourcePos3D"].dtype == probe.sourcePos3D.dtype
    assert np.allclose(probe_group["sourcePos3D"][:], probe.sourcePos3D, atol=1e-9)

    # Check detector positions
    assert "detectorPos3D" in probe_group
    assert probe_group["detectorPos3D"].shape == probe.detectorPos3D.shape
    assert probe_group["detectorPos3D"].dtype == probe.detectorPos3D.dtype
    assert np.allclose(probe_group["detectorPos3D"][:], probe.detectorPos3D, atol=1e-9)

    # Check compression for non-empty arrays
    if probe.sourcePos3D.shape[0] > 0:
        assert probe_group["sourcePos3D"].compression == "gzip"
    if probe.detectorPos3D.shape[0] > 0:
        assert probe_group["detectorPos3D"].compression == "gzip"

    # Check source labels (can be empty list, see fixture_empty_nirs)
    if probe.sourceLabels is not None:  #  and len(probe.sourceLabels) > 0
        assert "sourceLabels" in probe_group
        src_labels = [label.decode("utf-8") for label in probe_group["sourceLabels"]]
        assert src_labels == probe.sourceLabels
    else:
        assert "sourceLabels" not in probe_group

    # Check detector labels
    if probe.detectorLabels is not None:  # and len(probe.detectorLabels) > 0:
        assert "detectorLabels" in probe_group
        det_labels = [label.decode("utf-8") for label in probe_group["detectorLabels"]]
        assert det_labels == probe.detectorLabels
    else:
        assert "detectorLabels" not in probe_group


def validate_stim_group(stims: list[model.Stim] | None, file_handle: h5py.File) -> None:
    """Validate that stimuli in file match the source stim objects."""
    nirs_group = file_handle["/nirs"]

    # Count stim groups in file
    stim_count = len([k for k in nirs_group.keys() if k.startswith("stim")])

    # If no stims provided or empty list
    if stims is None or len(stims) == 0:
        assert stim_count == 0
        return

    # Check that we have the right number of stim groups
    assert stim_count == len(stims)

    # Validate each stimulus
    for i, stim in enumerate(stims, start=1):
        stim_key = f"stim{i}"
        assert stim_key in nirs_group
        stim_group = nirs_group[stim_key]

        # Check name
        assert "name" in stim_group
        assert stim_group["name"][()].decode("utf-8") == stim.name

        # Check data
        assert "data" in stim_group
        stim_data = stim_group["data"]

        # Shape should be (N, 3)
        assert stim_data.shape == (len(stim.data), 3)

        # First column should match input onset times
        assert np.allclose(stim_data[:, 0], stim.data, atol=1e-9)

        # Second column (duration) should be 0
        assert np.allclose(stim_data[:, 1], 0, atol=1e-9)

        # Third column (value) should be 1
        assert np.allclose(stim_data[:, 2], 1, atol=1e-9)

        # Check data type matches input
        assert stim_data.dtype == stim.data.dtype

        # Check compression for non-empty arrays
        if len(stim.data) > 0:
            assert stim_data.compression == "gzip"


def validate_snirf_file(nirs: model.Nirs, file_handle: h5py.File) -> None:
    """Validate entire SNIRF file structure and contents."""
    # Check format version
    assert "formatVersion" in file_handle
    assert file_handle["formatVersion"][()].decode("utf-8") == "1.1"

    # Check main /nirs group exists
    assert "/nirs" in file_handle
    nirs_group = file_handle["/nirs"]

    # Validate metadata
    assert "metaDataTags" in nirs_group
    validate_metadata_group(nirs.metadata, file_handle)

    # Validate data (only data1 is supported)
    assert "data1" in nirs_group
    assert len(nirs.data) >= 1
    validate_data_group(nirs.data[0], file_handle)

    # Validate probe
    assert "probe" in nirs_group
    validate_probe_group(nirs.probe, file_handle)

    # Validate stimuli
    validate_stim_group(nirs.stim, file_handle)


class TestWriteMetadataGroup:
    @pytest.mark.parametrize(
        "nirs_fixture",
        [
            "minimal_nirs",
            "full_nirs",
            "large_nirs",
            "unicode_nirs",
            "mixed_types_nirs",
            "float32_nirs",
            "empty_nirs",
            "negative_nirs",
            "unsorted_nirs",
            "zero_nirs",
            "whitespace_nirs",
        ],
    )
    def test_write_metadata_group_valid_datasets_succeeds(
        self,
        tmp_path,
        nirs_fixture,
        request,
    ):
        """Test writing metadata with various valid datasets."""
        nirs = request.getfixturevalue(nirs_fixture)
        output_file = tmp_path / f"test_metadata_{nirs_fixture}.h5"

        with h5py.File(output_file, "w") as f:
            nirs_group = f.create_group("/nirs")
            metadata_group = nirs_group.create_group("metaDataTags")
            _write_metadata_group(nirs.metadata, metadata_group)

        with h5py.File(output_file, "r") as f:
            validate_metadata_group(nirs.metadata, f)

    def test_write_metadata_group_with_logging_succeeds(
        self,
        tmp_path,
        minimal_nirs,
        caplog,
    ):
        """Test that metadata writing produces correct log messages."""
        output_file = tmp_path / "test_metadata_logging.h5"

        with caplog.at_level(logging.DEBUG):
            with h5py.File(output_file, "w") as f:
                nirs_group = f.create_group("/nirs")
                metadata_group = nirs_group.create_group("metaDataTags")
                _write_metadata_group(minimal_nirs.metadata, metadata_group)

        # Check logging output
        assert any(
            "Writing metadata entries into /nirs/metaDataTags" in record.message
            for record in caplog.records
        )

    def test_write_metadata_group_wrong_path_fails(self, tmp_path, caplog):
        """Test that wrong group path raises SnirfWriteError with proper logging."""
        output_file = tmp_path / "test_metadata_wrong_path.h5"

        metadata = model.Metadata(
            SubjectID="S007",
            MeasurementDate="2024-07-25",
            MeasurementTime="15:20:00",
        )

        with caplog.at_level(logging.ERROR):
            with h5py.File(output_file, "w") as f:
                wrong_group = f.create_group("/wrong/path")

                with pytest.raises(
                    SnirfWriteError,
                    match="Metadata group must be at /nirs/metaDataTags",
                ):
                    _write_metadata_group(metadata, wrong_group)

        # Check error logging
        assert any(
            "Metadata group must be at /nirs/metaDataTags" in record.message
            for record in caplog.records
        )
        assert any(record.levelname == "ERROR" for record in caplog.records)


class TestWriteDataGroup:
    @pytest.mark.parametrize(
        "nirs_fixture",
        [
            "minimal_nirs",
            "full_nirs",
            "large_nirs",
            "unicode_nirs",
            "mixed_types_nirs",
            "float32_nirs",
            "empty_nirs",
            "negative_nirs",
            "unsorted_nirs",
            "zero_nirs",
            "whitespace_nirs",
        ],
    )
    def test_write_data_group_valid_datasets_succeeds(
        self,
        tmp_path,
        nirs_fixture,
        request,
    ):
        """Test writing data group with various valid datasets."""
        nirs = request.getfixturevalue(nirs_fixture)
        output_file = tmp_path / f"test_data_{nirs_fixture}.h5"

        with h5py.File(output_file, "w") as f:
            nirs_group = f.create_group("/nirs")
            data_group = nirs_group.create_group("data1")
            _write_data_group(nirs.data[0], data_group)

        with h5py.File(output_file, "r") as f:
            validate_data_group(nirs.data[0], f)

    def test_write_data_group_logging_output_succeeds(
        self,
        tmp_path,
        full_nirs,
        caplog,
    ):
        """Test that data writing produces correct log messages at various levels."""
        output_file = tmp_path / "test_data_logging.h5"

        with caplog.at_level(logging.DEBUG):
            with h5py.File(output_file, "w") as f:
                nirs_group = f.create_group("/nirs")
                data_group = nirs_group.create_group("data1")
                _write_data_group(full_nirs.data[0], data_group)

        # Check info level logging with exact path
        assert any(
            record.message == "Writing data entries into /nirs/data1"
            for record in caplog.records
        )

        # Check debug level logging with actual values from fixture
        assert any(
            record.message == f"Writing time with {len(full_nirs.data[0].time)} points"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing dataTimeSeries with shape {full_nirs.data[0].dataTimeSeries.shape}"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing measurementList with {len(full_nirs.data[0].measurementList)} entries"
            for record in caplog.records
        )

    def test_write_data_group_wrong_path_fails(self, tmp_path, minimal_nirs):
        """Test that wrong group path raises SnirfWriteError with proper logging."""
        output_file = tmp_path / "test_data_wrong_path.h5"

        with pytest.raises(SnirfWriteError, match="Data group must be at /nirs/data1"):
            with h5py.File(output_file, "w") as f:
                wrong_group = f.create_group("/wrong/path/data1")
                _write_data_group(minimal_nirs.data[0], wrong_group)


class TestWriteStimGroup:
    @pytest.mark.parametrize(
        "nirs_fixture",
        [
            # "minimal_nirs",  # minimal_nirs has no stimuli and isn't invoked by write_snirf
            "full_nirs",
            "large_nirs",
            "unicode_nirs",
            "mixed_types_nirs",
            "float32_nirs",
            "empty_nirs",
            "negative_nirs",
            "unsorted_nirs",
            "zero_nirs",
            "whitespace_nirs",
        ],
    )
    def test_write_stim_group_valid_datasets_succeeds(
        self,
        tmp_path,
        nirs_fixture,
        request,
    ):
        """Test writing stim group with various valid datasets."""
        nirs = request.getfixturevalue(nirs_fixture)
        output_file = tmp_path / f"test_stim_{nirs_fixture}.h5"

        with h5py.File(output_file, "w") as f:
            nirs_group = f.create_group("/nirs")
            _write_stim_group(nirs.stim, nirs_group)

        with h5py.File(output_file, "r") as f:
            validate_stim_group(nirs.stim, f)

    def test_write_stim_group_logging_succeeds(self, tmp_path, full_nirs, caplog):
        """Test logging output for stimulus writing."""
        output_file = tmp_path / "test_stim_logging.h5"

        with caplog.at_level(logging.DEBUG):
            with h5py.File(output_file, "w") as f:
                nirs_group = f.create_group("/nirs")
                _write_stim_group(full_nirs.stim, nirs_group)

        # Check info level logging with exact path
        assert any(
            record.message == "Writing stimulus entries into /nirs"
            for record in caplog.records
        )

        # Check debug level logging for each stimulus with actual values
        for i, stim in enumerate(full_nirs.stim, start=1):
            expected_message = (
                f"Writing stimulus {i:02d}/{len(full_nirs.stim):02d}: "
                f"{stim.name} with {len(stim.data)} data points into /nirs/stim{i}"
            )
            assert any(record.message == expected_message for record in caplog.records)

    def test_write_stim_group_wrong_group_path_fails(self, tmp_path, full_nirs, caplog):
        """Test error when writing to wrong group path."""
        output_file = tmp_path / "test_stim_wrong_path.h5"

        with h5py.File(output_file, "w") as f:
            # Create group at wrong location
            wrong_group = f.create_group("/wrong/path")

            with caplog.at_level(logging.ERROR):
                with pytest.raises(
                    SnirfWriteError,
                    match="Stimulus group must be at /nirs",
                ):
                    _write_stim_group(full_nirs.stim, wrong_group)

            # Check error was logged
            assert any(
                "Stimulus group must be at /nirs" in record.message
                and record.levelname == "ERROR"
                for record in caplog.records
            )

    def test_write_stim_group_nested_wrong_path_fails(self, tmp_path, full_nirs):
        """Test error with nested but incorrect path."""
        output_file = tmp_path / "test_stim_nested_wrong.h5"

        with h5py.File(output_file, "w") as f:
            nirs_group = f.create_group("/nirs")
            data_group = nirs_group.create_group("data1")

            with pytest.raises(SnirfWriteError):
                _write_stim_group(full_nirs.stim, data_group)


class TestWriteProbeGroup:
    @pytest.mark.parametrize(
        "nirs_fixture",
        [
            "minimal_nirs",
            "full_nirs",
            "large_nirs",
            "unicode_nirs",
            "mixed_types_nirs",
            "float32_nirs",
            "empty_nirs",
            "negative_nirs",
            "unsorted_nirs",
            "zero_nirs",
            "whitespace_nirs",
        ],
    )
    def test_write_probe_group_valid_datasets_succeeds(
        self,
        tmp_path,
        nirs_fixture,
        request,
    ):
        """Test writing probe group with various valid datasets."""
        nirs = request.getfixturevalue(nirs_fixture)
        output_file = tmp_path / f"test_probe_{nirs_fixture}.h5"

        with h5py.File(output_file, "w") as f:
            nirs_group = f.create_group("/nirs")
            probe_group = nirs_group.create_group("probe")
            _write_probe_group(nirs.probe, probe_group)

        with h5py.File(output_file, "r") as f:
            validate_probe_group(nirs.probe, f)

    def test_write_probe_group_logging_output_succeeds(
        self,
        tmp_path,
        full_nirs,
        minimal_nirs,
        caplog,
    ):
        """Test that probe writing produces correct log messages with and without labels."""
        output_file = tmp_path / "test_probe_logging.h5"

        # Test with labels (full_nirs has sourceLabels and detectorLabels)
        with caplog.at_level(logging.DEBUG):
            with h5py.File(output_file, "w") as f:
                nirs_group = f.create_group("/nirs")
                probe_group = nirs_group.create_group("probe")
                _write_probe_group(full_nirs.probe, probe_group)

        # Check info level logging with exact path
        assert any(
            record.message == "Writing probe information into /nirs/probe"
            for record in caplog.records
        )

        # Check debug level logging for arrays with actual values
        assert any(
            record.message == f"Writing {len(full_nirs.probe.wavelengths)} wavelengths"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing source positions with shape {full_nirs.probe.sourcePos3D.shape}"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing detector positions with shape {full_nirs.probe.detectorPos3D.shape}"
            for record in caplog.records
        )

        # Check logging for labels with actual counts
        assert any(
            record.message
            == f"Writing {len(full_nirs.probe.sourceLabels)} source labels"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing {len(full_nirs.probe.detectorLabels)} detector labels"
            for record in caplog.records
        )

        caplog.clear()

        # Test without labels (minimal_nirs has no labels)
        output_file2 = tmp_path / "test_probe_logging_no_labels.h5"
        with caplog.at_level(logging.DEBUG):
            with h5py.File(output_file2, "w") as f:
                nirs_group = f.create_group("/nirs")
                probe_group = nirs_group.create_group("probe")
                _write_probe_group(minimal_nirs.probe, probe_group)

        # Check that "No labels to write" messages appear with exact text
        assert any(
            record.message == "No source labels to write" for record in caplog.records
        )
        assert any(
            record.message == "No detector labels to write" for record in caplog.records
        )

    def test_write_probe_group_wrong_path_fails(self, tmp_path, minimal_nirs, caplog):
        """Test that wrong group path raises SnirfWriteError with proper logging."""
        output_file = tmp_path / "test_probe_wrong_path.h5"

        with caplog.at_level(logging.ERROR):
            with h5py.File(output_file, "w") as f:
                wrong_group = f.create_group("/wrong/path")

                with pytest.raises(
                    SnirfWriteError,
                    match="Probe group must be at /nirs/probe",
                ):
                    _write_probe_group(minimal_nirs.probe, wrong_group)

        # Check error logging
        assert any(
            "Probe group must be at /nirs/probe" in record.message
            and record.levelname == "ERROR"
            for record in caplog.records
        )


class TestWriteSnirfIntegration:
    """Integration tests for the write_snirf function."""

    @pytest.mark.parametrize(
        "nirs_fixture",
        [
            "minimal_nirs",
            "full_nirs",
            "large_nirs",
            "unicode_nirs",
            "mixed_types_nirs",
            "float32_nirs",
            "empty_nirs",
            "negative_nirs",
            "unsorted_nirs",
            "zero_nirs",
            "whitespace_nirs",
        ],
    )
    def test_write_snirf_valid_datasets_succeeds(self, tmp_path, nirs_fixture, request):
        """Test writing complete SNIRF files with various valid datasets."""
        nirs = request.getfixturevalue(nirs_fixture)
        output_file = tmp_path / f"test_snirf_{nirs_fixture}.snirf"

        write_snirf(nirs, output_file)

        assert output_file.exists()
        with h5py.File(output_file, "r") as f:
            validate_snirf_file(nirs, f)

    def test_write_snirf_non_snirf_extension_warning_succeeds(
        self,
        tmp_path,
        minimal_nirs,
        caplog,
    ):
        """Test that a warning is logged when output file doesn't have .snirf extension."""
        output_file = tmp_path / "test_file.hdf5"

        with caplog.at_level(logging.WARNING):
            write_snirf(minimal_nirs, output_file)

        # Check that warning was logged
        assert any(
            "doesn't have the .snirf extension" in record.message
            for record in caplog.records
        )
        # File should still be created
        assert output_file.exists()

    def test_write_snirf_overwrite_existing_file_succeeds(
        self,
        tmp_path,
        minimal_nirs,
        full_nirs,
    ):
        """Test that writing to an existing file overwrites it correctly."""
        output_file = tmp_path / "test_overwrite.snirf"

        # Write initial file with minimal_nirs
        write_snirf(minimal_nirs, output_file)

        # Overwrite with full_nirs
        write_snirf(full_nirs, output_file)

        # Verify the file contains the new data (full_nirs), not the old (minimal_nirs)
        with h5py.File(output_file, "r") as f:
            assert (
                f["/nirs/metaDataTags/SubjectID"][()].decode("utf-8")
                == full_nirs.metadata.SubjectID
            )
            assert (
                f["/nirs/metaDataTags/MeasurementDate"][()].decode("utf-8")
                == full_nirs.metadata.MeasurementDate
            )
            assert len(f["/nirs/data1/time"][:]) == full_nirs.data[0].time.shape[0]
            assert f["/nirs/probe/wavelengths"][0] == full_nirs.probe.wavelengths[0]

    def test_write_snirf_detailed_logging_succeeds(self, tmp_path, full_nirs, caplog):
        """Test that detailed logging is produced during SNIRF writing."""
        output_file = tmp_path / "test_logging.snirf"

        with caplog.at_level(logging.DEBUG):
            write_snirf(full_nirs, output_file)

        # Verify log messages with exact values and paths
        assert any(
            record.message == f"Writing SNIRF file: {output_file}"
            for record in caplog.records
        )
        assert any(
            record.message == "Created HDF5 file, writing SNIRF structure"
            for record in caplog.records
        )
        assert any(
            record.message == "Writing metadata entries into /nirs/metaDataTags"
            for record in caplog.records
        )
        assert any(
            record.message == "Writing data entries into /nirs/data1"
            for record in caplog.records
        )
        assert any(
            record.message == f"Writing time with {len(full_nirs.data[0].time)} points"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing dataTimeSeries with shape {full_nirs.data[0].dataTimeSeries.shape}"
            for record in caplog.records
        )
        assert any(
            record.message
            == f"Writing measurementList with {len(full_nirs.data[0].measurementList)} entries"
            for record in caplog.records
        )
        assert any(
            record.message == "Writing probe information into /nirs/probe"
            for record in caplog.records
        )
        assert any(
            record.message == f"Writing {len(full_nirs.probe.wavelengths)} wavelengths"
            for record in caplog.records
        )
        assert any(
            record.message == "Writing stimulus entries into /nirs"
            for record in caplog.records
        )
        for i, stim in enumerate(full_nirs.stim, start=1):
            expected_message = (
                f"Writing stimulus {i:02d}/{len(full_nirs.stim):02d}: "
                f"{stim.name} with {len(stim.data)} data points into /nirs/stim{i}"
            )
            assert any(record.message == expected_message for record in caplog.records)
        assert any(
            record.message == "SNIRF file write completed" for record in caplog.records
        )

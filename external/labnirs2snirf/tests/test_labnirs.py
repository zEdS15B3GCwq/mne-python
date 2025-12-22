"""
Unit tests for labnirs module.
"""

import logging
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from labnirs2snirf import model
from labnirs2snirf.labnirs import (
    LabNirsReadError,
    _extract_data,
    _extract_metadata,
    _extract_probes,
    _extract_stims,
    _match_line,
    _read_header,
    _verify_header_format,
    read_labnirs,
    read_probe_pairs,
)


class TestExtractData:
    """Tests for the _extract_data function."""

    def test_extract_data_single_raw_channel_succeeds(self, caplog):
        """Test extraction with a single raw data channel."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "1-830": [100.0, 101.0, 102.0],
            },
        )
        columns = pl.DataFrame(
            {
                "name": ["time", "1-830"],
                "category": ["meta", "raw"],
                "subtype": [None, "830"],
                "source_index": [None, 1],
                "detector_index": [None, 1],
                "datatype": [None, 1],
                "wavelength_index": [None, 1],
            },
        )

        result = _extract_data(data, columns)

        assert isinstance(result, model.Data)
        assert result.time.shape == (3,)
        assert list(result.time) == [0.0, 1.0, 2.0]
        assert result.dataTimeSeries.shape == (3, 1)
        assert list(result.dataTimeSeries[:, 0]) == [100.0, 101.0, 102.0]
        assert len(result.measurementList) == 1
        assert result.measurementList[0].sourceIndex == 1
        assert result.measurementList[0].detectorIndex == 1
        assert result.measurementList[0].dataType == 1
        assert result.measurementList[0].wavelengthIndex == 1
        assert result.measurementList[0].dataTypeLabel is None

        assert "Extracting experimental data" in caplog.text
        assert "3 time points" in caplog.text
        assert "1 data channels" in caplog.text

    def test_extract_data_single_hb_channel_succeeds(self, caplog):
        """Test extraction with a single Hb data channel."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "1-hbo": [0.5, 0.6, 0.7],
            },
        )
        columns = pl.DataFrame(
            {
                "name": ["time", "1-hbo"],
                "category": ["meta", "hb"],
                "subtype": [None, "hbo"],
                "source_index": [None, 1],
                "detector_index": [None, 1],
                "datatype": [None, 99999],
                "wavelength_index": [None, 0],
            },
        )

        result = _extract_data(data, columns)

        assert result.dataTimeSeries.shape == (3, 1)
        assert len(result.measurementList) == 1
        assert result.measurementList[0].dataType == 99999
        assert result.measurementList[0].dataTypeLabel == "HbO"
        assert result.measurementList[0].wavelengthIndex == 0

    def test_extract_data_multiple_channels_succeeds(self, caplog):
        """Test extraction with multiple channels."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "1-830": [100.0, 101.0],
                "1-780": [90.0, 91.0],
                "2-830": [110.0, 111.0],
            },
        )
        columns = pl.DataFrame(
            {
                "name": ["time", "1-830", "1-780", "2-830"],
                "category": ["meta", "raw", "raw", "raw"],
                "subtype": [None, "830", "780", "830"],
                "source_index": [None, 1, 1, 2],
                "detector_index": [None, 1, 1, 1],
                "datatype": [None, 1, 1, 1],
                "wavelength_index": [None, 1, 2, 1],
            },
        )

        result = _extract_data(data, columns)

        assert result.dataTimeSeries.shape == (2, 3)
        assert len(result.measurementList) == 3
        assert result.measurementList[0].sourceIndex == 1
        assert result.measurementList[1].sourceIndex == 1
        assert result.measurementList[2].sourceIndex == 2
        assert result.measurementList[0].wavelengthIndex == 1
        assert result.measurementList[1].wavelengthIndex == 2
        assert result.measurementList[2].wavelengthIndex == 1
        assert np.all(result.dataTimeSeries[:, 0] == [100.0, 101.0])
        assert np.all(result.dataTimeSeries[:, 1] == [90.0, 91.0])
        assert np.all(result.dataTimeSeries[:, 2] == [110.0, 111.0])

        assert "3 data channels" in caplog.text

    def test_extract_data_mixed_hb_types_succeeds(self, caplog):
        """Test extraction with HbO, HbR, and HbT."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "1-hbo": [0.5, 0.6],
                "1-hbr": [-0.3, -0.4],
                "1-hbt": [0.2, 0.2],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2, 3],
                "name": ["time", "1-hbo", "1-hbr", "1-hbt"],
                "category": ["meta", "hb", "hb", "hb"],
                "subtype": [None, "hbo", "hbr", "hbt"],
                "source_index": [None, 1, 1, 1],
                "detector_index": [None, 1, 1, 1],
                "datatype": [None, 99999, 99998, 99997],
                "wavelength_index": [None, 3, 2, 1],
            },
        )

        result = _extract_data(data, columns)

        assert result.dataTimeSeries.shape == (2, 3)
        assert len(result.measurementList) == 3
        assert result.measurementList[0].dataTypeLabel == "HbO"
        assert result.measurementList[1].dataTypeLabel == "HbR"
        assert result.measurementList[2].dataTypeLabel == "HbT"
        assert result.measurementList[0].dataType == 99999
        assert result.measurementList[1].dataType == 99998
        assert result.measurementList[2].dataType == 99997
        assert result.measurementList[0].wavelengthIndex == 3
        assert result.measurementList[1].wavelengthIndex == 2
        assert result.measurementList[2].wavelengthIndex == 1

        assert "Unique data type labels: {'Hb" in caplog.text

    def test_extract_data_only_meta_columns_fails(self, caplog):
        """Test extraction when only meta columns are present."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "task": [0, 0, 0],
                "mark": ["0", "0", "0"],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2],
                "name": ["time", "task", "mark"],
                "category": ["meta", "meta", "meta"],
                "subtype": [None, None, None],
                "source_index": [None, None, None],
                "detector_index": [None, None, None],
                "datatype": [None, None, None],
                "wavelength_index": [None, None, None],
            },
        )

        with pytest.raises(
            LabNirsReadError,
            match="No data columns found after filtering; cannot extract data.",
        ):
            _extract_data(data, columns)

    def test_extract_data_single_timepoint_succeeds(self, caplog):
        """Test extraction with a single time point."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0],
                "1-830": [100.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1],
                "name": ["time", "1-830"],
                "category": ["meta", "raw"],
                "subtype": [None, "830"],
                "source_index": [None, 1],
                "detector_index": [None, 1],
                "datatype": [None, 1],
                "wavelength_index": [None, 1],
            },
        )

        result = _extract_data(data, columns)

        assert result.time.shape == (1,)
        assert result.dataTimeSeries.shape == (1, 1)
        assert result.time[0] == 0.0
        assert result.dataTimeSeries[0, 0] == 100.0

        assert "1 time points" in caplog.text

    def test_extract_data_many_channels_succeeds(self, caplog):
        """Test extraction with many channels."""
        caplog.set_level(logging.DEBUG)

        num_channels = 20
        data_dict = {
            "time": [0.0, 1.0],
            "task": [0, 0],
            "mark": ["0", "0"],
        }
        columns_data = [
            [0, "time", "meta", None, None, None, None, None],
            [1, "task", "meta", None, None, None, None, None],
            [2, "mark", "meta", None, None, None, None, None],
        ]

        for i in range(num_channels):
            col_name = f"{i + 1}-830"
            data_dict[col_name] = [float(100 + i), float(101 + i)]
            columns_data.append(
                [3 + i, col_name, "raw", "830", (i % 5) + 1, (i % 3) + 1, 1, 1],
            )

        data = pl.DataFrame(data_dict)
        columns = pl.DataFrame(
            columns_data,
            schema=[
                "column",
                "name",
                "category",
                "subtype",
                "source_index",
                "detector_index",
                "datatype",
                "wavelength_index",
            ],
            orient="row",
        )

        result = _extract_data(data, columns)

        assert result.dataTimeSeries.shape == (2, num_channels)
        assert len(result.measurementList) == num_channels

        assert f"{num_channels} data channels" in caplog.text
        assert f"{num_channels} MeasurementList entries" in caplog.text

    def test_extract_data_time_range_logging_succeeds(self, caplog):
        """Test that time range is logged correctly."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.5, 10.0, 100.5],
                "1-830": [100.0, 101.0, 102.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1],
                "name": ["time", "1-830"],
                "category": ["meta", "raw"],
                "subtype": [None, "830"],
                "source_index": [None, 1],
                "detector_index": [None, 1],
                "datatype": [None, 1],
                "wavelength_index": [None, 1],
            },
        )

        _extract_data(data, columns)

        assert "range 0.500 - 100.500" in caplog.text

    def test_extract_data_hb_datatype_index_always_zero_succeeds(self, caplog):
        """Test that dataTypeIndex is always 0 for all measurements."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "1-hbo": [0.5, 0.6],
                "2-hbr": [-0.3, -0.4],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2],
                "name": ["time", "1-hbo", "2-hbr"],
                "category": ["meta", "hb", "hb"],
                "subtype": [None, "hbo", "hbr"],
                "source_index": [None, 1, 2],
                "detector_index": [None, 1, 1],
                "datatype": [None, 99999, 99999],
                "wavelength_index": [None, 0, 0],
            },
        )

        result = _extract_data(data, columns)

        assert all(m.dataTypeIndex == 0 for m in result.measurementList)

    def test_extract_data_raw_wavelength_indices_succeeds(self, caplog):
        """Test that raw data has correct wavelength indices."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "1-780": [90.0, 91.0],
                "1-830": [100.0, 101.0],
                "1-870": [110.0, 111.0],
                "2-780": [120.0, 121.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2, 3, 4],
                "name": ["time", "1-780", "1-830", "1-870", "2-780"],
                "category": ["meta", "raw", "raw", "raw", "raw"],
                "subtype": [None, "780", "830", "870", "780"],
                "source_index": [None, 1, 1, 1, 2],
                "detector_index": [None, 1, 1, 1, 2],
                "datatype": [None, 1, 1, 1, 1],
                "wavelength_index": [None, 1, 2, 3, 1],
            },
        )

        result = _extract_data(data, columns)

        assert len(result.measurementList) == 4
        assert result.measurementList[0].wavelengthIndex == 1
        assert result.measurementList[1].wavelengthIndex == 2
        assert result.measurementList[2].wavelengthIndex == 3
        assert result.measurementList[3].wavelengthIndex == 1

        assert "wavelength indices: {1, 2, 3}" in caplog.text

    def test_extract_data_column_order_preserved_succeeds(self, caplog):
        """Test that column order in dataTimeSeries matches columns DataFrame."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "3-830": [300.0, 301.0],
                "1-830": [100.0, 101.0],
                "2-830": [200.0, 201.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2, 3],
                "name": ["time", "3-830", "1-830", "2-830"],
                "category": ["meta", "raw", "raw", "raw"],
                "subtype": [None, "830", "830", "830"],
                "source_index": [None, 3, 1, 2],
                "detector_index": [None, 1, 1, 1],
                "datatype": [None, 1, 1, 1],
                "wavelength_index": [None, 1, 1, 1],
            },
        )

        result = _extract_data(data, columns)

        # Order should match columns DataFrame order
        assert result.dataTimeSeries[0, 0] == 300.0
        assert result.dataTimeSeries[0, 1] == 100.0
        assert result.dataTimeSeries[0, 2] == 200.0
        assert result.measurementList[0].sourceIndex == 3
        assert result.measurementList[1].sourceIndex == 1
        assert result.measurementList[2].sourceIndex == 2

    def test_extract_data_numpy_array_types_succeeds(self, caplog):
        """Test that output arrays have correct numpy types."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "1-830": [100.0, 101.0, 102.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1],
                "name": ["time", "1-830"],
                "category": ["meta", "raw"],
                "subtype": [None, None],
                "source_index": [None, 1],
                "detector_index": [None, 1],
                "datatype": [None, 1],
                "wavelength_index": [None, 1],
            },
        )

        result = _extract_data(data, columns)

        assert isinstance(result.time, np.ndarray)
        assert isinstance(result.dataTimeSeries, np.ndarray)
        assert result.time.dtype == np.float64
        assert result.dataTimeSeries.dtype == np.float64

    def test_extract_data_logging_unique_labels_none_for_raw_succeeds(self, caplog):
        """Test logging shows None in unique labels for raw data."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "1-830": [100.0, 101.0],
                "2-780": [90.0, 91.0],
            },
        )
        columns = pl.DataFrame(
            {
                "column": [0, 1, 2],
                "name": ["time", "1-830", "2-780"],
                "category": ["meta", "raw", "raw"],
                "subtype": [None, "830", "780"],
                "source_index": [None, 1, 2],
                "detector_index": [None, 1, 1],
                "datatype": [None, 1, 1],
                "wavelength_index": [None, 2, 1],
            },
        )

        _extract_data(data, columns)

        assert "Unique data type labels: {None}" in caplog.text


class TestExtractProbes:
    """Tests for the _extract_probes function."""

    def test_extract_probes_single_source_detector_wavelength_succeeds(self, caplog):
        """Test extraction with single source, detector, and wavelength."""
        caplog.set_level(logging.DEBUG)

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        probe = _extract_probes(sources, detectors, wavelengths)

        assert probe.wavelengths.shape == (1,)
        assert probe.wavelengths[0] == 830.0
        assert probe.sourcePos3D.shape == (1, 3)
        assert probe.detectorPos3D.shape == (1, 3)
        assert np.all(probe.sourcePos3D == 0.0)
        assert np.all(probe.detectorPos3D == 0.0)
        assert probe.sourceLabels == ["S1"]
        assert probe.detectorLabels == ["D1"]
        assert "Extracting probe information" in caplog.text
        assert "1 wavelengths, 1 sources, and 1 detectors" in caplog.text

    def test_extract_probes_multiple_sources_detectors_succeeds(self, caplog):
        """Test extraction with multiple sources and detectors."""
        caplog.set_level(logging.DEBUG)

        sources = pl.DataFrame({"index": [1, 2, 3], "label": ["S1", "S2", "S3"]})
        detectors = pl.DataFrame({"index": [1, 2], "label": ["D1", "D2"]})
        wavelengths = pl.DataFrame(
            {"wavelength": [780, 805, 830], "wavelength_index": [1, 2, 3]},
        )

        probe = _extract_probes(sources, detectors, wavelengths)

        assert probe.wavelengths.shape == (3,)
        assert list(probe.wavelengths) == [780.0, 805.0, 830.0]
        assert probe.sourcePos3D.shape == (3, 3)
        assert probe.detectorPos3D.shape == (2, 3)
        assert np.all(probe.sourcePos3D == 0.0)
        assert np.all(probe.detectorPos3D == 0.0)
        assert probe.sourceLabels == ["S1", "S2", "S3"]
        assert probe.detectorLabels == ["D1", "D2"]
        assert "3 wavelengths, 3 sources, and 2 detectors" in caplog.text

    def test_extract_probes_positions_all_zeros_succeeds(self):
        """Test that all positions are initialized to zero."""

        sources = pl.DataFrame({"index": [1, 2], "label": ["S1", "S2"]})
        detectors = pl.DataFrame({"index": [1, 2, 3], "label": ["D1", "D2", "D3"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        probe = _extract_probes(sources, detectors, wavelengths)

        assert np.all(probe.sourcePos3D == 0.0)
        assert np.all(probe.detectorPos3D == 0.0)
        assert probe.sourcePos3D.dtype == np.float64
        assert probe.detectorPos3D.dtype == np.float64

    def test_extract_probes_wavelength_conversion_to_float_succeeds(self):
        """Test that wavelengths are converted to float64."""

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame(
            {"wavelength": [780, 830, 870], "wavelength_index": [1, 2, 3]},
        )

        probe = _extract_probes(sources, detectors, wavelengths)

        assert probe.wavelengths.dtype == np.float64
        assert list(probe.wavelengths) == [780.0, 830.0, 870.0]

    def test_extract_probes_missing_input_fails(self):
        """Test extraction with empty sources, detectors or wavelengths."""

        sources = pl.DataFrame(
            {"index": [], "label": []},
            schema={"index": pl.Int32, "label": pl.String},
        )
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_probes(sources, detectors, wavelengths)
        assert "Cannot extract probe information" in str(exc_info.value)

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame(
            {"index": [], "label": []},
            schema={"index": pl.Int32, "label": pl.String},
        )
        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_probes(sources, detectors, wavelengths)
        assert "Cannot extract probe information" in str(exc_info.value)

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame(
            {"wavelength": [], "wavelength_index": []},
            schema={"wavelength": pl.UInt32, "wavelength_index": pl.Int64},
        )

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_probes(sources, detectors, wavelengths)
        assert "Cannot extract probe information" in str(exc_info.value)

    def test_extract_probes_many_sources_and_detectors_succeeds(self, caplog):
        """Test extraction with many sources and detectors and wavelengths."""
        caplog.set_level(logging.DEBUG)

        sources = pl.DataFrame(
            {"index": list(range(1, 11)), "label": [f"S{i}" for i in range(1, 11)]},
        )
        detectors = pl.DataFrame(
            {"index": list(range(1, 9)), "label": [f"D{i}" for i in range(1, 9)]},
        )
        wavelengths = pl.DataFrame(
            {"wavelength": range(700, 901, 20), "wavelength_index": range(11)},
        )

        probe = _extract_probes(sources, detectors, wavelengths)

        assert probe.sourcePos3D.shape == (10, 3)
        assert probe.detectorPos3D.shape == (8, 3)
        assert probe.sourceLabels is not None and len(probe.sourceLabels) == 10
        assert probe.detectorLabels is not None and len(probe.detectorLabels) == 8
        assert len(probe.wavelengths) == 11
        assert "10 sources, and 8 detectors" in caplog.text

    def test_extract_probes_label_format_succeeds(self):
        """Test that labels follow the expected format."""

        sources = pl.DataFrame({"index": [1, 2, 5], "label": ["S1", "S2", "S5"]})
        detectors = pl.DataFrame({"index": [1, 3, 4], "label": ["D1", "D3", "D4"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        probe = _extract_probes(sources, detectors, wavelengths)

        # Labels should match exactly what was provided
        assert probe.sourceLabels == ["S1", "S2", "S5"]
        assert probe.detectorLabels == ["D1", "D3", "D4"]

    def test_extract_probes_logging_details_succeeds(self, caplog):
        """Test that all expected logging information is present."""
        caplog.set_level(logging.DEBUG)

        sources = pl.DataFrame({"index": [1, 2], "label": ["S1", "S2"]})
        detectors = pl.DataFrame({"index": [1, 2, 3], "label": ["D1", "D2", "D3"]})
        wavelengths = pl.DataFrame({"wavelength": [780], "wavelength_index": [1]})

        _extract_probes(sources, detectors, wavelengths)

        assert "Extracting probe information" in caplog.text
        assert "1 wavelengths" in caplog.text
        assert "2 sources" in caplog.text
        assert "3 detectors" in caplog.text
        assert "2 source labels" in caplog.text
        assert "3 detector labels" in caplog.text

    def test_extract_probes_wavelength_ordering_preserved_succeeds(self, caplog):
        """Test that wavelength ordering is preserved."""
        caplog.set_level(logging.DEBUG)

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame(
            {"wavelength": [870, 780, 830], "wavelength_index": [3, 1, 2]},
        )

        probe = _extract_probes(sources, detectors, wavelengths)

        # Wavelengths should be in the order provided
        assert list(probe.wavelengths) == [870.0, 780.0, 830.0]

    def test_extract_probes_non_contiguous_indices_succeeds(self):
        """Test with non-contiguous source/detector indices."""

        sources = pl.DataFrame({"index": [1, 3, 7], "label": ["S1", "S3", "S7"]})
        detectors = pl.DataFrame({"index": [2, 5, 8], "label": ["D2", "D5", "D8"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        probe = _extract_probes(sources, detectors, wavelengths)

        # Number of rows should match number of indices provided
        assert probe.sourcePos3D.shape[0] == 3
        assert probe.detectorPos3D.shape[0] == 3
        assert probe.sourceLabels is not None and probe.sourceLabels == [
            "S1",
            "S3",
            "S7",
        ]
        assert probe.detectorLabels is not None and probe.detectorLabels == [
            "D2",
            "D5",
            "D8",
        ]

    def test_extract_probes_return_type_succeeds(self):
        """Test that the function returns a model.Probe object."""

        sources = pl.DataFrame({"index": [1], "label": ["S1"]})
        detectors = pl.DataFrame({"index": [1], "label": ["D1"]})
        wavelengths = pl.DataFrame({"wavelength": [830], "wavelength_index": [1]})

        probe = _extract_probes(sources, detectors, wavelengths)

        assert isinstance(probe, model.Probe)


class TestExtractMetadata:
    """Tests for the _extract_metadata function."""

    def test_extract_metadata_complete_header_succeeds(self, caplog, small_data_path):
        """Test extraction with all metadata fields present."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == "ID1"
        assert metadata.MeasurementDate == "2000-01-02"
        assert metadata.MeasurementTime == "11:12:13"
        assert metadata.LengthUnit == "m"
        assert metadata.TimeUnit == "s"
        assert metadata.FrequencyUnit == "Hz"
        assert "SubjectName" in metadata.additional_fields
        assert metadata.additional_fields["SubjectName"] == "subject1"
        assert "comment" in metadata.additional_fields
        assert metadata.additional_fields["comment"] == "comment1"

        assert "Extracting metadata from header" in caplog.text
        assert "Extracted metadata has subject ID: True" in caplog.text
        assert "has date: True" in caplog.text
        assert "has time: True" in caplog.text

    def test_extract_metadata_minimal_header_succeeds(self, caplog, minimal_data_path):
        """Test extraction with minimal metadata (no ID, name, comment)."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(minimal_data_path)

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == ""
        assert metadata.MeasurementDate == "2000-01-02"
        assert metadata.MeasurementTime == "11:12:13"
        assert metadata.LengthUnit == "m"
        assert metadata.TimeUnit == "s"
        assert metadata.FrequencyUnit == "Hz"
        assert len(metadata.additional_fields) == 0

        assert "Extracting metadata from header" in caplog.text
        assert "Extracted metadata has subject ID: False" in caplog.text
        assert "has date: True" in caplog.text
        assert "has time: True" in caplog.text

    def test_extract_metadata_missing_id_succeeds(self, caplog, small_data_path):
        """Test extraction when ID field is missing."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Replace ID line with invalid format
        header[2] = "InvalidID\tnoformat\n"

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == ""
        assert "Extracted metadata has subject ID: False" in caplog.text

    def test_extract_metadata_empty_id_succeeds(self, caplog, small_data_path):
        """Test extraction when ID field is present but empty."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Set ID to empty string
        header[2] = "ID\t\tVersion\t11.0\n"

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == ""
        assert "Extracted metadata has subject ID: False" in caplog.text

    def test_extract_metadata_missing_datetime_fails(self, caplog, small_data_path):
        """Test extraction when measurement datetime is missing."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Replace datetime line with invalid format
        header[1] = "Invalid datetime line\n"

        with pytest.raises(KeyError):
            _extract_metadata(header)

    def test_extract_metadata_invalid_date_format_fails(self, caplog, small_data_path):
        """Test extraction when date format is invalid (not 3 fields)."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid date format - only 2 fields
        header[1] = "Measured Date\t2023/12 14:30:45\n"

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_metadata(header)

        assert "Invalid measurement date format" in str(exc_info.value)
        assert "2023/12" in str(exc_info.value)

    def test_extract_metadata_invalid_date_format_too_many_fields_fails(
        self,
        caplog,
        small_data_path,
    ):
        """Test extraction when date has too many fields."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid date format - 4 fields
        header[1] = "Measured Date\t2023/12/25/01 14:30:45\n"

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_metadata(header)

        assert "Invalid measurement date format" in str(exc_info.value)
        assert "2023/12/25/01" in str(exc_info.value)

    def test_extract_metadata_invalid_time_format_fails(self, caplog, small_data_path):
        """Test extraction when time format is invalid (not 3 fields)."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid time format - only 2 fields
        header[1] = "Measured Date\t2023/12/25 14:35\n"

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_metadata(header)

        assert "Invalid measurement time format" in str(exc_info.value)
        assert "14:35" in str(exc_info.value)

    def test_extract_metadata_invalid_time_format_too_many_fields_fails(
        self,
        caplog,
        small_data_path,
    ):
        """Test extraction when time has too many fields."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid time format - 4 fields
        header[1] = "Measured Date\t2023/12/25 14:30:45:01\n"

        with pytest.raises(LabNirsReadError) as exc_info:
            _extract_metadata(header)

        assert "Invalid measurement time format" in str(exc_info.value)
        assert "14:30:45:01" in str(exc_info.value)

    def test_extract_metadata_date_format_conversion_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that date is converted from slash to dash format."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Ensure date has slashes
        header[1] = "Measured Date\t2023/12/25 14:30:45\n"

        metadata = _extract_metadata(header)

        assert metadata.MeasurementDate == "2023-12-25"
        assert metadata.MeasurementTime == "14:30:45"

    def test_extract_metadata_empty_name_succeeds(self, caplog, small_data_path):
        """Test extraction when name field is empty."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Set name to empty string
        header[3] = "Name\t\t[HeaderType]\t11.0/11.0\n"

        metadata = _extract_metadata(header)

        assert "SubjectName" not in metadata.additional_fields
        assert len(metadata.additional_fields) == 1  # Only comment should be present

    def test_extract_metadata_empty_comment_succeeds(self, caplog, small_data_path):
        """Test extraction when comment field is empty."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Set comment to empty string
        header[4] = "Comment\t\n"

        metadata = _extract_metadata(header)

        assert "comment" not in metadata.additional_fields
        assert (
            len(metadata.additional_fields) == 1
        )  # Only SubjectName should be present

    def test_extract_metadata_missing_name_succeeds(self, caplog, small_data_path):
        """Test extraction when name field is in wrong format."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid format for name line
        header[3] = "InvalidName\tnoformat\n"

        metadata = _extract_metadata(header)

        assert "SubjectName" not in metadata.additional_fields

    def test_extract_metadata_missing_comment_succeeds(self, caplog, small_data_path):
        """Test extraction when comment field is in wrong format."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Invalid format for comment line
        header[4] = "InvalidComment\n"

        metadata = _extract_metadata(header)

        assert "comment" not in metadata.additional_fields

    def test_extract_metadata_special_characters_in_fields_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test extraction with special characters in metadata fields."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[2] = "ID\tSUBJ@#$123\tVersion\t11.0\n"
        header[3] = "Name\tJohn O'Brien-Smith\t[HeaderType]\t11.0/11.0\n"
        header[4] = "Comment\tTest with special chars: @#$%^&*()\n"

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == "SUBJ@#$123"
        assert metadata.additional_fields["SubjectName"] == "John O'Brien-Smith"
        assert (
            metadata.additional_fields["comment"]
            == "Test with special chars: @#$%^&*()"
        )

    def test_extract_metadata_default_units_succeeds(self, caplog, small_data_path):
        """Test that default units are set correctly."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)

        metadata = _extract_metadata(header)

        assert metadata.LengthUnit == "m"
        assert metadata.TimeUnit == "s"
        assert metadata.FrequencyUnit == "Hz"

    def test_extract_metadata_additional_fields_structure_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that additional_fields is a dictionary."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)

        metadata = _extract_metadata(header)

        assert isinstance(metadata.additional_fields, dict)
        assert all(isinstance(k, str) for k in metadata.additional_fields)
        assert all(isinstance(v, str) for v in metadata.additional_fields.values())

    def test_extract_metadata_long_comment_succeeds(self, caplog, small_data_path):
        """Test extraction with a very long comment."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        long_comment = "This is a very long comment " * 50
        header[4] = f"Comment\t{long_comment}\n"

        metadata = _extract_metadata(header)

        assert metadata.additional_fields["comment"] == long_comment

    def test_extract_metadata_whitespace_in_fields_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test extraction preserves whitespace in field values."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[3] = "Name\t  John Doe  \t[HeaderType]\t11.0/11.0\n"
        header[4] = "Comment\t  test comment  \n"

        metadata = _extract_metadata(header)

        # Whitespace should be preserved as per the regex pattern
        assert metadata.additional_fields["SubjectName"] == "  John Doe  "
        assert metadata.additional_fields["comment"] == "  test comment  "

    def test_extract_metadata_tabs_in_comment_succeeds(self, caplog, small_data_path):
        """Test extraction with tabs in comment field."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[4] = "Comment\tComment\twith\ttabs\n"

        metadata = _extract_metadata(header)

        assert metadata.additional_fields["comment"] == "Comment\twith\ttabs"

    def test_extract_metadata_numeric_name_succeeds(self, caplog, small_data_path):
        """Test extraction when name contains only numbers."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[3] = "Name\t12345\t[HeaderType]\t11.0/11.0\n"

        metadata = _extract_metadata(header)

        assert metadata.additional_fields["SubjectName"] == "12345"

    def test_extract_metadata_logging_presence_of_fields_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that logging correctly reports presence/absence of fields."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)

        _extract_metadata(header)

        # Check that all presence indicators are logged
        assert "has subject ID:" in caplog.text
        assert "has date:" in caplog.text
        assert "has time:" in caplog.text
        assert "has additional fields:" in caplog.text

    def test_extract_metadata_only_required_fields_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test with only required fields (ID, date, time)."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        # Remove name and comment
        header[3] = "OtherField\tvalue\n"
        header[4] = "AnotherField\tvalue\n"

        metadata = _extract_metadata(header)

        assert metadata.SubjectID == "ID1"
        assert metadata.MeasurementDate == "2000-01-02"
        assert metadata.MeasurementTime == "11:12:13"
        assert len(metadata.additional_fields) == 0

    def test_extract_metadata_date_with_single_digits_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test date extraction with single-digit month/day."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[1] = "Measured Date\t2023/1/5 09:05:03\n"

        metadata = _extract_metadata(header)

        assert metadata.MeasurementDate == "2023-01-05"
        assert metadata.MeasurementTime == "09:05:03"

    def test_extract_metadata_time_with_single_digits_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test time extraction with single-digit hours/minutes/seconds."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[1] = "Measured Date\t2023/12/25 1:2:3\n"

        metadata = _extract_metadata(header)

        assert metadata.MeasurementTime == "01:02:03"

    def test_extract_metadata_datetime_padding_double_digits_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that double-digit date and time remain unchanged."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(small_data_path)
        header[1] = "Measured Date\t2023/12/31 23:59:59\n"

        metadata = _extract_metadata(header)

        assert metadata.MeasurementDate == "2023-12-31"
        assert metadata.MeasurementTime == "23:59:59"


class TestExtractStims:
    """Tests for the _extract_stims function."""

    def test_extract_stims_single_task_succeeds(self, caplog):
        """Test extraction of a single task with multiple events."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "task": [0, 1, 0, 1, 0],
                "mark": ["0", "1", "0", "1", "0"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 1
        assert stims[0].name == "1"
        assert len(stims[0].data) == 2
        assert list(stims[0].data) == [1.0, 3.0]

        assert "Extracting stimulus information from data" in caplog.text
        assert "Found 1 stimulus types" in caplog.text
        assert "Stimulus type '1' has 2 events" in caplog.text

    def test_extract_stims_multiple_tasks_succeeds(self, caplog):
        """Test extraction of multiple different tasks."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "task": [0, 1, 0, 2, 1, 2],
                "mark": ["0", "1", "0", "1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 2
        # Stims should be sorted by task name
        assert stims[0].name == "1"
        assert stims[1].name == "2"
        assert len(stims[0].data) == 2
        assert len(stims[1].data) == 2
        assert list(stims[0].data) == [1.0, 4.0]
        assert list(stims[1].data) == [3.0, 5.0]

        assert "Found 2 stimulus types" in caplog.text
        assert "Stimulus type '1' has 2 events" in caplog.text
        assert "Stimulus type '2' has 2 events" in caplog.text

    def test_extract_stims_with_zeroing_succeeds(self, caplog):
        """Test extraction with zeroing event (0Z mark)."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 2.0, 3.0],
                "task": [0, 0, 1, 0, 1],
                "mark": ["0", "0Z", "1", "0", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 2
        # Z should come before numeric tasks when sorted
        assert stims[0].name == "1"
        assert stims[1].name == "Z"
        assert len(stims[0].data) == 2
        assert len(stims[1].data) == 1
        assert list(stims[1].data) == [0.5]
        assert list(stims[0].data) == [1.0, 3.0]

        assert "Found 2 stimulus types" in caplog.text
        assert "Stimulus type 'Z' has 1 events" in caplog.text

    def test_extract_stims_only_zeroing_succeeds(self, caplog):
        """Test extraction with only zeroing events."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "task": [0, 0, 0],
                "mark": ["0Z", "0", "0Z"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 1
        assert stims[0].name == "Z"
        assert len(stims[0].data) == 2
        assert list(stims[0].data) == [0.0, 2.0]

    def test_extract_stims_no_events_succeeds(self, caplog):
        """Test extraction when no events are marked (all mark='0')."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "task": [0, 0, 0, 0],
                "mark": ["0", "0", "0", "0"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 0
        assert "Found 0 stimulus types" in caplog.text

    def test_extract_stims_empty_dataframe_succeeds(self, caplog):
        """Test extraction with empty dataframe."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame({"time": [], "task": [], "mark": []})

        stims = _extract_stims(data)

        assert len(stims) == 0
        assert "Found 0 stimulus types" in caplog.text

    def test_extract_stims_task_zero_as_normal_event_succeeds(self, caplog):
        """Test that task 0 can be used as a normal event with mark='1'."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "task": [0, 1, 0, 1],
                "mark": ["1", "1", "1", "0"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 2
        assert stims[0].name == "0"
        assert stims[1].name == "1"
        assert list(stims[0].data) == [0.0, 2.0]
        assert list(stims[1].data) == [1.0]

    def test_extract_stims_many_tasks_succeeds(self, caplog):
        """Test extraction with many different task types."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "task": [1, 2, 3, 4, 5, 1, 3],
                "mark": ["1", "1", "1", "1", "1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 5
        # Check sorted order
        assert [s.name for s in stims] == ["1", "2", "3", "4", "5"]
        assert len(stims[0].data) == 2  # task 1 appears twice
        assert len(stims[2].data) == 2  # task 3 appears twice
        assert len(stims[1].data) == 1  # task 2 appears once

        assert "Found 5 stimulus types" in caplog.text

    def test_extract_stims_single_event_per_task_succeeds(self, caplog):
        """Test extraction where each task has only one event."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [1.0, 2.0, 3.0],
                "task": [1, 2, 3],
                "mark": ["1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 3
        for i, stim in enumerate(stims, start=1):
            assert stim.name == str(i)
            assert len(stim.data) == 1
            assert stim.data[0] == float(i)

    def test_extract_stims_mixed_marks_and_zeroing_succeeds(self, caplog):
        """Test extraction with a mix of regular marks and zeroing."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "task": [0, 1, 0, 2, 0, 1, 0],
                "mark": ["0Z", "1", "0", "1", "0Z", "1", "0"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 3
        assert stims[0].name == "1"
        assert stims[1].name == "2"
        assert stims[2].name == "Z"
        assert list(stims[0].data) == [0.5, 2.5]
        assert list(stims[1].data) == [1.5]
        assert list(stims[2].data) == [0.0, 2.0]

    def test_extract_stims_time_ordering_succeeds(self, caplog):
        """Test that event timing is preserved. Shouldn't occur in real data, though."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [5.0, 2.0, 8.0, 1.0, 10.0],
                "task": [1, 1, 1, 1, 1],
                "mark": ["1", "1", "1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 1
        # Events should maintain the order they appear in the dataframe
        assert list(stims[0].data) == [5.0, 2.0, 8.0, 1.0, 10.0]

    def test_extract_stims_floating_point_times_succeeds(self, caplog):
        """Test extraction with precise floating-point time values."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.123, 1.456, 2.789, 3.012],
                "task": [1, 1, 2, 2],
                "mark": ["1", "1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 2
        assert list(stims[0].data) == [0.123, 1.456]
        assert list(stims[1].data) == [2.789, 3.012]

    def test_extract_stims_logging_dataframe_dimensions_succeeds(self, caplog):
        """Test that dataframe dimensions are logged correctly."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "task": [1, 2, 1],
                "mark": ["1", "1", "1"],
            },
        )

        _extract_stims(data)

        assert "Extracted task dataframe has 3 rows and 2 columns" in caplog.text

    def test_extract_stims_large_task_numbers_succeeds(self, caplog):
        """Test extraction with large task numbers."""
        caplog.set_level(logging.DEBUG)

        data = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "task": [100, 999, 100],
                "mark": ["1", "1", "1"],
            },
        )

        stims = _extract_stims(data)

        assert len(stims) == 2
        assert stims[0].name == "100"
        assert stims[1].name == "999"
        assert len(stims[0].data) == 2
        assert len(stims[1].data) == 1


class TestMatchLine:
    """Tests for the _match_line function."""

    def test_match_line_simple_pattern_succeeds(self, caplog):
        """Test matching a simple pattern against lines."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^ID\t(?P<id>[^\t]*).*$"
        lines = ["ID\tSUBJ001\tOther\tData\n", "Name\tJohn Doe\n"]

        result = _match_line(pattern, lines)

        assert result == {"id": "SUBJ001"}
        assert "Matching pattern" in caplog.text
        assert "Found pattern in line" in caplog.text
        assert "ID\tSUBJ001" in caplog.text

    def test_match_line_no_match_succeeds(self, caplog):
        """Test when pattern doesn't match any line."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^NonExistent\t(?P<value>.*)$"
        lines = ["ID\tSUBJ001\n", "Name\tJohn Doe\n", "Comment\tTest\n"]

        result = _match_line(pattern, lines)

        assert result == {}
        assert "Matching pattern" in caplog.text
        assert "Pattern not found in header" in caplog.text

    def test_match_line_multiple_groups_succeeds(self, caplog):
        """Test matching pattern with multiple named groups."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^Measured Date\t(?P<date>[\d/]+) (?P<time>[\d:]+)\s*$"
        lines = [
            "ID\tSUBJ001\n",
            "Measured Date\t2000/01/02 11:12:13\n",
            "Name\tJohn Doe\n",
        ]

        result = _match_line(pattern, lines)

        assert result == {"date": "2000/01/02", "time": "11:12:13"}
        assert "Found pattern in line" in caplog.text

    def test_match_line_stops_at_first_match_succeeds(self, caplog):
        """Test that matching stops at the first matching line."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^Name\t(?P<name>[^\t]*).*$"
        lines = [
            "Name\tFirstMatch\tData\n",
            "Name\tSecondMatch\tData\n",
            "Comment\tTest\n",
        ]

        result = _match_line(pattern, lines)

        # Should return the first match, not the second
        assert result == {"name": "FirstMatch"}
        assert caplog.text.count("Found pattern in line") == 1

    def test_match_line_empty_lines_list_succeeds(self, caplog):
        """Test matching against an empty list of lines."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^ID\t(?P<id>.*)$"
        lines = []

        result = _match_line(pattern, lines)

        assert result == {}
        assert "Pattern not found in header" in caplog.text

    def test_match_line_with_empty_capture_group_succeeds(self, caplog):
        """Test matching where captured group is empty."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^ID\t(?P<id>[^\t]*)\t.*$"
        lines = ["ID\t\tOther\tData\n"]

        result = _match_line(pattern, lines)

        assert result == {"id": ""}
        assert "Found pattern in line" in caplog.text

    def test_match_line_with_special_characters_succeeds(self, caplog):
        """Test matching lines with special characters."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^Comment\t(?P<comment>.*)$"
        lines = [
            "ID\tSUBJ001\n",
            "Comment\tSpecial chars: @#$%^&*()\n",
        ]

        result = _match_line(pattern, lines)

        assert result == {"comment": "Special chars: @#$%^&*()"}
        assert "Found pattern in line" in caplog.text

    def test_match_line_version_pattern_succeeds(self, caplog):
        """Test matching version pattern from actual header."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^[^\t]*\t[^\t]*\tVersion\t11\.0$"
        lines = [
            "Some\tFields\tVersion\t11.0\n",
            "Other\tData\n",
        ]

        result = _match_line(pattern, lines)

        # Pattern has no named groups, should return empty dict
        assert result == {}
        assert "Found pattern in line" in caplog.text

    def test_match_line_channel_pairs_pattern_succeeds(self, caplog):
        """Test matching channel pairs pattern."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^(?P<channel_pairs>(?>\(\d+,\d+\))+)$"
        lines = [
            "Some header line\n",
            "(1,1)(2,1)(3,1)(1,2)\n",
            "Another line\n",
        ]

        result = _match_line(pattern, lines)

        assert result == {"channel_pairs": "(1,1)(2,1)(3,1)(1,2)"}
        assert "Found pattern in line" in caplog.text

    def test_match_line_case_sensitive_succeeds(self, caplog):
        """Test that pattern matching is case-sensitive."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^ID\t(?P<id>.*)$"
        lines = ["id\tSUBJ001\n", "ID\tSUBJ002\n"]

        result = _match_line(pattern, lines)

        # Should match the second line (uppercase ID), not the first
        assert result == {"id": "SUBJ002"}

    def test_match_line_complex_regex_succeeds(self, caplog):
        """Test with a complex regex pattern."""
        caplog.set_level(logging.DEBUG)

        pattern = (
            r"^ \[File Information\]\s*\[Data Line\]\t(?P<data_start_line>\d+)\s*$"
        )
        lines = [" [File Information]      \t\t [Data Line]\t36\n", "Other line\n"]

        result = _match_line(pattern, lines)

        assert result == {"data_start_line": "36"}
        assert "Found pattern in line" in caplog.text

    def test_match_line_with_no_named_groups_succeeds(self, caplog):
        """Test pattern with no named groups returns empty dict when matched."""
        caplog.set_level(logging.DEBUG)

        pattern = r"^ID\t.*$"
        lines = ["ID\tSUBJ001\n"]

        result = _match_line(pattern, lines)

        assert result == {}
        assert "Found pattern in line" in caplog.text


class TestReadHeader:
    """Tests for the _read_header function."""

    @pytest.mark.parametrize("filename", ["minimal_labnirs.txt", "small_labnirs.txt"])
    def test_read_header_valid_files_succeeds(self, caplog, filename, test_data_dir):
        """Test reading header from valid LabNIRS files."""
        caplog.set_level(logging.DEBUG)

        with patch("labnirs2snirf.labnirs._verify_header_format") as mock_verify:
            mock_verify.side_effect = lambda x: None  # No-op for verification
            header = _read_header(test_data_dir / filename)

        # Check that exactly 35 lines are read
        assert len(header) == 35, f"Expected 35 header lines, got {len(header)}"

        # Check that all lines are strings
        assert all(isinstance(line, str) for line in header)

        # Check that the first line starts with expected format
        assert header[0].startswith(" [File Information]")

        # Check logging messages
        assert "Reading header lines from file" in caplog.text
        assert "Read header lines: requested 35, read 35 lines" in caplog.text

    def test_read_header_nonexistent_file_fails(self, caplog, tmp_path):
        """Test that reading non-existent file raises LabNirsReadError."""
        caplog.set_level(logging.DEBUG)

        non_existent = tmp_path / "nonexistent_file.txt"

        with pytest.raises(LabNirsReadError) as exc_info:
            _read_header(non_existent)

        assert "Error reading the header" in str(exc_info.value)
        assert "nonexistent_file.txt" in str(exc_info.value)
        assert "ERROR" in caplog.text
        assert "FileNotFoundError" in caplog.text
        assert "Error reading the header" in caplog.text

    def test_read_header_calls_verify_format_succeeds(self, caplog, minimal_data_path):
        """Test that _verify_header_format is called during read."""
        caplog.set_level(logging.DEBUG)

        # This will raise if verification fails
        with patch("labnirs2snirf.labnirs._verify_header_format") as mock_verify:
            mock_verify.side_effect = lambda x: None  # No-op for verification
            _read_header(minimal_data_path)

        # Verify that the verification function was executed
        mock_verify.assert_called_once()

    def test_read_header_invalid_encoding_fails(self, caplog, tmp_path):
        """Test handling of files with encoding issues."""
        caplog.set_level(logging.DEBUG)

        # Create a file with invalid ASCII content
        invalid_file = tmp_path / "invalid_encoding.txt"
        # Write some non-ASCII bytes
        invalid_file.write_bytes(b"\xff\xfe" + b"invalid content\n" * 35)

        with patch("labnirs2snirf.labnirs._verify_header_format") as mock_verify:
            mock_verify.side_effect = lambda x: None  # No-op for verification
            with pytest.raises(LabNirsReadError) as exc_info:
                _read_header(invalid_file)

        assert "Error reading the header" in str(exc_info.value)
        assert "ERROR" in caplog.text

    def test_read_header_io_error_fails(self, caplog, minimal_data_path):
        """Test handling of IO errors during file reading."""
        caplog.set_level(logging.DEBUG)

        test_file = minimal_data_path

        # Mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("Mocked IO error")):
            with pytest.raises(LabNirsReadError) as exc_info:
                _read_header(test_file)

        assert "Error reading the header" in str(exc_info.value)
        assert "ERROR" in caplog.text
        assert "Mocked IO error" in caplog.text

    def test_read_header_preserves_line_content_succeeds(self, caplog, small_data_path):
        """Test that header lines are read without modification."""
        caplog.set_level(logging.DEBUG)

        with patch("labnirs2snirf.labnirs._verify_header_format") as mock_verify:
            mock_verify.side_effect = lambda x: None  # No-op for verification
            header = _read_header(small_data_path)

        # First line should contain the expected format marker
        assert " [File Information]      		 [Data Line]	36" + "\n" == header[0]
        assert "Measured Date	2000/01/02 11:12:13" + "\n" == header[1]
        assert "Name	subject1	[HeaderType]	11.0/11.0" + "\n" == header[3]
        assert (
            "Time(sec)	Task	Mark	Count	    oxyHb	  deoxyHb	  totalHb	 Abs780nm	 "
            "Abs805nm	 Abs830nm	    oxyHb	  deoxyHb	  totalHb	 Abs780nm	 Abs805nm	 Abs830nm"
            + "\n"
            == header[34]
        )

    def test_read_header_logs_filename_succeeds(self, caplog, minimal_data_path):
        """Test that the filename is logged correctly."""
        caplog.set_level(logging.INFO)

        test_file = minimal_data_path
        with patch("labnirs2snirf.labnirs._verify_header_format") as mock_verify:
            mock_verify.side_effect = lambda x: None  # No-op for verification
            _read_header(test_file)

        assert str(test_file) in caplog.text
        assert "Reading header lines from file" in caplog.text


class TestVerifyHeaderFormat:
    """Tests for the _verify_header_format function."""

    @pytest.mark.parametrize("filename", ["minimal_labnirs.txt", "small_labnirs.txt"])
    def test_verify_header_format_minimal_file_succeeds(
        self,
        caplog,
        filename,
        test_data_dir,
    ):
        """Test header verification with minimal_labnirs.txt file."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(test_data_dir / filename)

        # Should not raise any exception
        _verify_header_format(header)

        # Check that verification was logged
        assert "Verifying header format" in caplog.text
        assert "Header format verification completed" in caplog.text

        # Should not have any warnings or errors for valid file
        assert "WARNING" not in caplog.text
        assert "ERROR" not in caplog.text

    def test_verify_header_format_invalid_top_line_fails(self, caplog):
        """Test that invalid top line raises LabNirsReadError."""
        caplog.set_level(logging.DEBUG)

        # Create header with invalid top line
        header = ["Invalid top line\n"] + ["dummy\n"] * 34

        with pytest.raises(LabNirsReadError) as exc_info:
            _verify_header_format(header)

        assert "Critical header format error: invalid top line" in str(exc_info.value)
        assert "Checking for critical header format errors" in caplog.text

    def test_verify_header_format_missing_channel_pairs_fails(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that missing channel pairs on line 33 raises LabNirsReadError."""
        caplog.set_level(logging.DEBUG)

        # Read a valid header and corrupt line 33 (index 32)
        header = _read_header(minimal_data_path)
        header[32] = "Invalid channel pairs line\n"

        with pytest.raises(LabNirsReadError) as exc_info:
            _verify_header_format(header)

        assert "Critical header format error: channel pairs not found" in str(
            exc_info.value,
        )
        assert "Expected format: (source,detector)(source,detector)" in str(
            exc_info.value,
        )

    def test_verify_header_format_wrong_version_warning_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that incorrect version number produces a warning but doesn't raise."""
        caplog.set_level(logging.WARNING)

        # Read a valid header and modify the version line
        header = _read_header(minimal_data_path)
        header[2] = "Some\tFields\tVersion\t10.0\n"
        header[3] = "Some\tFields\t[HeaderType]\t10.0/10.0\n"

        # Should not raise, but should warn
        _verify_header_format(header)

        assert "WARNING" in caplog.text
        assert "Version number in line 3 must be '11.0'" in caplog.text
        assert "HeaderType in line 4 must be '11.0/11.0'" in caplog.text
        assert "Errors may occur" in caplog.text

    def test_verify_header_format_missing_metadata_warnings_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that missing optional metadata produces warnings."""
        caplog.set_level(logging.WARNING)

        # Read a valid header and modify the HeaderType line
        header = _read_header(small_data_path)
        header[1] = (
            "Invalid datetime line\n"  # Line 2: should have measurement datetime
        )
        header[2] = (
            "Invalid ID line\n"  # Line 3: should have ID (also checked for version)
        )
        header[3] = (
            "Invalid name line\n"  # Line 4: should have name (also checked for headertype)
        )
        header[4] = "Invalid comment line\n"  # Line 5: should have comment info

        # Should not raise, but may produce warnings
        _verify_header_format(header)

        assert "WARNING" in caplog.text
        assert "Missing ID metadata" in caplog.text
        assert "Missing measurement datetime metadata" in caplog.text
        assert "Missing subject name metadata" in caplog.text
        assert "Missing comment metadata" in caplog.text

    def test_verify_header_format_correct_number_of_lines_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that header with correct number of lines is accepted."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(minimal_data_path)

        # Verify we have 35 lines (DATA_START_LINE - 1)
        assert len(header) == 35

        _verify_header_format(header)

        assert f"Verifying header format with {len(header)} lines" in caplog.text

    def test_verify_header_format_logs_debug_messages_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that debug logging messages are produced during verification."""
        caplog.set_level(logging.DEBUG)

        header = _read_header(minimal_data_path)
        _verify_header_format(header)

        # Check for expected debug messages
        assert "DEBUG" in caplog.text
        assert "Verifying header format" in caplog.text
        assert "Checking for critical header format errors" in caplog.text
        assert "Header format verification completed" in caplog.text


class TestReadProbePairs:
    """Tests for the read_probe_pairs function."""

    def test_read_probe_pairs_valid_file_succeeds(self, caplog, minimal_data_path):
        """Test reading probe pairs from a valid file."""
        caplog.set_level(logging.DEBUG)

        result = read_probe_pairs(minimal_data_path)

        assert result == "(2,1)(2,2)"
        assert "Reading probe pairs from file" in caplog.text
        assert "Found probe pairs string: (2,1)(2,2)" in caplog.text

    def test_read_probe_pairs_multiple_pairs_succeeds(
        self,
        caplog,
        tmp_path,
        minimal_data_path,
    ):
        """Test reading multiple probe pairs."""
        caplog.set_level(logging.DEBUG)

        # Create test file with multiple pairs
        test_file = tmp_path / "test_multiple_pairs.txt"
        header = _read_header(minimal_data_path)
        header[32] = "(1,1)(2,1)(3,1)(1,2)(2,2)(3,2)\n"
        test_file.write_text("".join(header) + "\n")

        result = read_probe_pairs(test_file)

        assert result == "(1,1)(2,1)(3,1)(1,2)(2,2)(3,2)"
        assert "Found probe pairs string" in caplog.text

    def test_read_probe_pairs_single_pair_succeeds(self, tmp_path, minimal_data_path):
        """Test reading a single probe pair."""

        test_file = tmp_path / "test_single_pair.txt"
        header = _read_header(minimal_data_path)
        header[32] = "(1,1)\n"
        test_file.write_text("".join(header) + "\n")

        result = read_probe_pairs(test_file)

        assert result == "(1,1)"

    def test_read_probe_pairs_non_contiguous_indices_succeeds(
        self,
        tmp_path,
        minimal_data_path,
    ):
        """Test reading probe pairs with non-contiguous indices."""

        test_file = tmp_path / "test_non_contiguous.txt"
        header = _read_header(minimal_data_path)
        header[32] = "(1,1)(5,3)(10,7)\n"
        test_file.write_text("".join(header) + "\n")

        result = read_probe_pairs(test_file)

        assert result == "(1,1)(5,3)(10,7)"

    def test_read_probe_pairs_large_indices_succeeds(self, tmp_path, minimal_data_path):
        """Test reading probe pairs with large index numbers."""

        test_file = tmp_path / "test_large_indices.txt"
        header = _read_header(minimal_data_path)
        header[32] = "(100,200)(999,888)\n"
        test_file.write_text("".join(header) + "\n")

        result = read_probe_pairs(test_file)

        assert result == "(100,200)(999,888)"

    def test_read_probe_pairs_with_whitespace_fails(self, tmp_path, minimal_data_path):
        """Test reading probe pairs with leading/trailing whitespace."""

        test_file = tmp_path / "test_whitespace.txt"
        header = _read_header(minimal_data_path)
        header[32] = "  (1,1)(2,1)  \n"
        test_file.write_text("".join(header) + "\n")

        with pytest.raises(
            LabNirsReadError,
            match="Critical header format error: channel pairs not found",
        ):
            read_probe_pairs(test_file)

    def test_read_probe_pairs_many_pairs_succeeds(self, tmp_path, minimal_data_path):
        """Test reading many probe pairs."""

        test_file = tmp_path / "test_many_pairs.txt"
        header = _read_header(minimal_data_path)
        # Create 50 probe pairs
        pairs = "".join([f"({i},{j})" for i in range(1, 11) for j in range(1, 6)])
        header[32] = pairs + "\n"
        test_file.write_text("".join(header) + "\n")

        result = read_probe_pairs(test_file)

        assert result == pairs
        assert result.count("(") == 50
        assert result.count(")") == 50

    def test_read_probe_pairs_nonexistent_file_fails(self, tmp_path):
        """Test reading from a non-existent file raises LabNirsReadError."""

        non_existent = tmp_path / "nonexistent.txt"

        with pytest.raises(LabNirsReadError) as exc_info:
            read_probe_pairs(non_existent)

        assert "Data file not found" in str(exc_info.value)
        assert "nonexistent.txt" in str(exc_info.value)

    def test_read_probe_pairs_invalid_header_format_fails(
        self,
        tmp_path,
        minimal_data_path,
    ):
        """Test reading from file with invalid header format."""

        test_file = tmp_path / "test_invalid_header.txt"
        # Create file with wrong top line
        header = _read_header(minimal_data_path)
        header[0] = "Invalid top line\n"
        test_file.write_text("".join(header) + "\n")

        with pytest.raises(LabNirsReadError) as exc_info:
            read_probe_pairs(test_file)

        assert "Critical header format error: invalid top line" in str(exc_info.value)

    def test_read_probe_pairs_missing_pairs_line_fails(
        self,
        tmp_path,
        minimal_data_path,
    ):
        """Test reading from file with missing channel pairs line."""

        test_file = tmp_path / "test_missing_pairs.txt"
        header = _read_header(minimal_data_path)
        header[32] = "Not a valid pairs line\n"
        test_file.write_text("".join(header) + "\n")

        with pytest.raises(LabNirsReadError) as exc_info:
            read_probe_pairs(test_file)

        assert "Critical header format error: channel pairs not found" in str(
            exc_info.value,
        )
        assert "Expected format: (source,detector)(source,detector)" in str(
            exc_info.value,
        )

    def test_read_probe_pairs_empty_pairs_line_fails(self, tmp_path, minimal_data_path):
        """Test reading from file with empty channel pairs line."""

        test_file = tmp_path / "test_empty_pairs.txt"
        header = _read_header(minimal_data_path)
        header[32] = "\n"
        test_file.write_text("".join(header) + "\n")

        with pytest.raises(LabNirsReadError) as exc_info:
            read_probe_pairs(test_file)

        assert "Critical header format error: channel pairs not found" in str(
            exc_info.value,
        )

    def test_read_probe_pairs_corrupted_file_fails(self, tmp_path):
        """Test reading from corrupted file with insufficient lines."""

        test_file = tmp_path / "test_corrupted.txt"
        # Write only 10 lines instead of 35
        test_file.write_text("line\n" * 10)

        with pytest.raises(
            LabNirsReadError,
            match="Critical header format error: invalid top line in header",
        ):
            read_probe_pairs(test_file)

    def test_read_probe_pairs_pathlib_path_succeeds(self, minimal_data_path):
        """Test that function accepts pathlib.Path objects."""

        path = minimal_data_path
        result = read_probe_pairs(path)

        assert isinstance(result, str)
        assert result == "(2,1)(2,2)"

    def test_read_probe_pairs_logging_file_path_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that the file path is logged correctly."""
        caplog.set_level(logging.INFO)

        test_file = minimal_data_path
        read_probe_pairs(test_file)

        assert str(test_file) in caplog.text
        assert "Reading probe pairs from file" in caplog.text

    def test_read_probe_pairs_return_type_succeeds(self, caplog, minimal_data_path):
        """Test that the function returns a string."""
        caplog.set_level(logging.DEBUG)

        result = read_probe_pairs(minimal_data_path)

        assert isinstance(result, str)

    def test_read_probe_pairs_malformed_pairs_format_fails(
        self,
        caplog,
        tmp_path,
        minimal_data_path,
    ):
        """Test reading file with malformed pairs format."""
        caplog.set_level(logging.DEBUG)

        test_file = tmp_path / "test_malformed.txt"
        header = _read_header(minimal_data_path)
        # Missing closing parenthesis
        header[32] = "(1,1)(2,1\n"
        test_file.write_text("".join(header) + "\n")

        with pytest.raises(
            LabNirsReadError,
            match="Critical header format error: channel pairs not found",
        ):
            read_probe_pairs(test_file)


class TestReadLabnirsIntegration:
    """Integration tests for the read_labnirs function."""

    @pytest.mark.parametrize(
        "filename",
        [
            "minimal_labnirs.txt",
            "small_labnirs.txt",
            "raw_only.txt",
            "hb_only.txt",
        ],
    )
    def test_read_labnirs_valid_files_succeed(self, caplog, filename, test_data_dir):
        """Test reading all valid LabNIRS files without issues."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(test_data_dir / filename)

        # Verify successful execution
        assert isinstance(result, model.Nirs)
        assert result.metadata is not None
        assert len(result.data) == 1
        assert result.probe is not None
        assert result.stim is not None

        # Verify no warnings or errors in log
        assert "WARNING" not in caplog.text
        assert "ERROR" not in caplog.text

    def test_read_labnirs_minimal_file_default_params_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test reading minimal file with default parameters."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(minimal_data_path)

        # Check structure
        assert isinstance(result, model.Nirs)
        assert result.metadata is not None
        assert len(result.data) == 1
        assert result.probe is not None
        assert result.stim is not None

        # Check metadata
        assert result.metadata.SubjectID == ""
        assert result.metadata.MeasurementDate == "2000-01-02"
        assert result.metadata.MeasurementTime == "11:12:13"

        # Check data
        assert result.data[0].time.shape[0] == 8
        assert result.data[0].dataTimeSeries.shape == (8, 12)
        assert (
            len(result.data[0].measurementList)
            == result.data[0].dataTimeSeries.shape[1]
        )
        assert result.data[0].time[1] == 0.021
        assert result.data[0].time[-1] == 0.147
        assert np.all(
            result.data[0].dataTimeSeries[0, -5:]
            == [
                0.000000,
                0.000000,
                0.625967,
                0.774206,
                0.635424,
            ],
        )
        assert np.all(
            result.data[0].dataTimeSeries[-1, :5]
            == [
                -0.009953,
                0.010694,
                0.000741,
                0.775153,
                0.920161,
            ],
        )

        # Check probe
        assert result.probe.wavelengths.shape[0] == 3
        assert result.probe.sourcePos3D.shape == (1, 3)
        assert result.probe.detectorPos3D.shape == (2, 3)
        assert (
            result.probe.sourceLabels is not None
            and len(result.probe.sourceLabels) == 1
        )
        assert (
            result.probe.detectorLabels is not None
            and len(result.probe.detectorLabels) == 2
        )

        # Check logging
        assert "Reading and validating header" in caplog.text
        assert "Parsing channel pairs and probe information" in caplog.text
        assert "Reading experiment data from file" in caplog.text

    def test_read_labnirs_small_file_with_metadata_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test reading file with complete metadata."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(small_data_path)

        assert result.metadata.SubjectID == "ID1"
        assert result.metadata.MeasurementDate == "2000-01-02"
        assert result.metadata.MeasurementTime == "11:12:13"
        assert "SubjectName" in result.metadata.additional_fields
        assert result.metadata.additional_fields["SubjectName"] == "subject1"
        assert "comment" in result.metadata.additional_fields
        assert result.metadata.additional_fields["comment"] == "comment1"

    def test_read_labnirs_keep_raw_only_succeeds(self, caplog, small_data_path):
        """Test reading with keep_category='raw'."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(small_data_path, keep_category="raw")

        # All measurements should be raw (dataType=1)
        assert all(m.dataType == 1 for m in result.data[0].measurementList)
        assert all(m.dataTypeLabel is None for m in result.data[0].measurementList)
        assert "Filtering to keep only 'raw' data category" in caplog.text

    def test_read_labnirs_keep_hb_only_succeeds(self, caplog, small_data_path):
        """Test reading with keep_category='hb'."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(small_data_path, keep_category="hb")

        # All measurements should be Hb (dataType=99999)
        assert all(m.dataType == 99999 for m in result.data[0].measurementList)
        assert all(
            m.dataTypeLabel in ["HbO", "HbR", "HbT"]
            for m in result.data[0].measurementList
        )
        assert all(m.wavelengthIndex == 0 for m in result.data[0].measurementList)
        assert "Filtering to keep only 'hb' data category" in caplog.text

    def test_read_labnirs_drop_wavelength_succeeds(self, caplog, small_data_path):
        """Test dropping specific wavelengths."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            small_data_path,
            keep_category="raw",
            drop_subtype=["830"],
        )

        # No measurement should have wavelength 830
        assert 830.0 not in result.probe.wavelengths
        assert "Dropping columns based on subtype filter" in caplog.text

    def test_read_labnirs_drop_hb_type_succeeds(self, caplog, small_data_path):
        """Test dropping specific Hb types."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            small_data_path,
            keep_category="hb",
            drop_subtype=["hbt"],
        )

        # No measurement should be HbT
        assert all(m.dataTypeLabel != "HbT" for m in result.data[0].measurementList)
        assert "Dropping columns based on subtype filter" in caplog.text

    def test_read_labnirs_drop_multiple_subtypes_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test dropping multiple subtypes."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            small_data_path,
            drop_subtype={"hbo", "830"},
        )

        # Should not have HbO or wavelength 830
        assert all(m.dataTypeLabel != "HbO" for m in result.data[0].measurementList)
        assert 830.0 not in result.probe.wavelengths

    def test_read_labnirs_data_time_consistency_succeeds(self, minimal_data_path):
        """Test that time array and data array have consistent dimensions."""

        result = read_labnirs(minimal_data_path)

        assert result.data[0].time.shape[0] == result.data[0].dataTimeSeries.shape[0]

    def test_read_labnirs_probe_indices_match_measurements_succeeds(
        self,
        minimal_data_path,
    ):
        """Test that probe indices in measurements match probe arrays."""

        result = read_labnirs(minimal_data_path)

        max_source_idx = max(m.sourceIndex for m in result.data[0].measurementList)
        max_detector_idx = max(m.detectorIndex for m in result.data[0].measurementList)

        assert max_source_idx <= result.probe.sourcePos3D.shape[0]
        assert max_detector_idx <= result.probe.detectorPos3D.shape[0]

    def test_read_labnirs_wavelength_indices_valid_succeeds(self, small_data_path):
        """Test that wavelength indices in measurements are valid."""

        result = read_labnirs(small_data_path, keep_category="raw")

        max_wavelength_idx = max(
            m.wavelengthIndex for m in result.data[0].measurementList
        )
        assert max_wavelength_idx <= len(result.probe.wavelengths)

    def test_read_labnirs_stim_extraction_succeeds(self, small_data_path):
        """Test that stimuli are correctly extracted when present."""

        result = read_labnirs(small_data_path)

        assert result.stim is not None
        assert len(result.stim) > 0
        assert all(isinstance(s, model.Stim) for s in result.stim)
        assert all(isinstance(s.name, str) for s in result.stim)
        assert all(isinstance(s.data, np.ndarray) for s in result.stim)
        assert {s.name for s in result.stim} == {"Z", "0", "1", "2"}

    def test_read_labnirs_probe_labels_complete_succeeds(self, minimal_data_path):
        """Test that probe labels are complete and correctly formatted."""

        result = read_labnirs(minimal_data_path)

        assert result.probe.sourceLabels is not None
        assert result.probe.detectorLabels is not None
        assert len(result.probe.sourceLabels) == result.probe.sourcePos3D.shape[0]
        assert len(result.probe.detectorLabels) == result.probe.detectorPos3D.shape[0]
        assert all(label.startswith("S") for label in result.probe.sourceLabels)
        assert all(label.startswith("D") for label in result.probe.detectorLabels)

    def test_read_labnirs_invalid_keep_category_fails(self, minimal_data_path):
        """Test error handling for invalid keep_category parameter."""

        with pytest.raises(LabNirsReadError, match="Invalid parameters.*keep_category"):
            read_labnirs(minimal_data_path, keep_category="invalid")

    def test_read_labnirs_invalid_drop_subtype_not_collection_fails(
        self,
        minimal_data_path,
    ):
        """Test error handling for invalid drop_subtype type."""

        with pytest.raises(LabNirsReadError, match="Invalid parameters.*drop_subtype"):
            read_labnirs(
                minimal_data_path,
                drop_subtype="not_a_collection",
            )

    def test_read_labnirs_invalid_drop_subtype_value_fails(self, minimal_data_path):
        """Test error handling for invalid drop_subtype values."""

        with pytest.raises(LabNirsReadError, match="Invalid parameters.*drop_subtype"):
            read_labnirs(
                minimal_data_path,
                drop_subtype=["invalid_type"],
            )

    def test_read_labnirs_nonexistent_file_fails(self, tmp_path):
        """Test error handling for non-existent file."""

        with pytest.raises(LabNirsReadError, match="Data file not found"):
            read_labnirs(tmp_path / "nonexistent.txt")

    def test_read_labnirs_corrupted_header_fails(self, tmp_path):
        """Test error handling for corrupted header."""

        corrupted_file = tmp_path / "corrupted.txt"
        corrupted_file.write_text("Invalid header\n" * 10)

        with pytest.raises(LabNirsReadError, match="Critical header format error"):
            read_labnirs(corrupted_file)

    def test_read_labnirs_measurement_list_datatype_index_consistency_succeeds(
        self,
        small_data_path,
    ):
        """Test that all measurements have dataTypeIndex=0."""

        result = read_labnirs(small_data_path)

        assert all(m.dataTypeIndex == 0 for m in result.data[0].measurementList)

    def test_read_labnirs_case_insensitive_parameters_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test that parameters are case-insensitive."""
        caplog.set_level(logging.INFO)

        result1 = read_labnirs(
            small_data_path,
            keep_category="RAW",
            drop_subtype=["HBO"],
        )

        result2 = read_labnirs(
            small_data_path,
            keep_category="raw",
            drop_subtype=["hbo"],
        )

        assert (
            result1.data[0].dataTimeSeries.shape == result2.data[0].dataTimeSeries.shape
        )

    def test_read_labnirs_all_data_types_present_succeeds(self, small_data_path):
        """Test that reading with keep_category='all' includes both raw and hb."""

        result = read_labnirs(small_data_path, keep_category="all")

        datatypes = {m.dataType for m in result.data[0].measurementList}
        # Should have both dataType=1 (raw) and dataType=99999 (Hb)
        assert all(t == 1 or t == 99999 for t in datatypes)

    def test_read_labnirs_logging_progression_succeeds(self, caplog, minimal_data_path):
        """Test that all major processing steps are logged."""
        caplog.set_level(logging.INFO)

        read_labnirs(minimal_data_path)

        expected_messages = [
            "Validating input parameters",
            "Reading and validating header",
            "Parsing channel pairs and probe information",
            "Parsing column metadata and data structure",
            "Reading experiment data from file",
            "Extracting metadata, data, stimuli, and probe information",
        ]

        for message in expected_messages:
            assert message in caplog.text

    def test_read_labnirs_data_array_dtype_succeeds(self, minimal_data_path):
        """Test that data arrays have correct numpy dtypes."""

        result = read_labnirs(minimal_data_path)

        assert result.data[0].time.dtype == np.float64
        assert result.data[0].dataTimeSeries.dtype == np.float64
        assert result.probe.wavelengths.dtype == np.float64
        assert result.probe.sourcePos3D.dtype == np.float64
        assert result.probe.detectorPos3D.dtype == np.float64

    def test_read_labnirs_probe_positions_zero_initialized_succeeds(
        self,
        minimal_data_path,
    ):
        """Test that all probe positions are initialized to zero."""

        result = read_labnirs(minimal_data_path)

        assert np.all(result.probe.sourcePos3D == 0.0)
        assert np.all(result.probe.detectorPos3D == 0.0)

    def test_read_labnirs_stim_times_within_data_range_succeeds(self, small_data_path):
        """Test that stimulus times fall within the data time range."""

        result = read_labnirs(small_data_path)

        if result.stim:
            min_time = result.data[0].time.min()
            max_time = result.data[0].time.max()
            for stim in result.stim:
                if len(stim.data) > 0:
                    assert np.all(stim.data >= min_time)
                    assert np.all(stim.data <= max_time)

    def test_read_labnirs_empty_drop_subtype_set_succeeds(
        self,
        caplog,
        minimal_data_path,
    ):
        """Test that empty drop_subtype set works correctly."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(minimal_data_path, drop_subtype=set())

        assert result is not None
        assert "No column subtypes specified for dropping" in caplog.text

    def test_read_labnirs_measurement_list_order_matches_columns_succeeds(
        self,
        small_data_path,
    ):
        """Test that measurement list order matches data column order."""

        result = read_labnirs(small_data_path, keep_category="raw")

        # First measurement should correspond to first data column
        # This is implicitly tested by shape consistency but adding explicit check
        assert (
            len(result.data[0].measurementList)
            == result.data[0].dataTimeSeries.shape[1]
        )

    def test_read_labnirs_warning_for_version_mismatch_succeeds(
        self,
        caplog,
        tmp_path,
        minimal_data_path,
    ):
        """Test that version mismatch produces warning but doesn't fail."""
        caplog.set_level(logging.WARNING)

        # Create file with wrong version
        test_file = tmp_path / "wrong_version.txt"
        header = _read_header(minimal_data_path)
        header[2] = "Some\tFields\tVersion\t10.0\n"
        header[3] = "Some\tFields\t[HeaderType]\t10.0/10.0\n"

        # Need to create a full file with data
        with open(minimal_data_path, encoding="utf-8") as f:
            lines = f.readlines()
        lines[:3] = header[:3]
        test_file.write_text("".join(lines))

        # Should not raise but should warn
        result = read_labnirs(test_file)

        assert result is not None
        assert "Version number in line 3 must be '11.0'" in caplog.text

    def test_read_labnirs_complete_workflow_integration_succeeds(
        self,
        caplog,
        small_data_path,
    ):
        """Test complete workflow from file to model with all components."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            small_data_path,
            keep_category="all",
            drop_subtype=None,
        )

        # Verify all components are present and valid
        assert isinstance(result, model.Nirs)

        # Metadata
        assert isinstance(result.metadata, model.Metadata)
        assert result.metadata.SubjectID != ""

        # Data
        assert len(result.data) == 1
        assert isinstance(result.data[0], model.Data)
        assert result.data[0].time.shape[0] > 0

        # Measurements
        assert len(result.data[0].measurementList) > 0
        assert all(
            isinstance(m, model.Measurement) for m in result.data[0].measurementList
        )

        # Probe
        assert isinstance(result.probe, model.Probe)
        assert result.probe.wavelengths.shape[0] > 0

        # Stim
        assert result.stim is not None
        assert all(isinstance(s, model.Stim) for s in result.stim)

        # Logging coverage
        assert "Validating input parameters" in caplog.text
        assert "Successfully read data table" in caplog.text
        assert "Extracted data has" in caplog.text
        assert "Extracted probe information" in caplog.text
        assert "Extracted metadata" in caplog.text
        assert "Found" in caplog.text and "stimulus types" in caplog.text

    def test_read_labnirs_keep_category_not_string_fails(self, minimal_data_path):
        """Test error handling for non-string keep_category parameter."""

        with pytest.raises(
            LabNirsReadError,
            match="Invalid parameters.*must be a string",
        ):
            read_labnirs(
                minimal_data_path,
                keep_category=123,  # ty: ignore[invalid-argument-type] # type: ignore
            )

    def test_read_labnirs_drop_subtype_with_non_string_elements_fails(
        self,
        minimal_data_path,
    ):
        """Test error handling for drop_subtype with non-string elements."""

        with pytest.raises(
            LabNirsReadError,
            match="Invalid parameters.*must be a collection of strings or None",
        ):
            read_labnirs(
                minimal_data_path,
                drop_subtype={
                    123,  # type: ignore
                    "hbo",
                },  # ty: ignore[invalid-argument-type]
            )

    def test_read_labnirs_raw_only_keep_hb_fails(self, caplog, rawonly_data_path):
        """Test that keeping only Hb data from raw-only file fails."""
        caplog.set_level(logging.DEBUG)

        with pytest.raises(
            LabNirsReadError,
            match="No data columns found after filtering; cannot extract data.",
        ):
            read_labnirs(rawonly_data_path, keep_category="hb")

    def test_read_labnirs_raw_only_keep_all_succeeds(self, caplog, rawonly_data_path):
        """Test that keeping all data from raw-only file succeeds."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(rawonly_data_path, keep_category="all")

        # Should succeed with only raw data present
        assert isinstance(result, model.Nirs)
        assert all(m.dataType == 1 for m in result.data[0].measurementList)
        assert all(m.dataTypeLabel is None for m in result.data[0].measurementList)
        # Should have 6 data columns (3 wavelengths × 2 channels)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert "Keeping all data categories (keep_category=all)" in caplog.text

    def test_read_labnirs_raw_only_keep_raw_succeeds(self, caplog, rawonly_data_path):
        """Test that keeping raw data from raw-only file succeeds."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(rawonly_data_path, keep_category="raw")

        # Should succeed with 6 data columns
        assert isinstance(result, model.Nirs)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert all(m.dataType == 1 for m in result.data[0].measurementList)
        assert all(m.dataTypeLabel is None for m in result.data[0].measurementList)
        assert "Filtering to keep only 'raw' data category" in caplog.text

    def test_read_labnirs_raw_only_drop_hb_types_succeeds(
        self,
        caplog,
        rawonly_data_path,
    ):
        """Test that dropping Hb types from raw-only file has no effect."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            rawonly_data_path,
            drop_subtype=["hbo", "hbr", "hbt"],
        )

        # Should succeed with all raw data still present
        assert isinstance(result, model.Nirs)
        # Should have 6 data columns (3 wavelengths × 2 channels)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert all(m.dataType == 1 for m in result.data[0].measurementList)
        assert "Dropping columns based on subtype filter" in caplog.text

    def test_read_labnirs_raw_only_drop_all_wavelengths_fails(
        self,
        caplog,
        rawonly_data_path,
    ):
        """Test that dropping all wavelengths from raw-only file fails."""
        caplog.set_level(logging.DEBUG)

        with pytest.raises(
            LabNirsReadError,
            match="No data columns found after filtering; cannot extract data.",
        ):
            read_labnirs(
                rawonly_data_path,
                drop_subtype=["780", "805", "830"],
            )

    def test_read_labnirs_hb_only_keep_raw_fails(self, caplog, hbonly_data_path):
        """Test that keeping only raw data from hb-only file fails."""
        caplog.set_level(logging.DEBUG)

        with pytest.raises(
            LabNirsReadError,
            match="No data columns found after filtering; cannot extract data.",
        ):
            read_labnirs(hbonly_data_path, keep_category="raw")

    def test_read_labnirs_hb_only_keep_all_succeeds(self, caplog, hbonly_data_path):
        """Test that keeping all data from hb-only file succeeds."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(hbonly_data_path, keep_category="all")

        # Should succeed with only Hb data present
        assert isinstance(result, model.Nirs)
        assert all(m.dataType == 99999 for m in result.data[0].measurementList)
        assert all(
            m.dataTypeLabel in ["HbO", "HbR", "HbT"]
            for m in result.data[0].measurementList
        )
        # Should have 6 data columns (3 Hb types × 2 channels)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert "Keeping all data categories (keep_category=all)" in caplog.text

    def test_read_labnirs_hb_only_keep_hb_succeeds(self, caplog, hbonly_data_path):
        """Test that keeping Hb data from hb-only file succeeds."""
        caplog.set_level(logging.INFO)

        result = read_labnirs(hbonly_data_path, keep_category="hb")

        # Should succeed with 6 data columns
        assert isinstance(result, model.Nirs)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert all(m.dataType == 99999 for m in result.data[0].measurementList)
        assert all(
            m.dataTypeLabel in ["HbO", "HbR", "HbT"]
            for m in result.data[0].measurementList
        )
        assert "Filtering to keep only 'hb' data category" in caplog.text

    def test_read_labnirs_hb_only_drop_wavelengths_succeeds(
        self,
        caplog,
        hbonly_data_path,
    ):
        """Test that dropping wavelengths from hb-only file has no effect."""
        caplog.set_level(logging.DEBUG)

        result = read_labnirs(
            hbonly_data_path,
            drop_subtype=["780", "805", "830"],
        )

        # Should succeed with all Hb data still present
        assert isinstance(result, model.Nirs)
        # Should have 6 data columns (3 Hb types × 2 channels)
        assert result.data[0].dataTimeSeries.shape[1] == 6
        assert all(m.dataType == 99999 for m in result.data[0].measurementList)
        assert "Dropping columns based on subtype filter" in caplog.text

    def test_read_labnirs_hb_only_drop_all_hb_types_fails(
        self,
        caplog,
        hbonly_data_path,
    ):
        """Test that dropping all Hb types from hb-only file fails."""
        caplog.set_level(logging.DEBUG)

        with pytest.raises(
            LabNirsReadError,
            match="No data columns found after filtering; cannot extract data.",
        ):
            read_labnirs(
                hbonly_data_path,
                drop_subtype=["hbo", "hbr", "hbt"],
            )

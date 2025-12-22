"""
Unit tests for layout.py module.
"""

import logging

import numpy as np
import polars as pl
import pytest

from labnirs2snirf.layout import LayoutError, read_layout, update_layout
from labnirs2snirf.model import Data, Measurement, Metadata, Nirs, Probe


# Fixtures for creating test data
@pytest.fixture(name="sample_layout_3d")
def fixture_sample_layout_3d():
    """Sample 3D layout dictionary."""
    return {
        "S1": (10.0, 20.0, 30.0),
        "S2": (11.0, 21.0, 31.0),
        "D1": (15.0, 25.0, 35.0),
        "D2": (16.0, 26.0, 36.0),
    }


@pytest.fixture(name="base_nirs")
def fixture_base_nirs():
    """Create a minimal mutable Nirs object that tests can modify."""
    probe = Probe(
        wavelengths=np.array([780.0, 850.0]),
        sourcePos3D=np.zeros((2, 3)),
        detectorPos3D=np.zeros((2, 3)),
        sourceLabels=["S1", "S2"],
        detectorLabels=["D1", "D2"],
    )
    metadata = Metadata(
        SubjectID="test",
        MeasurementDate="2024-01-01",
        MeasurementTime="12:00:00",
    )
    measurement = Measurement(
        sourceIndex=1,
        detectorIndex=1,
        dataType=1,
        dataTypeIndex=1,
        wavelengthIndex=1,
    )
    data = Data(
        time=np.array([0.0, 1.0]),
        dataTimeSeries=np.zeros((2, 1)),
        measurementList=[measurement],
    )
    return Nirs(metadata=metadata, data=[data], probe=probe)


# Tests for read_layout function
class TestReadLayout:
    """Tests for the read_layout function."""

    def test_read_valid_layout_succeeds(self, tmp_path):
        """Test reading a valid layout file."""
        layout_file = tmp_path / "valid_layout.sfp"
        layout_file.write_text(
            "S1\t10.0\t20.0\t30.0\nD1\t15.0\t25.0\t35.0\nS2\t11.0\t21.0\t31.0\nD2\t16.0\t26.0\t36.0",
        )

        layout = read_layout(layout_file)

        assert len(layout) == 4
        assert "S1" in layout
        assert "D1" in layout
        assert "S2" in layout
        assert "D2" in layout
        assert layout["S1"] == (10.0, 20.0, 30.0)
        assert layout["D1"] == (15.0, 25.0, 35.0)
        assert layout["S2"] == (11.0, 21.0, 31.0)
        assert layout["D2"] == (16.0, 26.0, 36.0)

    def test_read_layout_with_whitespace_succeeds(self, tmp_path):
        """Test that whitespace is properly stripped from labels."""
        layout_file = tmp_path / "whitespace_layout.sfp"
        layout_file.write_text(" S1 \t10.0\t20.0\t30.0\n  D1  \t15.0\t25.0\t35.0")

        layout = read_layout(layout_file)

        assert len(layout) == 2
        assert "S1" in layout
        assert "D1" in layout
        assert " S1 " not in layout
        assert "  D1  " not in layout
        assert layout["S1"] == (10.0, 20.0, 30.0)

    def test_read_empty_layout_fails(self, tmp_path, caplog):
        """Test reading an empty layout file raises NoDataError and logs exception."""
        layout_file = tmp_path / "empty_layout.sfp"
        layout_file.write_text("")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(pl.exceptions.NoDataError):
                read_layout(layout_file)

        # Check that exception was logged
        assert any(
            "Failed to read layout file" in record.message for record in caplog.records
        )

    def test_read_single_optode_succeeds(self, tmp_path):
        """Test reading a layout file with a single optode."""
        layout_file = tmp_path / "single.sfp"
        layout_file.write_text("S1\t1.0\t2.0\t3.0")

        layout = read_layout(layout_file)

        assert len(layout) == 1
        assert layout["S1"] == (1.0, 2.0, 3.0)

    def test_read_duplicate_labels_succeeds(self, tmp_path, caplog):
        """Test reading a layout file with duplicate labels (last one wins)."""
        layout_file = tmp_path / "duplicate.sfp"
        layout_file.write_text("S1\t1.0\t2.0\t3.0\nS1\t10.0\t20.0\t30.0")

        with caplog.at_level(logging.WARNING):
            layout = read_layout(layout_file)

        assert len(layout) == 1
        assert layout["S1"] == (10.0, 20.0, 30.0)

        # Check that warning was logged about duplicate labels
        assert any(
            "Duplicate labels found" in record.message and "S1" in record.message
            for record in caplog.records
        )

    def test_read_negative_coordinates_succeeds(self, tmp_path):
        """Test reading layout with negative coordinates."""
        layout_file = tmp_path / "negative.sfp"
        layout_file.write_text("S1\t-10.5\t-20.3\t-30.7")

        layout = read_layout(layout_file)

        assert layout["S1"] == (-10.5, -20.3, -30.7)

    def test_read_zero_coordinates_succeeds(self, tmp_path):
        """Test reading layout with zero coordinates."""
        layout_file = tmp_path / "zero.sfp"
        layout_file.write_text("S1\t0.0\t0.0\t0.0")

        layout = read_layout(layout_file)

        assert layout["S1"] == (0.0, 0.0, 0.0)

    def test_read_large_coordinates_succeeds(self, tmp_path):
        """Test reading layout with large coordinate values."""
        layout_file = tmp_path / "large.sfp"
        layout_file.write_text("S1\t1000.5\t2000.3\t3000.7")

        layout = read_layout(layout_file)

        assert layout["S1"] == (1000.5, 2000.3, 3000.7)

    def test_read_scientific_notation_succeeds(self, tmp_path):
        """Test reading layout with scientific notation."""
        layout_file = tmp_path / "scientific.sfp"
        layout_file.write_text("S1\t1.5e-3\t2.3e2\t-4.7e-1")

        layout = read_layout(layout_file)

        assert layout["S1"] == (1.5e-3, 2.3e2, -4.7e-1)

    def test_read_mixed_label_types_succeeds(self, tmp_path):
        """Test reading layout with various label formats."""
        layout_file = tmp_path / "mixed.sfp"
        layout_file.write_text(
            "S1\t1.0\t2.0\t3.0\n"
            "D1\t4.0\t5.0\t6.0\n"
            "Source2\t7.0\t8.0\t9.0\n"
            "Detector_3\t10.0\t11.0\t12.0",
        )

        layout = read_layout(layout_file)

        assert len(layout) == 4
        assert "S1" in layout
        assert "Source2" in layout
        assert "Detector_3" in layout

    def test_read_nonexistent_file_fails(self, tmp_path):
        """Test reading a file that doesn't exist raises exception."""
        nonexistent = tmp_path / "nonexistent.sfp"

        with pytest.raises(LayoutError):
            read_layout(nonexistent)

    def test_read_malformed_layout_fails(self, tmp_path):
        """Test reading a malformed layout file raises exception."""
        layout_file = tmp_path / "malformed_layout.sfp"
        layout_file.write_text("S1\t10.0\t20.0\nD1\t15.0")

        with pytest.raises(LayoutError):
            read_layout(layout_file)

    def test_read_non_numeric_coordinates_fails(self, tmp_path):
        """Test reading layout with non-numeric coordinates raises exception."""
        layout_file = tmp_path / "non_numeric.sfp"
        layout_file.write_text("S1\tabc\t20.0\t30.0\nD1\t15.0\txyz\t35.0")

        with pytest.raises(LayoutError):
            read_layout(layout_file)

    def test_read_missing_columns_fails(self, tmp_path):
        """Test reading layout with missing columns raises exception."""
        layout_file = tmp_path / "missing_cols.sfp"
        layout_file.write_text("S1\t10.0")

        with pytest.raises(LayoutError):
            read_layout(layout_file)

    def test_read_extra_columns_ignored_fails(self, tmp_path):
        """Test reading layout with extra columns raises exception."""
        layout_file = tmp_path / "extra_cols.sfp"
        layout_file.write_text("S1\t10.0\t20.0\t30.0\t40.0\t50.0")

        # This might raise an exception depending on polars' strict mode
        with pytest.raises(LayoutError):
            read_layout(layout_file)

    def test_read_layout_logs_debug_messages_succeeds(self, tmp_path, caplog):
        """Test that reading a layout file logs DEBUG level messages."""
        layout_file = tmp_path / "valid_layout.sfp"
        layout_file.write_text("S1\t10.0\t20.0\t30.0\nD1\t15.0\t25.0\t35.0")

        with caplog.at_level(logging.DEBUG):
            layout = read_layout(layout_file)

        assert len(layout) == 2
        # Check that debug messages were logged
        assert any("Reading layout file" in record.message for record in caplog.records)
        assert any(
            "Successfully read" in record.message
            and "optode positions" in record.message
            for record in caplog.records
        )

    def test_read_layout_logs_exception(self, tmp_path, caplog):
        """Test that exceptions during layout reading are logged."""
        layout_file = tmp_path / "nonexistent.sfp"

        with caplog.at_level(logging.ERROR):
            with pytest.raises(LayoutError):
                read_layout(layout_file)

        # Check that exception was logged
        assert any(
            "Layout file not found" in record.message for record in caplog.records
        )

    def test_read_layout_with_whitespace_in_coordinates_fails(self, tmp_path):
        """Test reading layout with whitespace around coordinates raises ComputeError."""
        layout_file = tmp_path / "whitespace_coords.sfp"
        layout_file.write_text("S1\t 10.0 \t 20.0 \t 30.0 \nD1\t15.0\t 25.0\t35.0")

        with pytest.raises(LayoutError):
            read_layout(layout_file)

    def test_read_layout_with_varied_coordinate_types_succeeds(self, tmp_path):
        """Test reading layout with negative, integer, and zero coordinate values."""
        layout_file = tmp_path / "varied_coords.sfp"
        layout_file.write_text(
            "S1\t-10.5\t0\t30.0\n"
            "D1\t15\t-25.3\t0.0\n"
            "S2\t0.0\t100\t-50\n"
            "D2\t-0.1\t-0.2\t-0.3",
        )

        layout = read_layout(layout_file)

        assert len(layout) == 4
        assert layout["S1"] == (-10.5, 0.0, 30.0)
        assert layout["D1"] == (15.0, -25.3, 0.0)
        assert layout["S2"] == (0.0, 100.0, -50.0)
        assert layout["D2"] == (-0.1, -0.2, -0.3)


# Tests for update_layout function
class TestUpdateLayout:
    """Tests for the update_layout function."""

    def test_update_all_positions_succeeds(self, base_nirs, sample_layout_3d):
        """Test updating all source and detector positions."""
        update_layout(base_nirs, sample_layout_3d)

        # Check sources
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            sample_layout_3d["S1"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[1, :],
            sample_layout_3d["S2"],
        )

        # Check detectors
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[0, :],
            sample_layout_3d["D1"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[1, :],
            sample_layout_3d["D2"],
        )

    def test_update_partial_positions_succeeds(self, base_nirs):
        """Test updating only some positions (missing labels in layout)."""
        partial_layout = {
            "S1": (10.0, 20.0, 30.0),
            "D2": (16.0, 26.0, 36.0),
        }

        update_layout(base_nirs, partial_layout)

        # Check updated source
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            partial_layout["S1"],
        )

        # Check non-updated source (should remain zero)
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[1, :],
            [0.0, 0.0, 0.0],
        )

        # Check non-updated detector (should remain zero)
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[0, :],
            [0.0, 0.0, 0.0],
        )

        # Check updated detector
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[1, :],
            partial_layout["D2"],
        )

    def test_update_with_empty_layout_succeeds(self, base_nirs):
        """Test updating with an empty layout dictionary."""
        empty_layout = {}
        update_layout(base_nirs, empty_layout)

        # All positions should remain zero
        np.testing.assert_array_equal(base_nirs.probe.sourcePos3D, np.zeros((2, 3)))
        np.testing.assert_array_equal(base_nirs.probe.detectorPos3D, np.zeros((2, 3)))

    def test_update_without_probe_labels_fails(self, base_nirs, sample_layout_3d):
        """Test that updating without probe labels raises LayoutError."""
        # Modify base_nirs to remove labels
        nirs_no_labels = Nirs(
            metadata=base_nirs.metadata,
            data=base_nirs.data,
            probe=Probe(
                wavelengths=base_nirs.probe.wavelengths,
                sourcePos3D=base_nirs.probe.sourcePos3D,
                detectorPos3D=base_nirs.probe.detectorPos3D,
                sourceLabels=None,
                detectorLabels=None,
            ),
        )

        with pytest.raises(LayoutError) as exc_info:
            update_layout(nirs_no_labels, sample_layout_3d)

        assert "no probe labels" in str(exc_info.value).lower()

    def test_update_with_only_source_labels_succeeds(self, base_nirs, sample_layout_3d):
        """Test updating when only source labels are present."""
        # Modify base_nirs to have only source labels
        nirs_source_only = Nirs(
            metadata=base_nirs.metadata,
            data=base_nirs.data,
            probe=Probe(
                wavelengths=base_nirs.probe.wavelengths,
                sourcePos3D=base_nirs.probe.sourcePos3D,
                detectorPos3D=base_nirs.probe.detectorPos3D,
                sourceLabels=base_nirs.probe.sourceLabels,
                detectorLabels=None,
            ),
        )

        update_layout(nirs_source_only, sample_layout_3d)

        # Check sources are updated
        np.testing.assert_array_equal(
            nirs_source_only.probe.sourcePos3D[0, :],
            sample_layout_3d["S1"],
        )
        np.testing.assert_array_equal(
            nirs_source_only.probe.sourcePos3D[1, :],
            sample_layout_3d["S2"],
        )

        # Detectors should remain zero
        np.testing.assert_array_equal(
            nirs_source_only.probe.detectorPos3D,
            np.zeros_like(nirs_source_only.probe.detectorPos3D),
        )

    def test_update_with_only_detector_labels_succeeds(
        self,
        base_nirs,
        sample_layout_3d,
    ):
        """Test updating when only detector labels are present."""
        # Modify base_nirs to have only detector labels
        nirs_detector_only = Nirs(
            metadata=base_nirs.metadata,
            data=base_nirs.data,
            probe=Probe(
                wavelengths=base_nirs.probe.wavelengths,
                sourcePos3D=base_nirs.probe.sourcePos3D,
                detectorPos3D=base_nirs.probe.detectorPos3D,
                sourceLabels=None,
                detectorLabels=base_nirs.probe.detectorLabels,
            ),
        )

        update_layout(nirs_detector_only, sample_layout_3d)

        # Sources should remain zero
        np.testing.assert_array_equal(
            nirs_detector_only.probe.sourcePos3D,
            np.zeros_like(nirs_detector_only.probe.sourcePos3D),
        )

        # Check detectors are updated
        np.testing.assert_array_equal(
            nirs_detector_only.probe.detectorPos3D[0, :],
            sample_layout_3d["D1"],
        )
        np.testing.assert_array_equal(
            nirs_detector_only.probe.detectorPos3D[1, :],
            sample_layout_3d["D2"],
        )

    def test_update_with_extra_layout_labels_succeeds(self, base_nirs):
        """Test updating with layout containing extra labels not in data."""
        layout_with_extra = {
            "S1": (10.0, 20.0, 30.0),
            "S2": (11.0, 21.0, 31.0),
            "S3": (12.0, 22.0, 32.0),  # Extra source
            "D1": (15.0, 25.0, 35.0),
            "D2": (16.0, 26.0, 36.0),
            "D3": (17.0, 27.0, 37.0),  # Extra detector
        }

        update_layout(base_nirs, layout_with_extra)

        # Should only update existing probes
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            layout_with_extra["S1"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[1, :],
            layout_with_extra["S2"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[0, :],
            layout_with_extra["D1"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[1, :],
            layout_with_extra["D2"],
        )

        # Verify that only 2 sources and 2 detectors exist (no S3 or D3 in data)
        assert base_nirs.probe.sourcePos3D.shape[0] == 2
        assert base_nirs.probe.detectorPos3D.shape[0] == 2
        assert len(base_nirs.probe.sourceLabels) == 2
        assert len(base_nirs.probe.detectorLabels) == 2

    def test_update_with_negative_coordinates_succeeds(self, base_nirs):
        """Test updating with negative coordinates."""
        layout_negative = {
            "S1": (-10.0, -20.0, -30.0),
            "D1": (-15.0, -25.0, -35.0),
        }

        update_layout(base_nirs, layout_negative)

        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            layout_negative["S1"],
        )
        np.testing.assert_array_equal(
            base_nirs.probe.detectorPos3D[0, :],
            layout_negative["D1"],
        )

    def test_update_preserves_other_probe_attributes_succeeds(
        self,
        base_nirs,
        sample_layout_3d,
    ):
        """Test that updating positions doesn't modify other probe attributes."""
        original_wavelengths = base_nirs.probe.wavelengths.copy()
        original_source_labels = base_nirs.probe.sourceLabels.copy()
        original_detector_labels = base_nirs.probe.detectorLabels.copy()

        update_layout(base_nirs, sample_layout_3d)

        np.testing.assert_array_equal(base_nirs.probe.wavelengths, original_wavelengths)
        assert base_nirs.probe.sourceLabels == original_source_labels
        assert base_nirs.probe.detectorLabels == original_detector_labels

    def test_update_single_probe_succeeds(self, base_nirs):
        """Test updating a single probe position."""
        single_probe_layout = {"S1": (100.0, 200.0, 300.0)}

        update_layout(base_nirs, single_probe_layout)

        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            single_probe_layout["S1"],
        )
        # Others should remain zero
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[1, :],
            [0.0, 0.0, 0.0],
        )

    def test_update_case_sensitive_labels_succeeds(self, base_nirs):
        """Test that label matching is case-sensitive."""
        case_layout = {
            "s1": (10.0, 20.0, 30.0),  # lowercase - should not match
            "S1": (100.0, 200.0, 300.0),  # uppercase - should match
        }

        update_layout(base_nirs, case_layout)

        # Should match "S1" (uppercase) from probe labels
        np.testing.assert_array_equal(
            base_nirs.probe.sourcePos3D[0, :],
            case_layout["S1"],
        )

    def test_update_modifies_in_place_succeeds(self, base_nirs, sample_layout_3d):
        """Test that update_layout modifies the Nirs object in place."""
        original_id = id(base_nirs)
        original_probe_id = id(base_nirs.probe)

        result = update_layout(  # pylint: disable=assignment-from-no-return
            base_nirs,
            sample_layout_3d,
        )

        # Function returns None
        assert result is None
        # Object identity should be unchanged
        assert id(base_nirs) == original_id
        assert id(base_nirs.probe) == original_probe_id

    def test_update_with_large_number_of_probes_succeeds(self, base_nirs):
        """Test updating with many probes."""
        num_probes = 100
        source_labels = [f"S{i + 1}" for i in range(num_probes)]
        detector_labels = [f"D{i + 1}" for i in range(num_probes)]

        probe = Probe(
            wavelengths=np.array([780.0]),
            sourcePos3D=np.zeros((num_probes, 3)),
            detectorPos3D=np.zeros((num_probes, 3)),
            sourceLabels=source_labels,
            detectorLabels=detector_labels,
        )
        nirs = Nirs(metadata=base_nirs.metadata, data=base_nirs.data, probe=probe)

        # Create layout for all probes
        layout = {
            **{
                label: (float(i), float(i * 2), float(i * 3))
                for i, label in enumerate(source_labels)
            },
            **{
                label: (float(i + 100), float(i * 2 + 100), float(i * 3 + 100))
                for i, label in enumerate(detector_labels)
            },
        }

        update_layout(nirs, layout)

        # Verify a few positions using layout references
        np.testing.assert_array_equal(nirs.probe.sourcePos3D[0, :], layout["S1"])
        np.testing.assert_array_equal(nirs.probe.sourcePos3D[50, :], layout["S51"])
        np.testing.assert_array_equal(nirs.probe.detectorPos3D[0, :], layout["D1"])
        np.testing.assert_array_equal(nirs.probe.detectorPos3D[50, :], layout["D51"])

    def test_update_logs_debug_and_info_messages_succeeds(
        self,
        base_nirs,
        sample_layout_3d,
        caplog,
    ):
        """Test that updating layout logs DEBUG and INFO level messages."""
        with caplog.at_level(logging.DEBUG):
            update_layout(base_nirs, sample_layout_3d)

        # Check that debug message was logged
        assert any(
            "Updating probe positions" in record.message for record in caplog.records
        )
        # Check that info message about updated positions was logged
        assert any(
            f"Updated positions for {len(base_nirs.probe.sourceLabels)} sources and {len(base_nirs.probe.detectorLabels)} detectors"
            in record.message
            for record in caplog.records
        )

    def test_update_logs_warnings_for_missing_labels_succeeds(self, base_nirs, caplog):
        """Test that warnings are logged for probes missing position data."""
        partial_layout = {"S1": (10.0, 20.0, 30.0)}  # Missing S2, D1, D2

        with caplog.at_level(logging.WARNING):
            update_layout(base_nirs, partial_layout)

        # Check that warnings were logged for missing labels
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        # Get the actual missing labels from the probe
        missing_sources = [
            label
            for label in base_nirs.probe.sourceLabels
            if label not in partial_layout
        ]
        missing_detectors = [
            label
            for label in base_nirs.probe.detectorLabels
            if label not in partial_layout
        ]

        for label in missing_sources:
            assert any(
                label in msg and "missing position data" in msg
                for msg in warning_messages
            )
        for label in missing_detectors:
            assert any(
                label in msg and "missing position data" in msg
                for msg in warning_messages
            )

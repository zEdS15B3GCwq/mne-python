# type: ignore
# pylint: disable=E1101
"""
Integration tests for labnirs2snirf.py module.
"""

import logging
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from labnirs2snirf.labnirs2snirf import main


@pytest.fixture(name="output_snirf")
def fixture_output_snirf(tmp_path):
    """Provide temporary output SNIRF file path."""
    return tmp_path / "output.snirf"


@pytest.fixture(name="layout_file")
def fixture_layout_file(tmp_path):
    """Create a temporary layout file."""
    layout = tmp_path / "layout.sfp"
    layout.write_text("S2\t10.0\t20.0\t30.0\nD1\t15.0\t25.0\t35.0\n", encoding="utf-8")
    return layout


@pytest.fixture(autouse=True, name="cleanup_logging")
def fixture_cleanup_logging():
    """Clean up logging handlers after each test."""
    yield
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


class TestMainSuccessfulConversion:
    """Tests for successful conversion scenarios."""

    def test_main_minimal_file_default_args_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test successful conversion with minimal file and default arguments."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf)],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        # Verify SNIRF file structure
        with h5py.File(output_snirf, "r") as f:
            assert "formatVersion" in f
            assert "/nirs" in f
            assert "/nirs/metaDataTags" in f
            assert "/nirs/data1" in f
            assert "/nirs/probe" in f
            np.testing.assert_array_equal(
                f["/nirs/data1/time"],
                np.array([0.0, 0.021, 0.042, 0.063, 0.084, 0.105, 0.126, 0.147]),
            )
            np.testing.assert_array_equal(
                f["/nirs/probe/wavelengths"],
                [780.0, 805.0, 830.0],
            )
            assert (
                f["/nirs/metaDataTags/MeasurementDate"][()].decode("utf-8")
                == "2000-01-02"
            )
            assert (
                f["/nirs/metaDataTags/MeasurementTime"][()].decode("utf-8")
                == "11:12:13"
            )
            assert f["/nirs/data1/dataTimeSeries"][()].shape == (8, 12)
            assert f["/nirs/probe/sourcePos3D"][()].shape == (1, 3)
            assert f["/nirs/probe/detectorPos3D"][()].shape == (2, 3)
            assert f["/nirs/probe/sourceLabels"][0].decode("utf-8") == "S2"
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_small_file_succeeds(self, output_snirf, small_data_path):
        """Test successful conversion with small file containing metadata."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf)],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        # Verify metadata was extracted
        with h5py.File(output_snirf, "r") as f:
            assert "formatVersion" in f
            assert "/nirs" in f
            assert "/nirs/metaDataTags" in f
            assert "/nirs/data1" in f
            assert "/nirs/probe" in f
            assert f["/nirs/metaDataTags/SubjectID"][()].decode("utf-8") == "ID1"
            assert f["/nirs/metaDataTags/SubjectName"][()].decode("utf-8") == "subject1"
            assert f["/nirs/metaDataTags/comment"][()].decode("utf-8") == "comment1"
            np.testing.assert_array_equal(
                f["/nirs/probe/wavelengths"],
                [780.0, 805.0, 830.0],
            )
            assert (
                f["/nirs/metaDataTags/MeasurementDate"][()].decode("utf-8")
                == "2000-01-02"
            )
            assert (
                f["/nirs/metaDataTags/MeasurementTime"][()].decode("utf-8")
                == "11:12:13"
            )
            assert f["/nirs/data1/dataTimeSeries"][()].shape == (8, 12)
            assert f["/nirs/probe/sourcePos3D"][()].shape == (1, 3)
            assert f["/nirs/probe/detectorPos3D"][()].shape == (2, 3)
            assert f["/nirs/probe/sourceLabels"][0].decode("utf-8") == "S2"

    def test_main_default_output_filename_succeeds(
        self,
        tmp_path,
        monkeypatch,
        minimal_data_path,
    ):
        """Test conversion with default output filename."""
        monkeypatch.chdir(tmp_path)

        with patch.object(sys, "argv", ["prog", str(minimal_data_path)]):
            result = main()

        assert result == 0
        default_output = tmp_path / "out.snirf"
        assert default_output.exists()

    def test_main_with_locations_file_succeeds(
        self,
        output_snirf,
        layout_file,
        minimal_data_path,
    ):
        """Test successful conversion with probe locations file."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(minimal_data_path),
                str(output_snirf),
                "--locations",
                str(layout_file),
            ],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        # Verify positions were updated
        # minimal data has S2 source and D1,D2 detectors; layout_file has S2 and D1 - we can check S2 and D1
        with h5py.File(output_snirf, "r") as f:
            np.testing.assert_array_equal(
                f["/nirs/probe/detectorPos3D"][0, :],
                [15.0, 25.0, 35.0],
            )
            np.testing.assert_array_equal(
                f["/nirs/probe/sourcePos3D"][0, :],
                [10.0, 20.0, 30.0],
            )

    def test_main_type_raw_only_succeeds(self, output_snirf, small_data_path):
        """Test conversion keeping only raw data."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf), "--type", "raw"],
        ):
            result = main()

        assert result == 0

        # Verify only raw data (dataType=1) is present
        with h5py.File(output_snirf, "r") as f:
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                assert ml["dataType"][()] == 1

    def test_main_type_hb_only_succeeds(
        self,
        output_snirf,
        small_data_path,
    ):
        """Test conversion keeping only hemoglobin data."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf), "--type", "hb"],
        ):
            result = main()

        assert result == 0

        # Verify only Hb data (dataType=99999) is present
        with h5py.File(output_snirf, "r") as f:
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                assert ml["dataType"][()] == 99999
                assert "dataTypeLabel" in ml

    def test_main_type_all_succeeds(self, output_snirf, small_data_path):
        """Test conversion keeping all data types."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf), "--type", "all"],
        ):
            result = main()

        assert result == 0

        # Verify both raw and Hb data are present
        with h5py.File(output_snirf, "r") as f:
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            datatypes = set()
            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                datatypes.add(ml["dataType"][()])

            assert 1 in datatypes  # raw
            assert 99999 in datatypes  # Hb

    def test_main_drop_single_wavelength_succeeds(self, output_snirf, small_data_path):
        """Test conversion dropping a specific wavelength."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--type",
                "raw",
                "--drop",
                "830",
            ],
        ):
            result = main()

        assert result == 0

        # Verify wavelength 830 is not present
        with h5py.File(output_snirf, "r") as f:
            wavelengths = f["/nirs/probe/wavelengths"][:]
            assert 830.0 not in wavelengths

    def test_main_drop_single_hb_type_succeeds(self, output_snirf, small_data_path):
        """Test conversion dropping a specific hemoglobin type."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--type",
                "hb",
                "--drop",
                "hbt",
            ],
        ):
            result = main()

        assert result == 0

        # Verify HbT is not present
        with h5py.File(output_snirf, "r") as f:
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                if "dataTypeLabel" in ml:
                    assert ml["dataTypeLabel"][()].decode("utf-8") != "HbT"

    def test_main_drop_multiple_types_succeeds(self, output_snirf, small_data_path):
        """Test conversion dropping multiple data types."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--drop",
                "hbo",
                "--drop",
                "830",
            ],
        ):
            result = main()

        assert result == 0

        with h5py.File(output_snirf, "r") as f:
            # Check wavelength
            wavelengths = f["/nirs/probe/wavelengths"][:]
            assert 830.0 not in wavelengths

            # Check HbO
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                if "dataTypeLabel" in ml:
                    assert ml["dataTypeLabel"][()].decode("utf-8") != "HbO"

    def test_main_case_insensitive_args_succeeds(self, output_snirf, small_data_path):
        """Test that type and drop arguments are case-insensitive."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--type",
                "HB",
                "--drop",
                "HbT",
            ],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

    def test_main_all_options_combined_succeeds(
        self,
        output_snirf,
        layout_file,
        small_data_path,
    ):
        """Test conversion with all optional arguments specified."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--locations",
                str(layout_file),
                "--type",
                "hb",
                "--drop",
                "hbt",
                "-vv",
                "--log",
            ],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        # Verify all options were applied
        with h5py.File(output_snirf, "r") as f:
            # Check locations
            source_pos = f["/nirs/probe/sourcePos3D"][0, :]
            assert source_pos[0] == 10.0

            # Check type filtering
            data_group = f["/nirs/data1"]
            ml_count = len(
                [k for k in data_group.keys() if k.startswith("measurementList")],
            )

            for i in range(1, ml_count + 1):
                ml = data_group[f"measurementList{i}"]
                assert ml["dataType"][()] == 99999
                if "dataTypeLabel" in ml:
                    assert ml["dataTypeLabel"][()].decode("utf-8") != "HbT"

    def test_main_logging_successful_conversion_at_debug_level_succeeds(
        self,
        output_snirf,
        layout_file,
        caplog,
        capsys,
        small_data_path,
    ):
        """Test that all expected log messages are produced during successful conversion."""
        with caplog.at_level(logging.DEBUG):
            with patch.object(
                sys,
                "argv",
                [
                    "prog",
                    str(small_data_path),
                    str(output_snirf),
                    "--locations",
                    str(layout_file),
                    "-vvv",
                ],
            ):
                result = main()

        assert result == 0
        assert output_snirf.exists()
        capsys.readouterr()  # prevent output from appearing in test output

        # Check for expected log messages produced by main() function only
        expected_messages = [
            ("Logger configured", logging.INFO),
            ("Parsed arguments:", logging.DEBUG),
            ("Reading labNIRS data", logging.INFO),
            ("Reading probe layout from file", logging.INFO),
            ("Writing SNIRF file", logging.INFO),
            ("Successfully completed conversion", logging.INFO),
        ]

        for message_text, expected_level in expected_messages:
            assert any(
                message_text in record.message
                and record.levelname == logging.getLevelName(expected_level)
                for record in caplog.records
            ), (
                f"Expected log message not found: {message_text} at {logging.getLevelName(expected_level)} level"
            )

    def test_running_module_in_subprocess_succeeds(
        self,
        output_snirf,
        minimal_data_path,
    ):
        """Test running the module as __main__ to cover sys.exit(main())."""
        # Import the module to trigger __main__ execution

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "labnirs2snirf.labnirs2snirf",
                str(minimal_data_path),
                str(output_snirf),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert output_snirf.exists()

    def test_running_module_with_runpy_succeeds(self, output_snirf, minimal_data_path):
        """Test running the module as __main__ to cover sys.exit(main())."""
        # Execute the __main__ block by running the module code
        # Need to remove the module from sys.modules to avoid the warning
        module_name = "labnirs2snirf.labnirs2snirf"
        saved_module = sys.modules.pop(module_name, None)

        try:
            with patch.object(
                sys,
                "argv",
                ["prog", str(minimal_data_path), str(output_snirf)],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    runpy.run_module(module_name, run_name="__main__")

            assert exc_info.value.code == 0
            assert output_snirf.exists()
        finally:
            # Restore the module if it was there before
            if saved_module is not None:
                sys.modules[module_name] = saved_module

    def test_running_module_through_main_with_runpy_succeeds(
        self,
        output_snirf,
        minimal_data_path,
    ):
        """Test running the module with module name, using __main__.py."""
        module_main_name = "labnirs2snirf"

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf)],
        ):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_module(module_main_name, run_name="__main__")

        assert exc_info.value.code == 0
        assert output_snirf.exists()

    def test_main_output_validates_with_pysnirf2_succeeds(
        self,
        output_snirf,
        small_data_path,
    ):
        """Test that converted SNIRF file passes pysnirf2 validation."""
        from snirf import validateSnirf

        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf)],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        # Validate using pysnirf2
        validation_result = validateSnirf(str(output_snirf))
        assert validation_result


class TestMainVerbosityLevels:
    """Tests for different verbosity levels."""

    def test_main_no_verbosity_no_console_output_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that no verbosity produces no console output."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf)],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_main_single_v_warning_level_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that single -v shows WARNING level messages."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-v"],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should not show DEBUG or INFO
        assert "DEBUG" not in output
        assert "INFO" not in output
        assert "Logger configured" not in output

    def test_main_double_v_info_level_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that -vv shows INFO level messages."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-vv"],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should show INFO
        assert "Logger configured" in output
        assert "Reading labNIRS data" in output

        # Should not show DEBUG
        assert "Parsed arguments:" not in output

    def test_main_triple_v_debug_level_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that -vvv shows DEBUG level messages."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-vvv"],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should show DEBUG
        assert "Parsed arguments:" in output
        assert "Reading header lines from file" in output

    def test_main_log_flag_creates_log_file_succeeds(
        self,
        output_snirf,
        tmp_path,
        monkeypatch,
        minimal_data_path,
    ):
        """Test that --log flag creates a log file."""
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "--log", "-vv"],
        ):
            result = main()

        assert result == 0
        log_file = tmp_path / "labnirs2snirf.log"
        assert log_file.exists()

        content = log_file.read_text()
        assert "Logger configured" in content
        assert "Successfully completed conversion" in content

    def test_main_log_with_verbosity_succeeds(
        self,
        output_snirf,
        tmp_path,
        monkeypatch,
        minimal_data_path,
    ):
        """Test that --log combined with -v works correctly."""
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "--log", "-vvv"],
        ):
            result = main()

        assert result == 0
        log_file = tmp_path / "labnirs2snirf.log"
        content = log_file.read_text()

        # Should have DEBUG messages in log file
        assert "DEBUG" in content
        assert "Parsed arguments:" in content


class TestMainArgumentErrors:
    """Tests for argument validation errors."""

    def test_main_nonexistent_source_file_fails(self, tmp_path, capsys):
        """Test that nonexistent source file produces error."""
        nonexistent = tmp_path / "nonexistent.txt"

        with patch.object(sys, "argv", ["prog", str(nonexistent), "output.snirf"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Argument error:" in captured.out
        assert "does not exist" in captured.out

    def test_main_existing_target_file_fails(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that existing target file produces error."""
        output_snirf.write_text("existing content")

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf)],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Argument error:" in captured.out
        assert "already exists" in captured.out

    def test_main_nonexistent_locations_file_fails(
        self,
        output_snirf,
        tmp_path,
        capsys,
        minimal_data_path,
    ):
        """Test that nonexistent locations file produces error."""
        nonexistent = tmp_path / "nonexistent.sfp"

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(minimal_data_path),
                str(output_snirf),
                "--locations",
                str(nonexistent),
            ],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Argument error:" in captured.out

    def test_main_invalid_type_argument_fails(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that invalid --type argument fails."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "--type", "invalid"],
        ):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        assert "invalid choice" in captured.err

    def test_main_invalid_drop_argument_fails(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that invalid --drop argument produces error."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(minimal_data_path),
                str(output_snirf),
                "--drop",
                "invalid_type",
            ],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Argument error:" in captured.out
        assert "Invalid drop type" in captured.out

    def test_main_no_arguments_shows_help_fails(self, capsys):
        """Test that no arguments shows help and exits."""
        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                result = main()
                assert result == 0

        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "Convert LabNIRS data to SNIRF format" in captured.out

    def test_main_help_flag_shows_help_succeeds(self, capsys):
        """Test that -h flag shows help."""
        with patch.object(sys, "argv", ["prog", "-h"]):
            with pytest.raises(SystemExit):
                result = main()
                assert result == 0

        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "--locations" in captured.out
        assert "--type" in captured.out
        assert "--drop" in captured.out


class TestMainConversionErrors:
    """Tests for errors during conversion process."""

    def test_main_corrupted_header_fails(self, tmp_path, output_snirf, capsys):
        """Test that corrupted header file produces conversion error."""
        corrupted = tmp_path / "corrupted.txt"
        corrupted.write_text("Invalid header\n" * 10)

        with patch.object(
            sys,
            "argv",
            ["prog", str(corrupted), str(output_snirf), "-v"],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Conversion failed:" in captured.out
        assert "Critical header format error" in captured.out

    def test_main_invalid_layout_file_fails(
        self,
        output_snirf,
        tmp_path,
        capsys,
        minimal_data_path,
    ):
        """Test that invalid layout file produces conversion error."""
        bad_layout = tmp_path / "bad_layout.sfp"
        bad_layout.write_text("S1\tabc\tdef\tghi\n")  # Non-numeric coordinates

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(minimal_data_path),
                str(output_snirf),
                "--locations",
                str(bad_layout),
                "-v",
            ],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Conversion failed:" in captured.out

    def test_main_empty_data_file_fails(self, tmp_path, output_snirf, capsys):
        """Test that empty data file produces conversion error."""
        empty = tmp_path / "empty.txt"
        empty.write_text("")

        with patch.object(sys, "argv", ["prog", str(empty), str(output_snirf), "-v"]):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Conversion failed:" in captured.out

    def test_main_simulated_exception_without_logger_fails(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test exception handling when logger not configured."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf)],
        ):
            with patch("labnirs2snirf.args.Arguments.parse") as mock_parse:
                mock_parse.side_effect = RuntimeError("Simulated error before logger")

                with pytest.raises(RuntimeError, match="Simulated error before logger"):
                    result = main()
                    assert result == 1

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Something went wrong" in output
        assert "Logging not configured" in output

    def test_main_simulated_exception_with_logger_fails(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test exception handling when logger is configured."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-v"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = RuntimeError("Simulated error after logger")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Something went wrong" in output
        assert (
            "Exception received. Error message: Simulated error after logger" in output
        )


class TestMainEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_main_output_without_snirf_extension_succeeds(
        self,
        tmp_path,
        caplog,
        capsys,
        minimal_data_path,
    ):
        """Test that output without .snirf extension produces warning."""
        output = tmp_path / "output.hdf5"

        with caplog.at_level(logging.WARNING):
            with patch.object(
                sys,
                "argv",
                ["prog", str(minimal_data_path), str(output), "-v"],
            ):
                result = main()

        capsys.readouterr()  # prevent output from appearing in test output
        assert result == 0
        assert output.exists()
        assert any(
            "doesn't have the .snirf extension" in record.message
            for record in caplog.records
        )

    def test_main_relative_paths_succeeds(
        self,
        output_snirf,
        monkeypatch,
        test_data_dir,
    ):
        """Test conversion with relative paths."""
        monkeypatch.chdir(test_data_dir.parent)

        relative_input = Path("test") / "minimal_labnirs.txt"
        output = output_snirf

        with patch.object(sys, "argv", ["prog", str(relative_input), str(output)]):
            result = main()

        assert result == 0
        assert output_snirf.exists()

    def test_main_drop_all_data_types_raises_error(
        self,
        output_snirf,
        capsys,
        small_data_path,
    ):
        """Test that dropping all available data types produces error."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--type",
                "hb",
                "--drop",
                "hbo",
                "--drop",
                "hbr",
                "--drop",
                "hbt",
                "-v",
            ],
        ):
            result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "Conversion failed:" in captured.out

    def test_main_duplicate_drop_values_succeeds(self, output_snirf, small_data_path):
        """Test that duplicate drop values are handled correctly."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                str(small_data_path),
                str(output_snirf),
                "--drop",
                "hbt",
                "--drop",
                "HbT",
                "--drop",
                "hbt",
            ],
        ):
            result = main()

        assert result == 0
        # Should deduplicate internally

    def test_main_multiple_verbosity_flags_succeeds(
        self,
        output_snirf,
        capsys,
        minimal_data_path,
    ):
        """Test that multiple -v flags accumulate correctly."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-v", "-v", "-v"],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should show DEBUG level
        assert "Parsed arguments:" in output


class TestMainLoggingIntegration:
    """Tests for logging integration with file and console."""

    def test_main_log_file_contains_detailed_info_succeeds(
        self,
        output_snirf,
        tmp_path,
        monkeypatch,
        small_data_path,
    ):
        """Test that log file contains detailed information with timestamps."""
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            ["prog", str(small_data_path), str(output_snirf), "--log", "-vvv"],
        ):
            result = main()

        assert result == 0

        log_file = tmp_path / "labnirs2snirf.log"
        content = log_file.read_text()

        # Should contain timestamps
        assert any(char.isdigit() for char in content.split("INFO")[0])

        # Should contain function names
        assert "main" in content

        # Should contain detailed messages
        assert "Parsed arguments:" in content
        assert "Successfully completed conversion" in content

    def test_main_console_vs_file_logging_format_succeeds(
        self,
        output_snirf,
        tmp_path,
        monkeypatch,
        capsys,
        minimal_data_path,
    ):
        """Test that console and file logging have different formats."""
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "--log", "-vv"],
        ):
            result = main()

        assert result == 0
        log_file = tmp_path / "labnirs2snirf.log"
        file_content = log_file.read_text()

        assert output_snirf.exists()
        output_snirf.unlink()  # Clean up
        capsys.readouterr()

        # File should have timestamps and function names
        assert "Logger configured" in file_content
        assert any(char.isdigit() for char in file_content.split("INFO")[0])
        assert "main" in file_content

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "-vv"],
        ):
            result = main()

        assert result == 0
        assert output_snirf.exists()

        captured = capsys.readouterr()
        console_output = captured.out + captured.err

        # Console should not have timestamps before module name
        assert "Logger configured" in console_output

        # File format includes more details
        assert len(file_content) > len(console_output)

    def test_main_exception_logged_to_file_with_traceback_succeeds(
        self,
        output_snirf,
        tmp_path,
        monkeypatch,
        capsys,
        minimal_data_path,
    ):
        """Test that exceptions are logged to file with full traceback."""
        monkeypatch.chdir(tmp_path)

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(output_snirf), "--log", "-v"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = RuntimeError("Test exception for logging")

                result = main()

        assert result == 1

        log_file = tmp_path / "labnirs2snirf.log"
        content = log_file.read_text()

        # Should contain exception details
        assert "Test exception for logging" in content
        assert "Exception received" in content
        assert "Traceback" in content or "ERROR" in content

        # Console should have user-friendly message
        captured = capsys.readouterr()
        assert "Something went wrong" in captured.out

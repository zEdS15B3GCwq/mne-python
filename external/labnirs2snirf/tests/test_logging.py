"""Tests for logging functionality."""

import logging
import sys
from unittest.mock import patch

import pytest

from labnirs2snirf.args import ArgumentError
from labnirs2snirf.error import Labnirs2SnirfError
from labnirs2snirf.labnirs import read_labnirs
from labnirs2snirf.labnirs2snirf import main
from labnirs2snirf.log import LOGFILE_NAME, config_logger


@pytest.fixture(name="outfile_path")
def fixture_outfile_path(tmp_path):
    """Provide temporary output file path and clean up after test."""
    outfile = tmp_path / "out.snirf"
    yield outfile


@pytest.fixture(autouse=True, name="cleanup_logging")
def fixture_cleanup_logging():
    """Clean up logging handlers after each test."""
    yield
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture(name="logfile_path")
def fixture_logfile_path(tmp_path, monkeypatch):
    """Provide temporary log file path and clean up after test."""
    logfile = tmp_path / LOGFILE_NAME
    monkeypatch.chdir(tmp_path)
    yield logfile


class TestVerbosityLevels:
    """Test different verbosity levels."""

    def test_verbosity_0_no_output_succeeds(self, capsys):
        """Verbosity 0 should produce no log output."""
        config_logger(file_logging=False, verbosity_level=0)
        log = logging.getLogger("test_module")

        log.debug("debug message")
        log.info("info message")
        log.warning("warning message")
        log.error("error message")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_verbosity_1_warning_level_succeeds(self, capsys):
        """Verbosity 1 should show WARNING and above."""
        config_logger(file_logging=False, verbosity_level=1)
        log = logging.getLogger("test_module")

        log.debug("debug message")
        log.info("info message")
        log.warning("warning message")
        log.error("error message")

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "debug message" not in output
        assert "info message" not in output
        assert "warning message" in output
        assert "error message" in output

    def test_verbosity_2_info_level_succeeds(self, capsys):
        """Verbosity 2 should show INFO and above."""
        config_logger(file_logging=False, verbosity_level=2)
        log = logging.getLogger("test_module")

        log.debug("debug message")
        log.info("info message")
        log.warning("warning message")
        log.error("error message")

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "debug message" not in output
        assert "info message" in output
        assert "warning message" in output
        assert "error message" in output

    def test_verbosity_3_debug_level_succeeds(self, capsys):
        """Verbosity 3 should show DEBUG and above."""
        config_logger(file_logging=False, verbosity_level=3)
        log = logging.getLogger("test_module")

        log.debug("debug message")
        log.info("info message")
        log.warning("warning message")
        log.error("error message")

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "debug message" in output
        assert "info message" in output
        assert "warning message" in output
        assert "error message" in output

    def test_invalid_verbosity_level_fails(self):
        """Invalid verbosity level should raise ValueError."""
        with pytest.raises(ValueError, match="verbosity_level must be between 0 and 3"):
            config_logger(file_logging=False, verbosity_level=5)


class TestConsoleVsFileLogging:
    """Test console vs file logging output formats."""

    def test_console_format_excludes_details_succeeds(self, capsys):
        """Console logging should use simple format."""
        config_logger(file_logging=False, verbosity_level=2)
        log = logging.getLogger("test_module")

        log.info("test message")

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "test message" in output
        assert "[test_module]" in output
        # Should NOT contain timestamp, line numbers, function names
        assert ":" not in output.split("[test_module]")[0]  # no timestamp before module

    def test_file_format_includes_details_succeeds(self, logfile_path):
        """File logging should include detailed format."""
        config_logger(file_logging=True, verbosity_level=2)
        log = logging.getLogger("test_module")

        log.info("test message")

        assert logfile_path.exists()
        content = logfile_path.read_text()

        assert "test message" in content
        assert "test_module" in content
        assert "test_file_format_includes_details" in content  # function name
        # Should contain timestamp
        assert any(c.isdigit() for c in content.split("INFO")[0])

    def test_file_logging_creates_file_succeeds(self, logfile_path):
        """File logging should create log file."""
        assert not logfile_path.exists()

        config_logger(file_logging=True, verbosity_level=1)
        log = logging.getLogger("test")
        log.warning("test")

        assert logfile_path.exists()

    def test_file_logging_appends_succeeds(self, logfile_path):
        """File logging should append to existing file."""
        config_logger(file_logging=True, verbosity_level=1)
        log = logging.getLogger("test")

        log.warning("first message")
        log.warning("second message")

        content = logfile_path.read_text()
        assert "first message" in content
        assert "second message" in content


class TestExceptionHandling:
    """Test exception handling in script vs library mode."""

    def test_main_catches_all_exceptions_succeeds(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """main() should catch all exceptions and return 1."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path)],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = RuntimeError("Simulated error")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Something went wrong" in captured.out
        assert "Simulated error" not in output

    def test_main_exception_logged_with_verbosity_succeeds(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """Exceptions in main() should be logged when verbosity enabled."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path), "-v"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = RuntimeError("Simulated error")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "Simulated error" in output or "Exception received" in output

    def test_main_exception_logged_to_file_succeeds(
        self,
        outfile_path,
        logfile_path,
        capsys,
        minimal_data_path,
    ):
        """Exceptions in main() should be logged to file when --log used."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path), "--log"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = RuntimeError("Simulated error")

                result = main()

        captured = capsys.readouterr()
        assert "Something went wrong" in captured.out
        assert result == 1
        assert logfile_path.exists()
        content = logfile_path.read_text()
        assert "Simulated error" in content
        assert "Exception received" in content

    def test_library_mode_raises_exceptions_fails(self, minimal_data_path):
        """Direct function calls should raise exceptions normally."""

        with pytest.raises(Exception):
            # Invalid keep_category should raise
            read_labnirs(minimal_data_path, keep_category="invalid")

    def test_argument_error_before_logger_setup_fails(self, capsys):
        """ArgumentError during parsing should print to stdout without logging."""
        with patch.object(sys, "argv", ["prog", "", ""]):
            with patch("labnirs2snirf.args.Arguments.parse") as mock_parse:
                mock_parse.side_effect = ArgumentError("Invalid argument value")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        # Should use print(), not logging
        assert "Invalid argument value" in captured.out
        assert "Something went wrong" not in captured.out
        # No traceback or exception details
        assert "Traceback" not in captured.out

    def test_labnirs2snirf_error_with_logger_fails(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """Labnirs2SnirfError with logger setup should be logged."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path), "-v"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = Labnirs2SnirfError("Test conversion failure")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "[labnirs2snirf" in output
        assert "Conversion failed: Test conversion failure" in output
        assert "Something went wrong" not in output
        assert "Traceback" not in captured.out

    def test_labnirs2snirf_error_without_logger_fails(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """Labnirs2SnirfError without logger should print to stdout."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path)],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = Labnirs2SnirfError("Test conversion failure")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "[labnirs2snirf" not in output
        assert "Conversion failed: Test conversion failure" in output
        assert "Something went wrong" not in output
        assert "Traceback" not in captured.out

    def test_unknown_exception_with_logger_fails(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """Unknown exception with logger should log full details."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path), "-v"],
        ):
            with patch("labnirs2snirf.labnirs2snirf.read_labnirs") as mock_read:
                mock_read.side_effect = ValueError("Unexpected value error")

                result = main()

        assert result == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        # Should log exception with details
        assert "Something went wrong" in output
        assert "Exception received" in output and "Unexpected value error" in output
        assert "Logging not configured" not in output

    def test_unknown_exception_without_logger_fails(
        self,
        capsys,
        outfile_path,
        minimal_data_path,
    ):
        """Unknown exception without logger should dump to stderr."""
        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(outfile_path)],
        ):
            with patch("labnirs2snirf.args.Arguments.parse") as mock_parse:
                mock_parse.side_effect = ValueError("Unexpected value error")

                with pytest.raises(ValueError, match="Unexpected value error"):
                    main()

        captured = capsys.readouterr()
        assert "Logging not configured" in captured.out


class TestModuleLevelLogging:
    """Test logging from different modules."""

    def test_module_loggers_inherit_config_succeeds(self, capsys):
        """Module-specific loggers should inherit root config."""
        config_logger(file_logging=False, verbosity_level=2)

        log1 = logging.getLogger("labnirs2snirf.snirf")
        log2 = logging.getLogger("labnirs2snirf.labnirs")

        log1.info("message from snirf")
        log2.info("message from labnirs")

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "message from snirf" in output
        assert "message from labnirs" in output
        assert "[labnirs2snirf.snirf]" in output
        assert "[labnirs2snirf.labnirs]" in output

    def test_different_modules_show_correctly_succeeds(self, logfile_path):
        """Different modules should be identifiable in file logs."""
        config_logger(file_logging=True, verbosity_level=2)

        log1 = logging.getLogger("labnirs2snirf.snirf")
        log2 = logging.getLogger("labnirs2snirf.layout")

        log1.info("from snirf module")
        log2.info("from layout module")

        content = logfile_path.read_text(encoding="utf-8")

        assert "labnirs2snirf.snirf" in content
        assert "labnirs2snirf.layout" in content
        assert "from snirf module" in content
        assert "from layout module" in content


class TestEndToEndLogging:
    """Test logging through complete conversion process."""

    def test_successful_conversion_logs_succeeds(
        self,
        tmp_path,
        capsys,
        minimal_data_path,
    ):
        """Successful conversion should produce appropriate log messages."""
        target = tmp_path / "out.snirf"

        with patch.object(
            sys,
            "argv",
            ["prog", str(minimal_data_path), str(target), "-vv"],
        ):
            result = main()

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should see various log messages from the process
        assert "Logger configured" in output
        assert "Reading labNIRS data" in output
        assert "Writing SNIRF file" in output
        assert "Successfully completed conversion" in output
        assert result == 0

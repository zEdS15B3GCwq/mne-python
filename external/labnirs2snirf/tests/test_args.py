"""
Unit tests for labnirs2snirf.args module.
"""

from pathlib import Path

import pytest

from labnirs2snirf.args import (
    ArgumentError,
    Arguments,
    _file_must_exist,
    _file_must_not_exist,
    _validate_drop_value,
)


@pytest.fixture(name="existing_file")
def fixture_existing_file(tmp_path):
    """Create a single existing file that can be used across all tests."""
    test_file = tmp_path / "existing_file.txt"
    test_file.write_text("test content")
    return test_file


@pytest.fixture(name="nonexistent_file")
def fixture_nonexistent_file(tmp_path):
    """Create a single non-existent file path."""
    return tmp_path / "does_not_exist.txt"


class TestFileMustExist:
    """Tests for file_must_exist function."""

    def test_existing_file_returns_path_succeeds(self, existing_file):
        """Test that existing file returns Path object."""
        result = _file_must_exist(str(existing_file))

        assert isinstance(result, Path)
        assert result == existing_file
        assert result.is_file()

    def test_nonexistent_file_raises_error(self, nonexistent_file):
        """Test that non-existent file raises ArgumentError."""

        with pytest.raises(ArgumentError) as exc_info:
            _file_must_exist(str(nonexistent_file))

        assert "does not exist" in str(exc_info.value)
        assert str(nonexistent_file) in str(exc_info.value)

    def test_directory_raises_error(self, tmp_path):
        """Test that directory path raises ArgumentError."""
        directory = tmp_path / "test_dir"
        directory.mkdir()

        with pytest.raises(ArgumentError) as exc_info:
            _file_must_exist(str(directory))

        assert "does not exist" in str(exc_info.value)

    def test_empty_string_raises_error(self):
        """Test that empty string raises ArgumentError."""
        with pytest.raises(ArgumentError):
            _file_must_exist("")

    def test_relative_path_existing_file_succeeds(self, tmp_path, monkeypatch):
        """Test relative path to existing file in subdirectory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(tmp_path)
        test_file = subdir / "relative.txt"
        test_file.write_text("content")

        result = _file_must_exist("subdir/relative.txt")

        assert result.is_file()
        assert not result.is_absolute()

    def test_absolute_path_existing_file_succeeds(self, existing_file):
        """Test absolute path to existing file."""
        result = _file_must_exist(str(existing_file.absolute()))

        assert result.is_absolute()
        assert result.is_file()

    def test_symlink_to_existing_file_succeeds(self, existing_file, tmp_path):
        """Test symlink pointing to existing file."""
        symlink = tmp_path / "link.txt"

        try:
            symlink.symlink_to(existing_file)
            result = _file_must_exist(str(symlink))
            assert result.is_file()
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

    def test_file_with_unicode_name_succeeds(self, tmp_path):
        """Test file with unicode characters in name."""
        test_file = tmp_path / "tęšt_fïlę.txt"
        test_file.write_text("content")

        result = _file_must_exist(str(test_file))

        assert result.is_file()

    def test_empty_path_raises_error(self):
        """Test that empty path raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            _file_must_exist("")

        assert "must not be empty" in str(exc_info.value)

    def test_special_characters_in_filename_succeeds(self, tmp_path):
        """Test file with special characters in name."""
        test_file = tmp_path / "test file (1) [copy].txt"
        test_file.write_text("content")

        result = _file_must_exist(str(test_file))

        assert result.is_file()


class TestFileMustNotExist:
    """Tests for file_must_not_exist function."""

    def test_nonexistent_file_returns_path_succeeds(self, nonexistent_file):
        """Test that non-existent file returns Path object."""
        result = _file_must_not_exist(str(nonexistent_file))

        assert isinstance(result, Path)
        assert result == nonexistent_file
        assert not result.exists()

    def test_existing_file_raises_error(self, existing_file):
        """Test that existing file raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            _file_must_not_exist(str(existing_file))

        assert "already exists" in str(exc_info.value)
        assert str(existing_file) in str(exc_info.value)

    def test_nonexistent_file_in_nonexistent_directory_succeeds(self, tmp_path):
        """Test non-existent file in non-existent directory."""
        test_file = tmp_path / "nonexistent_dir" / "file.txt"

        result = _file_must_not_exist(str(test_file))

        assert isinstance(result, Path)
        assert not result.exists()

    def test_relative_path_nonexistent_file_succeeds(self, tmp_path, monkeypatch):
        """Test relative path to non-existent file in subdirectory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        monkeypatch.chdir(tmp_path)
        result = _file_must_not_exist("subdir/new_relative.txt")

        assert not result.exists()
        assert not result.is_absolute()

    def test_absolute_path_nonexistent_file_succeeds(self, nonexistent_file):
        """Test absolute path to non-existent file."""
        result = _file_must_not_exist(str(nonexistent_file.absolute()))

        assert not result.exists()

    def test_symlink_to_existing_file_raises_error(self, existing_file, tmp_path):
        """Test symlink pointing to existing file raises error."""
        symlink = tmp_path / "link.txt"

        try:
            symlink.symlink_to(existing_file)
            with pytest.raises(ArgumentError):
                _file_must_not_exist(str(symlink))
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

    def test_empty_path_raises_error(self):
        """Test that empty path raises ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            _file_must_not_exist("")

        assert "must not be empty" in str(exc_info.value)


class TestValidateDropValue:
    """Tests for _validate_drop_value function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("hbt", "hbt"),
            ("HbT", "hbt"),
            ("HBT", "hbt"),
            ("hbo", "hbo"),
            ("HbO", "hbo"),
            ("HBO", "hbo"),
            ("hbr", "hbr"),
            ("HbR", "hbr"),
            ("HBR", "hbr"),
        ],
    )
    def test_valid_hemoglobin_types_succeeds(self, input_value, expected):
        """Test all valid hemoglobin types in various cases."""
        result = _validate_drop_value(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("7", "7"),
            ("780", "780"),
            ("850", "850"),
            ("999999", "999999"),
        ],
    )
    def test_valid_wavelength_integers_succeeds(self, input_value, expected):
        """Test valid wavelength integers."""
        result = _validate_drop_value(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("  hbt  ", "hbt"),
            ("  HbO  ", "hbo"),
            ("  780  ", "780"),
            ("\thbr\t", "hbr"),
        ],
    )
    def test_whitespace_handling_succeeds(self, input_value, expected):
        """Test that whitespace is properly trimmed."""
        result = _validate_drop_value(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_value",
        [
            "-780",
            "780.5",
            "invalid",
            "",
            "   ",
            "hb",
            "hbx",
            "780nm",
            "@#$%",
            "1e3",
            "0x10",
            "HbT123",
            "123HbO",
        ],
    )
    def test_invalid_values_raise_error(self, invalid_value):
        """Test that invalid values raise ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            _validate_drop_value(invalid_value)

        assert "Invalid drop type" in str(exc_info.value)

    def test_error_message_contains_valid_options_succeeds(self):
        """Test error message contains valid options."""
        with pytest.raises(ArgumentError) as exc_info:
            _validate_drop_value("invalid")

        error_msg = str(exc_info.value)
        assert "HbT" in error_msg
        assert "HbO" in error_msg
        assert "HbR" in error_msg
        assert "integer" in error_msg or "wavelength" in error_msg


class TestArgumentsParse:
    """Tests for Arguments.parse() method."""

    def test_parse_minimal_arguments_succeeds(
        self,
        existing_file,
        tmp_path,
        monkeypatch,
    ):
        """Test parsing with only required source file argument."""
        monkeypatch.chdir(tmp_path)  # Makes sure there's no existing output.snirf
        args = Arguments()
        result = args.parse([str(existing_file)])

        assert result.source_file == existing_file
        assert result.target_file == Path("out.snirf")
        assert result.type == "all"
        assert result.verbosity == 0
        assert result.log is False
        assert result.locations is None
        assert result.drop is None

    def test_parse_with_target_file_succeeds(self, existing_file, nonexistent_file):
        """Test parsing with both source and target files."""

        args = Arguments()
        result = args.parse([str(existing_file), str(nonexistent_file)])

        assert result.source_file == existing_file
        assert result.target_file == nonexistent_file

    def test_parse_with_locations_file_succeeds(
        self,
        existing_file,
        tmp_path,
        monkeypatch,
    ):
        """Test parsing with locations file."""
        locations = tmp_path / "locations.sfp"
        locations.write_text("S1\t0\t0\t0", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--locations", str(locations)])

        assert result.locations == locations

    @pytest.mark.parametrize("type_value", ["hb", "raw", "all", "HB", "RAW", "ALL"])
    def test_parse_with_type_argument_succeeds(
        self,
        existing_file,
        type_value,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with --type argument (case insensitive)."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--type", type_value])

        assert result.type == type_value.lower()

    def test_parse_with_single_drop_value_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with single --drop argument."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--drop", "hbt"])

        assert result.drop == {"hbt"}

    def test_parse_with_multiple_drop_values_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with multiple --drop arguments."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse(
            [str(existing_file), "--drop", "hbt", "--drop", "780", "--drop", "HbO"],
        )

        assert result.drop == {"hbt", "780", "hbo"}

    def test_parse_drop_removes_duplicates_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test that duplicate drop values are removed."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse(
            [str(existing_file), "--drop", "hbt", "--drop", "HbT", "--drop", "hbt"],
        )

        assert result.drop is not None
        assert result.drop == {"hbt"}
        assert len(result.drop) == 1

    def test_parse_with_verbose_flag_once_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with single -v flag."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "-v"])

        assert result.verbosity == 1

    def test_parse_with_verbose_flag_multiple_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with multiple -v flags."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "-vvv"])

        assert result.verbosity == 3

    def test_parse_with_verbose_long_form_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with --verbose flag."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--verbose", "--verbose"])

        assert result.verbosity == 2

    def test_parse_with_log_flag_succeeds(self, existing_file, monkeypatch, tmp_path):
        """Test parsing with --log flag."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--log"])

        assert result.log is True
        assert result.verbosity == 1  # --log implies at least -v

    def test_parse_log_with_explicit_verbosity_succeeds(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test parsing with --log flag and explicit verbosity."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file), "--log", "-vvv"])

        assert result.log is True
        assert result.verbosity == 3

    def test_parse_all_optional_arguments_succeeds(self, existing_file, tmp_path):
        """Test parsing with all optional arguments specified."""
        target = tmp_path / "output.snirf"
        locations = tmp_path / "locations.sfp"
        locations.write_text("S1\t0\t0\t0")

        args = Arguments()
        result = args.parse(
            [
                str(existing_file),
                str(target),
                "--locations",
                str(locations),
                "--type",
                "hb",
                "--drop",
                "hbt",
                "--drop",
                "780",
                "-vv",
                "--log",
            ],
        )

        assert result.source_file == existing_file
        assert result.target_file == target
        assert result.locations == locations
        assert result.type == "hb"
        assert result.drop == {"hbt", "780"}
        assert result.verbosity == 2
        assert result.log is True

    def test_parse_nonexistent_source_raises_error(self, tmp_path, monkeypatch):
        """Test that non-existent source file raises error."""
        monkeypatch.chdir(tmp_path)
        source = "nonexistent.txt"

        args = Arguments()
        with pytest.raises(ArgumentError):  # argparse exits on error
            args.parse([str(source)])

    def test_parse_existing_target_raises_error(self, existing_file):
        """Test that existing target file raises error."""
        args = Arguments()
        with pytest.raises(ArgumentError):
            args.parse([str(existing_file), str(existing_file)])

    def test_parse_invalid_type_value_raises_error(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
        capsys,
    ):
        """Test that invalid --type value raises error."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        with pytest.raises(SystemExit):
            args.parse([str(existing_file), "--type", "invalid"])
        capsys.readouterr()

    def test_parse_invalid_drop_value_raises_error(
        self,
        existing_file,
        monkeypatch,
        tmp_path,
    ):
        """Test that invalid --drop value raises error."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        with pytest.raises(ArgumentError):
            args.parse([str(existing_file), "--drop", "invalid_value"])

    def test_parse_no_arguments_shows_help_succeeds(self, capsys):
        """Test that parsing with no arguments shows help and exits."""
        args = Arguments()
        with pytest.raises(SystemExit) as exc_info:
            args.parse([])
        captured = capsys.readouterr()

        assert "usage:" in captured.out
        # Help exits with code 0
        assert exc_info.value.code == 0  # ty: ignore[unresolved-attribute]

    def test_parse_help_flag_exits_succeeds(self, capsys):
        """Test that -h/--help flag exits successfully."""
        args = Arguments()
        with pytest.raises(SystemExit) as exc_info:
            args.parse(["-h"])

        captured = capsys.readouterr()
        assert "usage:" in captured.out
        # Help exits with code 0
        assert exc_info.value.code == 0  # ty: ignore[unresolved-attribute]

    def test_parse_returns_self_succeeds(self, existing_file, tmp_path, monkeypatch):
        """Test that parse() returns self."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        result = args.parse([str(existing_file)])

        assert result is args

    def test_parse_removes_parser_attribute_succeeds(
        self,
        existing_file,
        tmp_path,
        monkeypatch,
    ):
        """Test that parse() removes the parser attribute."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        assert hasattr(args, "parser")
        args.parse([str(existing_file)])
        assert not hasattr(args, "parser")

    def test_parse_with_relative_paths_succeeds(self, tmp_path, monkeypatch):
        """Test parsing with relative paths."""
        monkeypatch.chdir(tmp_path)
        source = tmp_path / "source.txt"
        source.write_text("data")

        args = Arguments()
        result = args.parse(["source.txt"])

        assert result.source_file.name == "source.txt"

    def test_parse_with_absolute_paths_succeeds(self, existing_file, tmp_path):
        """Test parsing with absolute paths."""
        target = tmp_path / "output.snirf"

        args = Arguments()
        result = args.parse([str(existing_file.absolute()), str(target.absolute())])

        assert result.source_file.is_absolute()
        assert result.target_file.is_absolute()

    def test_str_representation_succeeds(self, existing_file, tmp_path, monkeypatch):
        """Test string representation of Arguments."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        args.parse([str(existing_file)])

        str_repr = str(args)
        assert "Arguments" in str_repr
        assert "source_file" in str_repr

    def test_repr_representation_succeeds(self, existing_file, tmp_path, monkeypatch):
        """Test repr representation of Arguments."""
        monkeypatch.chdir(tmp_path)
        args = Arguments()
        args.parse([str(existing_file)])

        repr_str = repr(args)
        assert "Arguments" in repr_str

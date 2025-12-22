"""
Shared fixtures for tests.
"""

import pathlib

import pytest


@pytest.fixture(scope="session", name="test_data_dir")
def fixture_test_data_dir():
    return pathlib.Path(__file__).parent.parent / "data" / "test"


@pytest.fixture(scope="session", name="minimal_data_path")
def fixture_minimal_data(test_data_dir):
    return test_data_dir / "minimal_labnirs.txt"


@pytest.fixture(scope="session", name="hbonly_data_path")
def fixture_hbonly_data(test_data_dir):
    return test_data_dir / "hb_only.txt"


@pytest.fixture(scope="session", name="rawonly_data_path")
def fixture_rawonly_data(test_data_dir):
    return test_data_dir / "raw_only.txt"


@pytest.fixture(scope="session", name="small_data_path")
def fixture_small_data(test_data_dir):
    return test_data_dir / "small_labnirs.txt"

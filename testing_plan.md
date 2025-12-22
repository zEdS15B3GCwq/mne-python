## Background:

MNE-Python is a neuroimaging data analysis toolkit in Python, that can be used
with MRI, EEG, MEG, and NIRS data, and perhaps more. In this particular case, we
are focusing on NIRS-related functionality. MNE has been designed to read NIRS
data with 2 wavelengths per channel (i.e. one source-detector pair uses 2
wavelengths to measure the absorption of light at that particular wavelength in
the brain). File IO and preprocessing steps were only able to, or were
restricted to process 2 wavelengths. A new PR is changing that, expanding
support to multiple wavelengths. IO, preprocessing routines and test files need
to be changed. At the moment, the first 2 steps are done, now we're working on
adding tests.

## Environment

- This folder contains the mne-python project from github, managed by git. The
  currently active branch 'testing' is where the tests need to be added.
- Python version, package and virtual environment management is handled by 'uv'.
- The local virtual environment (in .venv) has all necessary packages installed;
  it uses Python 3.13.11. Don't start installing packages without asking.
- 'git' and 'uv' are on the path
- To run python commands, use `uv run` or activate the virtual environment in
  '.venv'. Otherwise, commands will not work. Do not just invoke python.exe in
  the .venv folder - that won't work either.

## PR changes

These are the files with changes relevant to the task at hand:

- mne\preprocessing\nirs_beer_lambert_law.py - was updated to handle >=2
  wavelengths
- mne\reprocessing\nirs_scalp_coupling_index.py - scalp coupling index
  calculation now uses the minimum correlation in a group
- mne\preprocessing\nirs\nirs.py - validation checks were based on 2
  wavelengths, changed to handle >=2 (I'm running this on Windows, so paths use
  '\'. It might be necessary to change this to '/' in some cases.)

What exactly changed, can be checked with `git diff origin/main -- <filename>`

## Goal

We need to test the following:

- reading multi-wavelength .snirf files
- preprocessing multi-wavelength NIRS data: beer-lambert law conversion and
  scalp coupling index calculation, plus data validation steps

However, instead of generating a comprehensive suite of tests, we need to be
conservative and aim at testing only essential parts that are unique to
multi-wavelength data. This repo is already quite large and we need to avoid
bloating it further. Existing tests need to be re-used as much as possible with
minimal changes, adapted to be used with multi-wavelength data.

For file import testing, the plan is to use a pytest fixture that generates a
small multi-wavelength snirf file (more on the generation process later).
Another fixture should be used to generate a simple and small multi-wavelength
nirs dataset with raw wavelength data for preprocessing and validation tests.
These should be shared by as many new tests as possible to avoid duplication.

## Plan

This is a starting point for an implementation plan:

- "mne\io\snirf\tests\test_snirf.py": This file contains tests for SNIRF file
  import. Tests that should not be touched are ones that specifically use one
  input file and have assertations with values specific to that file. Other
  tests that are already parametrised to use multiple input files are good
  targets, those should also run with multi-wavelength data (from fixture). We
  also need a specific 'test_snirf_against_multiple_wavelengths' or similarly
  named test that checks if multiple wavelengths were imported correctly
  (channel names for example - see what other tests validate).
- Tests for changes in "mne\preprocessing\nirs\nirs.py" are in
  "mne\preprocessing\nirs\tests\test_nirs.py": here, two types of tests should
  be expanded. One type (e.g. "test_fnirs_channel_naming_and_order_readers") is
  parametrised to read multiple datasets (fname), so here a fixture with snirf
  data should be used (ideally same as in previous point). However, some
  validation checks here use fixed values that work with the existing data files
  but will not work with multi-wavelength ones - these need to be amended. The
  tests should check for number of wavelengths and if >2, use values specific to
  the fixture (e.g. channel names, wavelengths). The other type uses manually
  created `Raw` objects, often creating multiple in the same test. The `Raw`
  object creation should be removed into a fixture, then all subsequent
  variations should just modify the `Raw` object from the base fixture. After
  this refactoring is done, a new fixture with our multi-wavelength data should
  be created and used by these types of tests. Here too, some fixed values will
  need to be updated, or different evaluation paths should be used based on
  wavelength.
- Tests for changes in "\mne\preprocessing\nirs_scalp_coupling_index.py" are in
  "mne\preprocessing\nirs_scalp_coupling_index.py", and those for changes in
  "mne\preprocessing\nirs\tests\test_beer_lambert_law.py" are in
  "\mne\preprocessing\nirs_beer_lambert_law.py". Tests here should follow the
  same minimal adaptation guidelines as above.

In general, the guidelines are as above: 1) use existing tests instead of
creating new ones (except in test_snirf.py create one test specific to
multi-wavelength snirf). 2) In case of "parametrised" tests, add an additionsal
dataset generated in fixtures; or use fixtures to generate data in test. 3) Be
careful with fixed evaluation criteria, as those may not work with multi-
wavelength data. Adapt tests to check number of wavelengths if necessary, and
make evaluation data depend on that (e.g. how many channels to mark as bad, or
how many wavelengths to validate). Use values from the multi-wavelength
generator fixture where appropriate.

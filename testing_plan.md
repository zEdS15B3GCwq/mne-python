## Background:
MNE-Python is a neuroimaging data analysis toolkit in Python, that can be used with MRI, EEG, MEG, and NIRS data, and perhaps more. In this particular case, we are focusing on NIRS-related functionality. MNE has been designed to read NIRS data with 2 wavelengths per channel (i.e. one source-detector pair uses 2 wavelengths to measure the absorption of light at that particular wavelength in the brain). File IO and preprocessing steps were only able to, or were restricted to process 2 wavelengths. A new PR is changing that, expanding support to multiple wavelengths. IO, preprocessing routines and test files need to be changed. At the moment, the first 2 steps are done, now we're working on adding tests.

## Environment
- This folder contains the mne-python project from github, managed by git. The currently active branch 'nirs_more_wavelengths' is the PR in which the tests need to be added.
- 'uv' is used for Python version, package and virtual environment management
- The local virtual environment (in .venv) has all necessary packages installed; it uses Python 3.13.11
- 'git' and 'uv' are on the path
- To run python commands, use 'uv run' or activate the virtual environment in '.venv'. Otherwise, commands will not work. Do not just invoke python.exe in the .venv folder - that won't work either.

## PR changes
These are the files with changes relevant to the task at hand:
- mne\preprocessing\nirs\_beer_lambert_law.py - was updated to handle >=2 wavelengths
- mne\reprocessing\nirs\_scalp_coupling_index.py - scalp coupling index calculation now uses the minimum correlation in a group
- mne\preprocessing\nirs\nirs.py - validation checks were based on 2 wavelengths, changed to handle >=2
(I'm running this on Windows, so paths use \. It might be necessary to change this to / in some cases.)

What exactly changed can be checked with 'git diff origin/main -- <filename>'

## Goal
We need to test the following:
- reading multi-wavelength .snirf files
- preprocessing multi-wavelength NIRS data: beer-lambert law conversion and scalp coupling index calculation, plus data validation steps

However, instead of generating a comprehensive suite of tests, we need to be conservative and aim at testing only essential parts that are unique to multi-wavelength data. This repo is already quite large and we need to avoid bloating it further. Existing tests need to be re-used as much as possible with minimal changes, adapted to be used with multi-wavelength data.

My idea is to use a shared pytest fixture that generates a small multi-wavelength snirf file (more on the generation process later), which can be than fed to some of the tests that use snirf data, and perhaps another fixture shared by tests that use nirs data (in mne's own internal format).

We need to add tests to the following test modules:
- mne-python\mne\io\snirf\tests\test_snirf.py
  This file contains tests for SNIRF file import. Many of the tests use data specific to a dataset or generate data to test specific points. We don't need to adapt these tests. Generic tests that can run on multiple files without modification can be parametrised to also run with the multi-wavelength .snirf data (fixture above). We may need a specific 'test_snirf_against_multiple_wavelengths' or similarly named test that checks if multiple wavelengths were imported correctly (channel names for example - see what other tests validate).

- 
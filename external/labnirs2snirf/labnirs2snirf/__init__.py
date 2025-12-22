"""
Convert LabNIRS data files to SNIRF format.

This package provides tools to convert fNIRS data exported from LabNIRS software
to the Shared Near Infrared Spectroscopy Format (SNIRF), enabling compatibility
with analysis tools like MNE-Python and other NIRS analysis software.

Main modules:

- labnirs: Reading and parsing LabNIRS data files
- snirf: Writing data to SNIRF/HDF5 format
- layout: Importing and applying probe position data
- model: Data models following the SNIRF specification
- labnirs2snirf: Entrypoint when run as a script
- args: Command-line argument parsing
- log: Logging configuration for command-line usage
"""

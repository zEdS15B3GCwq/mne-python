"""
Entry point for running labnirs2snirf as a module.

This module provides the main entry point when the package is executed
with `python -m labnirs2snirf`.
"""

import sys

# Entry point for when the module is run as a script
if __name__ == "__main__":
    from .labnirs2snirf import main

    sys.exit(main())

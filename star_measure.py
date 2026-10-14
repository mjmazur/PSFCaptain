#!/usr/bin/env python
"""Backwards-compatible entry point.

The implementation now lives in the `psfcaptain` package; this shim keeps
`python star_measure.py ...` working. `python -m psfcaptain ...` is equivalent.
"""

from psfcaptain.cli import main

if __name__ == "__main__":
    main()

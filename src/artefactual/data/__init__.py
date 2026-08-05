"""Package anchor for the shipped calibration and weight files.

`artefactual.utils.io` reads `weights_*.json` / `calibration_*.json` from here via
`importlib.resources.files("artefactual.data")`, so this package must remain importable
even though it exposes no Python API.
"""

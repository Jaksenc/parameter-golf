"""Serialization-only adapter for the frozen v13 analyzer.

The original analyzer remains byte-for-byte unchanged. NumPy scalar values are
converted only when writing JSON and returning the in-memory report. No
inference, calibration, metric, filtering or arm-selection rule is changed.
"""
from __future__ import annotations
import argparse
import numpy as np
import analyze_probability as frozen

def json_primitives(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_primitives(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_primitives(item) for item in value]
    return value

def run(mode, root):
    if mode not in ("fit", "score"):
        raise ValueError("Expected fit or score")
    original_write = frozen.exp.write
    def write(path, value):
        return original_write(path, json_primitives(value))
    frozen.exp.write = write
    try:
        result = frozen.fit(root) if mode == "fit" else frozen.score(root)
        return json_primitives(result)
    finally:
        frozen.exp.write = original_write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    run(args.mode, args.root)

if __name__ == "__main__":
    main()

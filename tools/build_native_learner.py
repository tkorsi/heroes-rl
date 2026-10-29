#!/usr/bin/env python3
"""Build a fresh, verified C++ learner extension without starting any training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh external directory for native binary, build log and manifest")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    from experiments.learner.native_backend import build_backend
    print(json.dumps(build_backend(args.output_dir, jobs=args.jobs, verbose=args.verbose), indent=2))


if __name__ == "__main__":
    main()

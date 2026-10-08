#!/usr/bin/env python
"""Preprocess the local EEGMMIDB recordings and cache the 5 s frames.

Usage:
    python scripts/prepare_data.py [--data-dir DIR] [--channels C1 C2 ...] \
        [--subjects S001 S002 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as data_mod                      # noqa: E402
from src.config import ExperimentConfig, PAPER_CHANNELS  # noqa: E402


def default_data_dir() -> Path:
    local = Path(__file__).resolve().parents[1] / "data" / "eegmmidb"
    if local.exists():
        return local
    # fall back to the EEGMMIDB cache already present in the sibling project
    sibling = (Path.home() / "Desktop" / "Researches" / "EEGLearning" / "data"
               / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0")
    return sibling


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--channels", nargs="+", default=None)
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig(data_dir=args.data_dir)
    if args.channels:
        cfg.channels = args.channels
    if args.subjects:
        cfg.subjects = args.subjects
    cfg.subjects = cfg.subjects or data_mod.list_subjects(cfg.data_dir)

    print(f"Data dir : {cfg.data_dir}")
    print(f"Subjects : {cfg.subjects}")
    print(f"Channels : {cfg.channels}")
    path = data_mod.build_cache(cfg)
    print(f"\nCache written to {path}")


if __name__ == "__main__":
    main()

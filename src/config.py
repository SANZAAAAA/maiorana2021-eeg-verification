"""Configuration for the reproduction of

    E. Maiorana, "Learning deep features for task-independent EEG-based
    biometric verification", Pattern Recognition Letters 143 (2021) 122-129.

All paper-specific hyper-parameters are collected here so that the exact
settings reported in the article can be reproduced from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# EEG channels
# --------------------------------------------------------------------------- #
# The 19 electrodes of the 10-20 international system used in the paper
# (Fig. 2 / Table 5).  T3/T4/T5/T6 of the 10-20 system are named
# T7/T8/P7/P8 in the 10-10 system adopted by the EEGMMIDB substitute dataset.
PAPER_CHANNELS = [
    "F1", "F2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz",
    "T7", "T8", "P7", "P8",
    "P3", "P4", "Pz",
    "O1", "O2",
]

# Mapping from the paper channel labels to the EEGMMIDB (EDF) channel names.
EEGMMIDB_CHANNEL_MAP = {
    "F1": "F1..", "F2": "F2..", "F3": "F3..", "F4": "F4..",
    "F7": "F7..", "F8": "F8..", "Fz": "Fz..",
    "C3": "C3..", "C4": "C4..", "Cz": "Cz..",
    "T7": "T7..", "T8": "T8..", "P7": "P7..", "P8": "P8..",
    "P3": "P3..", "P4": "P4..", "Pz": "Pz..",
    "O1": "O1..", "O2": "O2..",
}

# --------------------------------------------------------------------------- #
# Preprocessing (Section 3.1)
# --------------------------------------------------------------------------- #
BAND = (8.0, 30.0)          # alpha-beta sub-band retained by the paper
TARGET_FS = 64.0            # downsampling to S = 64 Hz
FRAME_SECONDS = 5.0         # H = 5 s frames
FRAME_OVERLAP = 0.8         # 80 % overlap between consecutive frames


@dataclass
class TrainConfig:
    """Hyper-parameters of the siamese training (Section 4.1)."""

    margin: float = 1.0             # contrastive-loss margin d
    batch_size: int = 128
    lr: float = 1e-3
    momentum: float = 0.9           # SGDM
    weight_decay: float = 5e-3
    epochs: int = 40
    dropout: float = 0.5            # L13 dropout
    embedding_dim: int = 256        # L14 output size
    pairs_per_subject: int = 200    # genuine pairs sampled per subject/epoch
    impostors_per_genuine: int = 2  # paper: two impostor pairs per genuine pair
    seed: int = 2021


@dataclass
class ExperimentConfig:
    """Full experimental configuration."""

    data_dir: Path = PROJECT_ROOT / "data" / "eegmmidb"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    results_dir: Path = PROJECT_ROOT / "results"

    subjects: list[str] = field(default_factory=list)   # empty -> all found
    channels: list[str] = field(default_factory=lambda: list(PAPER_CHANNELS))

    # pseudo-sessions derived from the EEGMMIDB motor-imagery runs 4/8/12
    sessions: list[str] = field(default_factory=lambda: ["R1", "R2", "R3"])
    session_runs: dict[str, int] = field(
        default_factory=lambda: {"R1": 4, "R2": 8, "R3": 12}
    )
    tasks: list[str] = field(default_factory=lambda: ["T1", "T2"])

    n_folds: int = 5                # subject-wise folds (train / test disjoint)
    train: TrainConfig = field(default_factory=TrainConfig)


# Convenience: index of each paper channel, kept in a stable order everywhere.
CHANNEL_INDEX = {c: i for i, c in enumerate(PAPER_CHANNELS)}

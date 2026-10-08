"""Dataset adapter that turns the locally available EEGMMIDB recordings into
the multi-session / multi-task structure used by Maiorana (2021).

Substitute-dataset rationale
----------------------------
The original study relies on a proprietary database (45 subjects, 5 sessions,
6 tasks, 19 electrodes) that is not publicly distributed.  Reproducing the
*method* therefore requires a stand-in multi-session, multi-task EEG corpus.
The PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) is used here:

    * three repeated motor-imagery recordings (runs 4, 8, 12) act as the three
      pseudo-sessions  R1, R2, R3;
    * the two cued mental tasks (T1 = imagined left fist, T2 = imagined right
      fist) act as the two "protocols";
    * the 19 electrodes of the paper are selected out of the 64 EEGMMIDB
      channels (T3/T4/T5/T6 -> T7/T8/P7/P8 in the 10-10 nomenclature).

Everything downstream (preprocessing, siamese CNN, verification) is kept
identical to the paper; only the data source differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import preprocessing
from .config import EEGMMIDB_CHANNEL_MAP, PAPER_CHANNELS


# --------------------------------------------------------------------------- #
# Raw EDF loading
# --------------------------------------------------------------------------- #
def edf_path(data_dir: Path, subject: str, run: int) -> Path:
    """EEGMMIDB layout: <data_dir>/S001/S001R04.edf."""
    sid = subject if subject.startswith("S") else f"S{int(subject):03d}"
    return Path(data_dir) / sid / f"{sid}R{run:02d}.edf"


def list_subjects(data_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(data_dir).glob("S*") if p.is_dir())


def _task_segments(raw):
    """Return {task: [(start_sample, end_sample), ...]} from MNE annotations."""
    sfreq = raw.info["sfreq"]
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durs = np.asarray(raw.annotations.duration, dtype=float)
    descs = list(raw.annotations.description)
    total = raw.n_times / sfreq
    segments: dict[str, list[tuple[int, int]]] = {}
    for i, desc in enumerate(descs):
        dur = durs[i]
        if dur <= 0:
            nxt = onsets[i + 1] if i + 1 < len(onsets) else total
            dur = max(nxt - onsets[i], 0.0)
        start = int(round(onsets[i] * sfreq))
        end = int(round((onsets[i] + dur) * sfreq))
        segments.setdefault(desc, []).append((start, end))
    return segments


def load_run_task_streams(path: Path, channels_edf: list[str]) -> dict[str, np.ndarray]:
    """Load one EDF run and return {task: (C, T) float64 array}.

    Each task stream is the concatenation of all the annotated segments that
    belong to that task (the paper treats the recordings as continuous
    streams; concatenating the cued segments recovers a stream long enough to
    be framed at 5 s while keeping a clean task label).
    """
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    missing = [c for c in channels_edf if c not in raw.ch_names]
    if missing:
        raise KeyError(f"{path.name}: missing channels {missing}")
    raw.pick(channels_edf)
    data = raw.get_data()  # (C, T) at 160 Hz
    segments = _task_segments(raw)

    streams: dict[str, np.ndarray] = {}
    for task, segs in segments.items():
        parts = [data[:, s:e] for (s, e) in segs if e - s > 0]
        if parts:
            streams[task] = np.concatenate(parts, axis=1)
    return streams


# --------------------------------------------------------------------------- #
# Cache construction
# --------------------------------------------------------------------------- #
def build_cache(cfg) -> Path:
    """Preprocess every (subject, session, task) stream and cache the frames."""
    from .config import FRAME_OVERLAP, FRAME_SECONDS, TARGET_FS

    data_dir = Path(cfg.data_dir)
    subjects = cfg.subjects or list_subjects(data_dir)
    channels_edf = [EEGMMIDB_CHANNEL_MAP[c] for c in cfg.channels]

    frames: dict[str, np.ndarray] = {}
    for subj in subjects:
        for sess in cfg.sessions:
            run = cfg.session_runs[sess]
            path = edf_path(data_dir, subj, run)
            if not path.exists():
                print(f"  [skip] {path} not found")
                continue
            streams = load_run_task_streams(path, channels_edf)
            for task, stream in streams.items():
                if task not in cfg.tasks:
                    continue
                proc = preprocessing.preprocess_stream(
                    stream, fs_in=160.0, fs_out=TARGET_FS
                )
                fr = preprocessing.frame_stream(
                    proc, fs=TARGET_FS, win_s=FRAME_SECONDS, overlap=FRAME_OVERLAP
                )
                if fr.shape[0] == 0:
                    continue
                frames[f"{subj}|{sess}|{task}"] = fr.astype(np.float32)
                print(f"  {subj} {sess} {task}: {fr.shape[0]} frames "
                      f"({fr.shape[1]} ch x {fr.shape[2]} samples)")

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "frames.npz"
    np.savez_compressed(npz_path, **frames)
    index = {
        "channels": list(cfg.channels),
        "sessions": list(cfg.sessions),
        "tasks": list(cfg.tasks),
        "fs": TARGET_FS,
        "frame_seconds": FRAME_SECONDS,
        "overlap": FRAME_OVERLAP,
        "keys": sorted(frames.keys()),
    }
    (cache_dir / "index.json").write_text(json.dumps(index, indent=2))
    return npz_path


def load_cache(cache_dir: Path) -> dict:
    """Load the framed cache produced by :func:`build_cache`."""
    cache_dir = Path(cache_dir)
    npz = np.load(cache_dir / "frames.npz")
    index = json.loads((cache_dir / "index.json").read_text())
    frames = {k: npz[k] for k in npz.files}
    return {"frames": frames, "index": index}


def frame_key(subject: str, session: str, task: str) -> str:
    return f"{subject}|{session}|{task}"

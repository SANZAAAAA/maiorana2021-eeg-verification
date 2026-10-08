"""Verification stage and error-rate evaluation (Maiorana 2021, Section 3.1/4).

For every subject a *one-class SVM* is trained, for each electrode, on the
representations of the enrolment frames.  A verification probe produces C
per-channel scores that are fused into a single outcome; the equal error rate
(EER) is then computed over the genuine/impostor score distributions.

The same code supports:

    * within-task and cross-task probes;
    * single-session (SSE) and multiple-session (MSE) enrolment;
    * short/long time distance depending on the session pair selected.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.svm import OneClassSVM

from .train import embed_frames


def _as_encoder(obj, device: str = "cpu"):
    """Wrap a trained CNN into a plain ``(N, 320) -> (N, D)`` callable."""
    if callable(obj) and not isinstance(obj, torch.nn.Module):
        return obj
    return lambda arr: embed_frames(obj, arr, device=device)


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Return (EER, threshold).  Higher scores mean "more genuine"."""
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan"), float("nan")
    g = np.sort(genuine)
    imp = np.sort(impostor)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    # vectorised ROC via binary search (O(n log n) instead of O(n^2))
    frr = np.searchsorted(g, thresholds, side="left") / len(g)
    far = 1.0 - np.searchsorted(imp, thresholds, side="left") / len(imp)
    # EER is where FAR and FRR cross; interpolate on the FAR-FRR difference.
    diff = far - frr
    idx = np.where(diff <= 0)[0]
    if len(idx) == 0:                     # never crosses -> perfect separation
        return 0.0, float(thresholds[0])
    k = idx[0]
    if k == 0:
        return float((far[k] + frr[k]) / 2), float(thresholds[k])
    x0, x1 = diff[k - 1], diff[k]
    t = x0 / (x0 - x1) if x0 != x1 else 0.0
    eer = float((far[k - 1] + t * (far[k] - far[k - 1]) +
                 frr[k - 1] + t * (frr[k] - frr[k - 1])) / 2)
    thr = float(thresholds[k - 1] + t * (thresholds[k] - thresholds[k - 1]))
    return eer, thr


def _gather(frames, subject, spec, channel_idx, max_frames, rng):
    """Collect (N, C, 320) frames for a subject from a list of (session, task)."""
    out = []
    for (sess, task) in spec:
        key = f"{subject}|{sess}|{task}"
        if key not in frames:
            continue
        arr = frames[key][:, channel_idx, :]
        if len(arr) > max_frames:
            sel = rng.choice(len(arr), size=max_frames, replace=False)
            arr = arr[sel]
        out.append(arr)
    if not out:
        return None
    return np.concatenate(out, axis=0)


def evaluate_scenario(encoders, frames, channel_names, enrol_spec, probe_spec,
                      max_probe_frames: int = 20, nu: float = 0.1,
                      seed: int = 0, device: str = "cpu"):
    """Run one enrolment/verification scenario.

    Parameters
    ----------
    encoders : list of callables, one per channel, mapping a ``(N, 320)``
        array of frames to an ``(N, D)`` representation matrix.  A list of
        ``torch.nn.Module`` instances (the trained CNNs) is also accepted for
        convenience.
    enrol_spec : {subject: [(session, task), ...]}  enrolment frames.
    probe_spec : {subject: [(session, task), ...]}  verification probes.

    Returns
    -------
    dict with ``eer`` (fused), ``per_channel_eer`` and score arrays.
    """
    encoders = [_as_encoder(e, device) for e in encoders]
    rng = np.random.default_rng(seed)
    subjects = sorted(enrol_spec.keys())

    # ---- 1. per-channel, per-subject enrolment embeddings + one-class SVMs ----
    svms: dict[tuple[int, str], OneClassSVM] = {}
    for ci in range(len(channel_names)):
        for subj in subjects:
            spec = enrol_spec.get(subj, [])
            arr = _gather(frames, subj, spec, [ci], 10_000, rng)
            if arr is None:
                continue
            emb = encoders[ci](arr[:, 0, :])
            svms[(ci, subj)] = OneClassSVM(kernel="rbf", gamma="scale",
                                           nu=nu).fit(emb)

    # ---- 2. probe embeddings (computed once per channel) ----
    # probe_items: list of (probe_subject, channel_idx_array (n,320))
    probe_sets: dict[str, dict[int, np.ndarray]] = {}
    for subj in subjects:
        spec = probe_spec.get(subj, [])
        arr = _gather(frames, subj, spec, list(range(len(channel_names))),
                      max_probe_frames, rng)
        if arr is None:
            continue
        per_ch = {ci: encoders[ci](arr[:, ci, :])
                  for ci in range(len(channel_names))}
        probe_sets[subj] = per_ch

    # ---- 3. genuine / impostor scores, per channel and fused ----
    per_ch_gen: dict[int, list] = {ci: [] for ci in range(len(channel_names))}
    per_ch_imp: dict[int, list] = {ci: [] for ci in range(len(channel_names))}
    fused_gen, fused_imp = [], []
    for psubj, per_ch in probe_sets.items():
        n = next(iter(per_ch.values())).shape[0]
        for ci in range(len(channel_names)):
            # genuine: probe against its own enrolment model
            svm = svms.get((ci, psubj))
            if svm is not None:
                per_ch_gen[ci].extend(svm.decision_function(per_ch[ci]))
            # impostor: probe against every other subject's model
            for qsubj in subjects:
                if qsubj == psubj:
                    continue
                o_svm = svms.get((ci, qsubj))
                if o_svm is not None:
                    per_ch_imp[ci].extend(o_svm.decision_function(per_ch[ci]))
        # fused: mean decision across channels for each enrolled subject
        for qsubj in subjects:
            fused = np.mean(
                [svms[(ci, qsubj)].decision_function(per_ch[ci])
                 for ci in range(len(channel_names))
                 if (ci, qsubj) in svms], axis=0)
            if qsubj == psubj:
                fused_gen.extend(fused)
            else:
                fused_imp.extend(fused)

    per_channel_eer = {
        channel_names[ci]: compute_eer(per_ch_gen[ci], per_ch_imp[ci])[0]
        for ci in range(len(channel_names))
    }
    fused_eer, _ = compute_eer(fused_gen, fused_imp)
    return {
        "eer": fused_eer,
        "per_channel_eer": per_channel_eer,
        "genuine_scores": np.asarray(fused_gen),
        "impostor_scores": np.asarray(fused_imp),
    }

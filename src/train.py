"""Siamese training of the channel-specific CNNs (Maiorana 2021, Section 4.1).

Training protocol reproduced from the paper:

    * genuine pairs are built from frames of the *same* subject recorded in
      *different* sessions (and, when several protocols are used, possibly
      different tasks);
    * for every genuine pair, ``impostors_per_genuine`` (2) additional pairs
      with a frame from a *different* subject are added;
    * optimisation with stochastic gradient descent + momentum, batch size
      128, learning rate 1e-3 and weight decay 5e-3.

A separate model is optimised for every electrode.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import TrainConfig
from .model import ChannelCNN, contrastive_loss


def _pool(frames, subject, session, task, channel_idx):
    key = f"{subject}|{session}|{task}"
    if key not in frames:
        return None
    return frames[key][:, channel_idx, :]  # (n_frames, 320)


def build_pairs(frames, subjects, sessions, tasks, channel_idx, cfg: TrainConfig,
                allow_cross_task: bool, rng: np.random.Generator):
    """Sample genuine/impostor pairs for one training epoch of one channel.

    Returns ``(a, b, y)`` with ``y = 0`` for genuine and ``y = 1`` for impostor
    pairs; ``a`` and ``b`` have shape ``(N, 1, 320)``.
    """
    session_pairs = [(s1, s2) for i, s1 in enumerate(sessions)
                     for s2 in sessions[i + 1:]]
    a_list, b_list, y_list = [], [], []
    for subj in subjects:
        # cache the pools actually available for this subject
        avail = [(s, t) for s in sessions for t in tasks
                 if _pool(frames, subj, s, t, channel_idx) is not None]
        if len(avail) < 2:
            continue
        for _ in range(cfg.pairs_per_subject):
            s1, s2 = session_pairs[rng.integers(len(session_pairs))]
            t1 = tasks[rng.integers(len(tasks))]
            if allow_cross_task:
                t2 = tasks[rng.integers(len(tasks))]
            else:
                t2 = t1
            p1 = _pool(frames, subj, s1, t1, channel_idx)
            p2 = _pool(frames, subj, s2, t2, channel_idx)
            if p1 is None or p2 is None or len(p1) == 0 or len(p2) == 0:
                continue
            f1 = p1[rng.integers(len(p1))]
            f2 = p2[rng.integers(len(p2))]
            a_list.append(f1)
            b_list.append(f2)
            y_list.append(0)
            # impostor pairs: replace one element with another subject's frame
            for _ in range(cfg.impostors_per_genuine):
                other = subj
                while other == subj:
                    other = subjects[rng.integers(len(subjects))]
                s_o = sessions[rng.integers(len(sessions))]
                t_o = tasks[rng.integers(len(tasks))]
                p_o = _pool(frames, other, s_o, t_o, channel_idx)
                if p_o is None or len(p_o) == 0:
                    continue
                f_o = p_o[rng.integers(len(p_o))]
                if rng.integers(2) == 0:
                    a_list.append(f1); b_list.append(f_o)
                else:
                    a_list.append(f_o); b_list.append(f2)
                y_list.append(1)

    a = np.stack(a_list)[:, None, :] if a_list else np.empty((0, 1, 320), np.float32)
    b = np.stack(b_list)[:, None, :] if b_list else np.empty((0, 1, 320), np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return a.astype(np.float32), b.astype(np.float32), y


def train_channel_model(frames, subjects, sessions, tasks, channel_idx,
                        cfg: TrainConfig, allow_cross_task: bool = False,
                        device: str = "cpu", verbose: bool = False) -> ChannelCNN:
    """Train the CNN of a single electrode and return it (in eval mode)."""
    rng = np.random.default_rng(cfg.seed + channel_idx)
    torch.manual_seed(cfg.seed + channel_idx)

    model = ChannelCNN(dropout=cfg.dropout, embedding_dim=cfg.embedding_dim).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                            momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        a, b, y = build_pairs(frames, subjects, sessions, tasks, channel_idx,
                              cfg, allow_cross_task, rng)
        if len(y) == 0:
            break
        perm = rng.permutation(len(y))
        a, b, y = a[perm], b[perm], y[perm]
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(y), cfg.batch_size):
            xa = torch.from_numpy(a[i:i + cfg.batch_size]).to(device)
            xb = torch.from_numpy(b[i:i + cfg.batch_size]).to(device)
            yy = torch.from_numpy(y[i:i + cfg.batch_size]).to(device)
            optim.zero_grad()
            loss = contrastive_loss(model(xa), model(xb), yy, margin=cfg.margin)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        if verbose and (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"    [ch{channel_idx}] epoch {epoch+1}/{cfg.epochs} "
                  f"loss={epoch_loss / max(n_batches,1):.4f} n_pairs={len(y)}")
    model.eval()
    return model


@torch.no_grad()
def embed_frames(model: ChannelCNN, frames_ch: np.ndarray, device: str = "cpu",
                 batch_size: int = 512) -> np.ndarray:
    """Return the (N, 256) representation of ``(N, 320)`` single-channel frames."""
    model.eval()
    out = []
    for i in range(0, len(frames_ch), batch_size):
        x = torch.from_numpy(frames_ch[i:i + batch_size])[:, None, :].to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, model.conv5.out_channels))

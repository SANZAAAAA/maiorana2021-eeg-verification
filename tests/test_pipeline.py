"""Fast sanity checks for the reproduction building blocks.

Run with:  ../EEGLearning/.venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate, preprocessing                       # noqa: E402
from src.model import ChannelCNN, contrastive_loss            # noqa: E402


def test_preprocessing_shapes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((19, 1600))          # 19 ch x 10 s @160 Hz
    proc = preprocessing.preprocess_stream(x, fs_in=160.0, fs_out=64.0)
    assert proc.shape[0] == 19
    assert abs(proc.shape[1] - 640) <= 2         # 10 s @64 Hz
    frames = preprocessing.frame_stream(proc, fs=64.0, win_s=5.0, overlap=0.8)
    assert frames.shape[1:] == (19, 320)         # 1 x S*H per channel


def test_car_removes_channel_mean():
    x = np.ones((5, 100), dtype=np.float64)
    x[0] += 3.0
    out = preprocessing.common_average_reference(x)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)


def test_cnn_output_dimension():
    net = ChannelCNN()
    y = net(torch.zeros(4, 1, 320))
    assert y.shape == (4, 256)


def test_contrastive_loss_behaviour():
    a = torch.zeros(2, 8)
    same = a.clone()
    far = a + 10.0
    y_same = torch.zeros(2)
    y_diff = torch.ones(2)
    # identical embeddings with a genuine label -> (numerically) zero loss
    assert abs(float(contrastive_loss(a, same, y_same))) < 1e-9
    # far-apart embeddings beyond the margin with an impostor label -> zero loss
    assert abs(float(contrastive_loss(a, far, y_diff, margin=1.0))) < 1e-9
    # close embeddings with an impostor label -> positive loss
    assert float(contrastive_loss(a, a + 0.1, y_diff, margin=1.0)) > 0.0


def test_eer_perfect_separation():
    eer, _ = evaluate.compute_eer(np.array([1.0, 2.0]), np.array([-1.0, -2.0]))
    assert eer == 0.0


def test_eer_matches_theory_for_gaussian_scores():
    # genuine ~ N(+1,1), impostor ~ N(0,1): EER = Phi(-1/2) ~= 0.309
    rng = np.random.default_rng(1)
    genuine = rng.standard_normal(20000) + 1.0
    impostor = rng.standard_normal(20000)
    eer, _ = evaluate.compute_eer(genuine, impostor)
    assert abs(eer - 0.3085) < 0.02

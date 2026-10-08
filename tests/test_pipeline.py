"""快速单元测试：不依赖真实 EEG 数据，几秒内跑完。

    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate, preprocessing                       # noqa: E402
from src.model import ChannelCNN, contrastive_loss            # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing_shapes(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((19, 1600))          # 19 ch x 10 s @160 Hz
        proc = preprocessing.preprocess_stream(x, fs_in=160.0, fs_out=64.0)
        self.assertEqual(proc.shape[0], 19)
        self.assertLessEqual(abs(proc.shape[1] - 640), 2)     # 10 s @64 Hz
        frames = preprocessing.frame_stream(proc, fs=64.0, win_s=5.0, overlap=0.8)
        self.assertEqual(tuple(frames.shape[1:]), (19, 320))  # 1 x S*H per channel

    def test_car_removes_channel_mean(self):
        x = np.ones((5, 100), dtype=np.float64)
        x[0] += 3.0
        out = preprocessing.common_average_reference(x)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)


class TestModel(unittest.TestCase):
    def test_cnn_output_dimension(self):
        net = ChannelCNN()
        y = net(torch.zeros(4, 1, 320))
        self.assertEqual(tuple(y.shape), (4, 256))

    def test_contrastive_loss_behaviour(self):
        a = torch.zeros(2, 8)
        same = a.clone()
        far = a + 10.0
        y_same = torch.zeros(2)
        y_diff = torch.ones(2)
        # identical embeddings with a genuine label -> (numerically) zero loss
        self.assertLess(abs(float(contrastive_loss(a, same, y_same))), 1e-9)
        # far-apart embeddings beyond the margin with an impostor label -> zero loss
        self.assertLess(abs(float(contrastive_loss(a, far, y_diff, margin=1.0))), 1e-9)
        # close embeddings with an impostor label -> positive loss
        self.assertGreater(float(contrastive_loss(a, a + 0.1, y_diff, margin=1.0)), 0.0)


class TestEvaluate(unittest.TestCase):
    def test_eer_perfect_separation(self):
        eer, _ = evaluate.compute_eer(np.array([1.0, 2.0]), np.array([-1.0, -2.0]))
        self.assertEqual(eer, 0.0)

    def test_eer_matches_theory_for_gaussian_scores(self):
        # genuine ~ N(+1,1), impostor ~ N(0,1): EER = Phi(-1/2) ~= 0.309
        rng = np.random.default_rng(1)
        genuine = rng.standard_normal(20000) + 1.0
        impostor = rng.standard_normal(20000)
        eer, _ = evaluate.compute_eer(genuine, impostor)
        self.assertLess(abs(eer - 0.3085), 0.02)


if __name__ == "__main__":
    unittest.main()

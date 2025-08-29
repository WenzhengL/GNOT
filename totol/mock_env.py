# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from typing import Tuple


def make_sample(n_points: int = 512) -> list:
    X = np.random.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)
    Y = np.zeros((n_points, 5), dtype=np.float32)
    Y[:, 0] = np.sin(X[:, 0] * 3.14) + 0.1 * np.random.randn(n_points).astype(np.float32)  # pressure
    Y[:, 1:] = 0.1 * np.random.randn(n_points, 4).astype(np.float32)  # shear...
    Theta = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(0.1, 1.0)], dtype=np.float32)
    empty_branch = (np.zeros((n_points, 1), dtype=np.float32),)
    return [X, Y, Theta, empty_branch]


def create_synthetic_al_split(out_dir: str, n_labeled=10, n_unlabeled=100, n_test=20) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    labeled = [make_sample(np.random.randint(400, 800)) for _ in range(n_labeled)]
    unlabeled = [make_sample(np.random.randint(400, 800)) for _ in range(n_unlabeled)]
    test = [make_sample(np.random.randint(400, 800)) for _ in range(n_test)]
    lp = os.path.join(out_dir, 'al_labeled.pkl')
    up = os.path.join(out_dir, 'al_unlabeled.pkl')
    tp = os.path.join(out_dir, 'al_test.pkl')
    with open(lp, 'wb') as f:
        pickle.dump(labeled, f)
    with open(up, 'wb') as f:
        pickle.dump(unlabeled, f)
    with open(tp, 'wb') as f:
        pickle.dump(test, f)
    return lp, up, tp


def build_mock_model_tuple(example_sample: list):
    # 返回 (model, metric_func, device)
    class MockModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, g, u_p, g_u):
            import torch
            Y = example_sample[1]
            return torch.zeros_like(torch.from_numpy(Y))

    def metric_func(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    device = 'cpu'
    return (MockModel(), metric_func, device)

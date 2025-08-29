# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import pickle
import shutil
from typing import Dict, List

from .strategy_registry import StrategySpec, build_default_strategies


def _try_import_albz():
    try:
        import albz
        return albz
    except Exception:
        return None


ALBZ = _try_import_albz()

TEMP_DIR = "/home/v-wenliao/gnot/GNOT/data/al_bz/data"
os.makedirs(TEMP_DIR, exist_ok=True)


def _cheap_reward_proxy(samples: List):
    # 模型不可用时的奖励代理：Y 振幅越大→潜在收益大
    vals = []
    for s in samples:
        try:
            Y = np.array(s[1])
            vals.append(float(np.mean(np.abs(Y))))
        except Exception:
            vals.append(0.0)
    if not vals:
        return 0.0
    v = float(np.mean(vals))
    return float(1.0 - np.exp(-v / max(1.0, v + 1e-6)))


def _predict_mae_on_samples(model_tuple, samples: List) -> float:
    if model_tuple is None or ALBZ is None:
        return _cheap_reward_proxy(samples)
    try:
        from data_utils import get_dataset, MIODataLoader
        args = ALBZ.get_al_args()
        args.dataset = 'al_bz'
    except Exception:
        return _cheap_reward_proxy(samples)

    model, metric_func, device = model_tuple
    maes = []
    for i, sample in enumerate(samples):
        tmp, std_path = None, None
        try:
            ts = int(time.time() * 1_000_000)
            tmp = os.path.join(TEMP_DIR, f'agent_eval_{i}_{ts}.pkl')
            with open(tmp, 'wb') as f:
                pickle.dump([sample], f)
            std_path = os.path.join(TEMP_DIR, 'al_test.pkl')
            shutil.copy2(tmp, std_path)

            _, ds = get_dataset(args)
            loader = MIODataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

            Y_true = np.array(sample[1])
            Y_pred = None
            import torch
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    g, u_p, g_u = batch
                    g = g.to(device)
                    u_p = u_p.to(device)
                    if hasattr(g_u, 'to'):
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    model = model.to(device)
                    out = model(g, u_p, g_u)
                    if hasattr(out, 'detach'):
                        Y_pred = out.detach().cpu().numpy()
                    else:
                        Y_pred = np.array(out)
                    break

            if Y_pred is not None and Y_pred.shape == Y_true.shape:
                mae = float(np.mean(np.abs(Y_pred - Y_true)))
                maes.append(mae)
            else:
                maes.append(0.0)
        except Exception:
            maes.append(0.0)
        finally:
            try:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
                if std_path and os.path.exists(std_path):
                    os.remove(std_path)
            except Exception:
                pass

    if not maes:
        return 0.0
    avg_mae = float(np.mean(maes))
    scale = max(1.0, 10.0)
    reward = 1.0 - np.exp(-avg_mae / scale)
    return float(np.clip(reward, 0.0, 1.0))


class StrategyAgent:
    def __init__(
        self,
        strategies: Dict[str, StrategySpec] = None,
        state_path: str = os.path.join(os.path.dirname(__file__), "state.json"),
        eta: float = 0.3,
        candidate_size: int = 16,
        bz_scaling_method: str = 'adaptive',
    ):
        self.strategies = strategies or build_default_strategies()
        self.state_path = state_path
        self.eta = eta
        self.candidate_size = candidate_size
        self.bz_scaling_method = bz_scaling_method
        self.weights = {name: 1.0 for name in self.strategies.keys()}
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    obj = json.load(f)
                if "weights" in obj:
                    for k, v in obj["weights"].items():
                        if k in self.weights:
                            self.weights[k] = float(v)
        except Exception:
            pass

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump({"weights": self.weights}, f, indent=2)

    def _call_strategy(self, spec: StrategySpec, model_tuple, labeled_data, unlabeled_data, k: int) -> List[int]:
        try:
            kwargs = {}
            if spec.name == 'bz_scaled':
                kwargs['scaling_method'] = self.bz_scaling_method
            return spec.func(model_tuple, labeled_data, unlabeled_data, k, **kwargs)
        except Exception:
            return []

    def _update_weight_exp3(self, name: str, reward: float):
        self.weights[name] = float(self.weights[name] * np.exp(self.eta * reward))

    def evaluate_rewards(self, model_tuple, labeled_data, unlabeled_data) -> Dict[str, float]:
        rewards = {}
        for name, spec in self.strategies.items():
            k = max(4, min(self.candidate_size, len(unlabeled_data)))
            cand_idx = self._call_strategy(spec, model_tuple, labeled_data, unlabeled_data, k)
            if not cand_idx:
                rewards[name] = 0.0
                continue
            samples = [unlabeled_data[i] for i in cand_idx if 0 <= i < len(unlabeled_data)]
            reward = _predict_mae_on_samples(model_tuple, samples)
            rewards[name] = reward
            self._update_weight_exp3(name, reward)
        self._save_state()
        return rewards

    def select_indices(self, model_tuple, labeled_data, unlabeled_data, select_num: int) -> List[int]:
        k = max(select_num, self.candidate_size)
        N = len(unlabeled_data)
        scores = np.zeros(N, dtype=float)

        rewards = self.evaluate_rewards(model_tuple, labeled_data, unlabeled_data)
        _ = rewards  # 可用于日志
        w_sum = sum(self.weights.values()) or 1.0
        norm_w = {k: v / w_sum for k, v in self.weights.items()}

        for name, spec in self.strategies.items():
            idxs = self._call_strategy(spec, model_tuple, labeled_data, unlabeled_data, k)
            if not idxs:
                continue
            for rank, idx in enumerate(idxs):
                if 0 <= idx < N:
                    scores[idx] += norm_w[name] * (1.0 - rank / max(1, k - 1))

        chosen = np.argsort(scores)[-select_num:]
        return chosen.tolist()

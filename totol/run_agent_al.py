# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import numpy as np

# Support running both as a module and as a standalone script from the totol folder
try:
    # Case 1: python -m totol.run_agent_al (package-relative imports work)
    from .agent_selector import StrategyAgent
    from .strategy_registry import build_default_strategies
    from .llm_selector import LLMStrategyDecider
    from .mock_env import create_synthetic_al_split, build_mock_model_tuple
except Exception:
    # Case 2: python totol/run_agent_al.py (from project root) or from inside totol/
    import sys
    from pathlib import Path
    THIS_FILE = Path(__file__).resolve()
    PKG_DIR = str(THIS_FILE.parent)
    ROOT = str(THIS_FILE.parents[1])
    # Prefer root first to enable 'totol.*' absolute imports
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from totol.agent_selector import StrategyAgent
        from totol.strategy_registry import build_default_strategies
        from totol.llm_selector import LLMStrategyDecider
        from totol.mock_env import create_synthetic_al_split, build_mock_model_tuple
    except Exception:
        # Case 3: last resort, import local modules directly
        if PKG_DIR not in sys.path:
            sys.path.insert(0, PKG_DIR)
        from agent_selector import StrategyAgent
        from strategy_registry import build_default_strategies
        from llm_selector import LLMStrategyDecider
        from mock_env import create_synthetic_al_split, build_mock_model_tuple


def load_al_data(data_update_dir: str):
    lp = os.path.join(data_update_dir, 'al_labeled.pkl')
    up = os.path.join(data_update_dir, 'al_unlabeled.pkl')
    tp = os.path.join(data_update_dir, 'al_test.pkl')
    with open(lp, 'rb') as f:
        labeled = pickle.load(f)
    with open(up, 'rb') as f:
        unlabeled = pickle.load(f)
    with open(tp, 'rb') as f:
        test = pickle.load(f)
    return labeled, unlabeled, test


def try_train_model(labeled):
    # 优先调用 albz.train_model，失败则回退 mock
    try:
        import albz
        model_tuple = albz.train_model(labeled)
        if model_tuple is None:
            raise RuntimeError("train_model returned None")
        return model_tuple
    except Exception:
        return build_mock_model_tuple(labeled[0])


def try_evaluate_model(model_tuple, test):
    try:
        import albz
        metric = albz.evaluate_model(model_tuple, test)
        return metric
    except Exception:
        # 简易 mock 评估
        return {"metric": float(np.random.uniform(0.05, 0.15))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_update_dir', type=str, default="/home/v-wenliao/gnot/GNOT/data/al_bz")
    parser.add_argument('--select_num', type=int, default=20)
    parser.add_argument('--mode', type=str, default='bandit', choices=['bandit', 'llm', 'hybrid'])
    parser.add_argument('--bz_scaling', type=str, default='adaptive')
    parser.add_argument('--create_synth', action='store_true', help='生成合成数据')
    parser.add_argument('--provider', type=str, default='none', help='LLM provider: none|openai')
    parser.add_argument('--model', type=str, default='', help='LLM model id, e.g., gpt-4o-mini')
    parser.add_argument('--metrics_csv', type=str, default='', help='Path to metrics.csv for LLM decisions')
    args = parser.parse_args()

    if args.create_synth:
        create_synthetic_al_split(args.data_update_dir, n_labeled=8, n_unlabeled=60, n_test=12)

    labeled, unlabeled, test = load_al_data(args.data_update_dir)
    print(f"Data: labeled={len(labeled)}, unlabeled={len(unlabeled)}, test={len(test)}")

    model_tuple = try_train_model(labeled)

    strategies = build_default_strategies()
    agent = StrategyAgent(strategies, candidate_size=max(8, min(16, len(unlabeled) // 10)), bz_scaling_method=args.bz_scaling)

    def read_metrics_csv(path: str):
        if not path or not os.path.exists(path):
            return {}
        import csv
        cols = ["pressure","wall-shear","x-wall-shear","y-wall-shear","z-wall-shear"]
        series = {c: [] for c in cols}
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for c in cols:
                    try:
                        series[c].append(float(row[c]))
                    except Exception:
                        pass
        return series

    if args.mode == 'bandit':
        sel = agent.select_indices(model_tuple, labeled, unlabeled, args.select_num)
        print(f"[Bandit] Selected {len(sel)} indices. Head: {sel[:10]}")
    elif args.mode == 'llm':
        decider = LLMStrategyDecider(provider=args.provider, model=args.model)
        recent_metrics = read_metrics_csv(args.metrics_csv) or {"pressure": [0.12, 0.10], "wall-shear": [0.02, 0.02]}
        choice = decider.decide(recent_metrics, list(strategies.keys()))
        spec = strategies.get(choice) or list(strategies.values())[0]
        sel = spec.func(model_tuple, labeled, unlabeled, args.select_num)
        print(f"[LLM] Strategy={choice}, Selected {len(sel)} indices. Head: {sel[:10]}")
    else:  # hybrid
        decider = LLMStrategyDecider(provider=args.provider, model=args.model)
        recent_metrics = read_metrics_csv(args.metrics_csv) or {"pressure": [0.12, 0.10], "wall-shear": [0.02, 0.02]}
        choice = decider.decide(recent_metrics, list(strategies.keys()))
        print(f"[Hybrid] LLM choice={choice}")
        base_sel = strategies[choice].func(model_tuple, None, unlabeled, max(args.select_num, 16))
        bandit_sel = agent.select_indices(model_tuple, None, unlabeled, args.select_num)
        half = args.select_num // 2
        union = list(dict.fromkeys((base_sel[:half] if base_sel else []) + bandit_sel))[:args.select_num]
        sel = union
        print(f"[Hybrid] Selected {len(sel)} indices. Head: {sel[:10]}")

    metric = try_evaluate_model(model_tuple, test)
    print(f"Eval metric: {metric}")


if __name__ == "__main__":
    main()

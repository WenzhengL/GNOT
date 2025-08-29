# -*- coding: utf-8 -*-
import importlib
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class StrategySpec:
    name: str
    func: Callable
    needs_model: bool = True
    needs_labeled: bool = False
    # 统一调用签名：call(model_tuple, labeled_data, unlabeled_data, select_num)


def _wrap_strategy(func, needs_model=True, needs_labeled=False):
    def caller(model_tuple, labeled_data, unlabeled_data, select_num, **kwargs):
        # 根据被包装函数的签名进行适配
        try:
            if needs_model and needs_labeled:
                try:
                    return func(model_tuple, labeled_data, unlabeled_data, select_num, **kwargs)
                except TypeError:
                    # 兼容 geometry_variance_query(labeled_data, unlabeled_data, select_num)
                    return func(labeled_data, unlabeled_data, select_num, **kwargs)
            elif needs_model and not needs_labeled:
                return func(model_tuple, unlabeled_data, select_num, **kwargs)
            elif not needs_model and needs_labeled:
                return func(labeled_data, unlabeled_data, select_num, **kwargs)
            else:
                return func(unlabeled_data, select_num, **kwargs)
        except Exception:
            return []

    return caller


def _safe_import(module_name: str, attr: str) -> Optional[Callable]:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    except Exception:
        return None


def build_default_strategies() -> Dict[str, StrategySpec]:
    specs: Dict[str, StrategySpec] = {}

    # 来自 albz 的策略（可选）
    pdiff = _safe_import('albz', 'prediction_difference_query')
    if pdiff:
        specs['pred_diff'] = StrategySpec(
            name='pred_diff',
            func=_wrap_strategy(pdiff, needs_model=True, needs_labeled=False),
            needs_model=True,
            needs_labeled=False,
        )

    pdiff_fast = _safe_import('albz', 'prediction_difference_query_fast')
    if pdiff_fast:
        specs['pred_diff_fast'] = StrategySpec(
            name='pred_diff_fast',
            func=_wrap_strategy(pdiff_fast, needs_model=True, needs_labeled=False),
            needs_model=True,
            needs_labeled=False,
        )

    qbc = _safe_import('albz', 'qbc_query_fixed')
    if qbc:
        specs['qbc'] = StrategySpec(
            name='qbc',
            func=_wrap_strategy(qbc, needs_model=True, needs_labeled=False),
            needs_model=True,
            needs_labeled=False,
        )

    pa = _safe_import('albz', 'pa_query_fixed')
    if pa:
        specs['pa'] = StrategySpec(
            name='pa',
            func=_wrap_strategy(pa, needs_model=True, needs_labeled=False),
            needs_model=True,
            needs_labeled=False,
        )

    gv = _safe_import('albz', 'geometry_variance_query')
    if gv:
        specs['gv'] = StrategySpec(
            name='gv',
            func=_wrap_strategy(gv, needs_model=False, needs_labeled=True),
            needs_model=False,
            needs_labeled=True,
        )

    gv_fast = _safe_import('albz', 'geometry_variance_query_fast')
    if gv_fast:
        specs['gv_fast'] = StrategySpec(
            name='gv_fast',
            func=_wrap_strategy(gv_fast, needs_model=False, needs_labeled=True),
            needs_model=False,
            needs_labeled=True,
        )

    # 可选 BZ 系列
    bz = _safe_import('alpa', 'bz_query')
    if bz:
        specs['bz'] = StrategySpec(
            name='bz',
            func=_wrap_strategy(bz, needs_model=True, needs_labeled=False),
            needs_model=True,
            needs_labeled=False,
        )

    bz_scaled = _safe_import('bz_strategy_scale_fix', 'bz_query_with_dimension_scaling')
    if bz_scaled:
        def bz_scaled_wrapped(model_tuple, labeled_data, unlabeled_data, select_num, **kwargs):
            scaling_method = kwargs.get('scaling_method', 'adaptive')
            try:
                return bz_scaled(model_tuple, unlabeled_data, select_num, scaling_method=scaling_method)
            except Exception:
                return []

        specs['bz_scaled'] = StrategySpec(
            name='bz_scaled',
            func=bz_scaled_wrapped,
            needs_model=True,
            needs_labeled=False,
        )

    # 本地回退策略
    try:
        from .local_strategies import random_query, magnitude_query, diversity_query

        specs['random'] = StrategySpec(
            name='random',
            func=_wrap_strategy(random_query, needs_model=False, needs_labeled=False),
            needs_model=False,
            needs_labeled=False,
        )
        specs['magnitude'] = StrategySpec(
            name='magnitude',
            func=_wrap_strategy(magnitude_query, needs_model=False, needs_labeled=False),
            needs_model=False,
            needs_labeled=False,
        )
        specs['diversity'] = StrategySpec(
            name='diversity',
            func=_wrap_strategy(diversity_query, needs_model=False, needs_labeled=True),
            needs_model=False,
            needs_labeled=True,
        )
    except Exception:
        pass

    return specs

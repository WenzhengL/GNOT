# -*- coding: utf-8 -*-
from typing import Dict, Optional, List, Any
import os
import json


class LLMStrategyDecider:
    """
    规则回退/LLM决策器：
    - 输入: recent_metrics = {"pressure":[...], "wall-shear":[...], ...}
    - 输出: 一个策略名称
    """

    def __init__(self, provider: str = "none", model: str = "", api_key: Optional[str] = None, timeout: float = 20.0):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout

    def decide(self, recent_metrics: Dict[str, List[float]], candidate_strategies: List[str]) -> str:
        """Choose a strategy using OpenAI when configured, else rules fallback."""
        if self.provider == "openai" and self.api_key:
            try:
                result = self._decide_with_openai(recent_metrics, candidate_strategies)
                if isinstance(result, dict):
                    choice = result.get("strategy")
                    if isinstance(choice, str) and choice in candidate_strategies:
                        return choice
            except Exception:
                # fall through to rules
                pass

        # Rules fallback
        try:
            last_p = recent_metrics.get("pressure", [])[-1] if recent_metrics.get("pressure") else None
            last_ws = recent_metrics.get("wall-shear", [])[-1] if recent_metrics.get("wall-shear") else None

            if last_p is not None and last_ws is not None and last_p > 5 * last_ws and "bz_scaled" in candidate_strategies:
                return "bz_scaled"

            # 简单停滞判定（按最后一项与倒数第二项）
            agg_last = []
            agg_prev = []
            for k in ["pressure", "wall-shear", "x-wall-shear", "y-wall-shear", "z-wall-shear"]:
                series = recent_metrics.get(k) or []
                if len(series) >= 2:
                    agg_last.append(series[-1])
                    agg_prev.append(series[-2])
            if agg_last and agg_prev:
                last_sum = sum(agg_last)
                prev_sum = sum(agg_prev)
                if prev_sum > 0 and last_sum >= 0.98 * prev_sum and "pred_diff_fast" in candidate_strategies:
                    return "pred_diff_fast"

            for pref in [
                "bz",
                "bz_scaled",
                "pred_diff",
                "qbc",
                "pa",
                "gv_fast",
                "gv",
                "magnitude",
                "diversity",
                "random",
            ]:
                if pref in candidate_strategies:
                    return pref
            return candidate_strategies[0]
        except Exception:
            return candidate_strategies[0]

    # --- OpenAI integration ---
    def _decide_with_openai(self, recent_metrics: Dict[str, List[float]], candidate_strategies: List[str]) -> Dict[str, Any]:
        client = None
        try:
            # New-style SDK
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=self.api_key)
            model = self.model or "gpt-4o-mini"
            sys_prompt, user_prompt = self._build_prompts(recent_metrics, candidate_strategies)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=self.timeout,
            )
            content = resp.choices[0].message.content if resp and resp.choices else ""
            return self._parse_llm_json(content, candidate_strategies)
        except Exception:
            # Legacy SDK fallback
            try:
                import openai  # type: ignore
                openai.api_key = self.api_key
                model = self.model or "gpt-4o-mini"
                sys_prompt, user_prompt = self._build_prompts(recent_metrics, candidate_strategies)
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""
                return self._parse_llm_json(content, candidate_strategies)
            except Exception as e2:
                raise e2

    @staticmethod
    def _build_prompts(recent_metrics: Dict[str, List[float]], candidate_strategies: List[str]):
        sys_prompt = (
            "You are an active-learning strategy decider. Choose a strategy from candidates for the next round. "
            "Only output strict JSON with keys: strategy (string), rationale (string), weight_adjustments (object of floats -1..1), confidence (0..1)."
        )
        payload = {
            "recent_metrics": recent_metrics,
            "candidates": candidate_strategies,
            "instructions": [
                "Prefer 'bz_scaled' if pressure dominates (>> wall-shear).",
                "Prefer 'pred_diff_fast' if recent improvements stagnate (<~2%).",
                "Otherwise choose the most promising from candidates.",
            ],
            "output_format": {
                "strategy": "one of candidates",
                "rationale": "short reasoning",
                "weight_adjustments": {c: 0.0 for c in candidate_strategies},
                "confidence": 0.8,
            },
        }
        user_prompt = (
            "Decide strategy based on the following data. Output JSON only, no prose.\n" + json.dumps(payload)
        )
        return sys_prompt, user_prompt

    @staticmethod
    def _parse_llm_json(text: str, candidate_strategies: List[str]) -> Dict[str, Any]:
        if not text:
            return {"strategy": candidate_strategies[0], "rationale": "empty", "weight_adjustments": {}, "confidence": 0.0}
        # Try parse as-is
        try:
            obj = json.loads(text)
            return obj
        except Exception:
            pass
        # Extract first JSON object heuristically
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj
            except Exception:
                pass
        # Fallback minimal
        return {"strategy": candidate_strategies[0], "rationale": "fallback", "weight_adjustments": {}, "confidence": 0.0}


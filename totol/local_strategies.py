# -*- coding: utf-8 -*-
import random
import numpy as np
from typing import List


def random_query(unlabeled_data: List, select_num: int) -> List[int]:
    n = len(unlabeled_data)
    idx = list(range(n))
    random.shuffle(idx)
    return idx[:select_num]


def magnitude_query(unlabeled_data: List, select_num: int) -> List[int]:
    scores = []
    for i, s in enumerate(unlabeled_data):
        try:
            Y = np.array(s[1])
            scores.append((i, float(np.mean(np.abs(Y)))))
        except Exception:
            scores.append((i, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:select_num]]


def diversity_query(labeled_data: List, unlabeled_data: List, select_num: int) -> List[int]:
    # 简单的最远点采样（基于 X 的均值特征）
    if not unlabeled_data:
        return []
    try:
        feats = []
        for s in unlabeled_data:
            X = np.array(s[0])
            feats.append(np.mean(X, axis=0))
        feats = np.array(feats)
        center = np.mean(feats, axis=0)
        dists = np.linalg.norm(feats - center, axis=1)
        first = int(np.argmax(dists))
        selected = [first]
        mask = np.ones(len(unlabeled_data), dtype=bool)
        mask[first] = False

        while len(selected) < min(select_num, len(unlabeled_data)):
            rem_idx = np.where(mask)[0]
            if len(rem_idx) == 0:
                break
            min_d = []
            for ri in rem_idx:
                d = np.min([np.linalg.norm(feats[ri] - feats[sj]) for sj in selected])
                min_d.append(d)
            pick = rem_idx[int(np.argmax(min_d))]
            selected.append(pick)
            mask[pick] = False
        return selected
    except Exception:
        return random_query(unlabeled_data, select_num)

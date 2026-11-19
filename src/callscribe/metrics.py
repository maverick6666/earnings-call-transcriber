"""점수: 전체 오류율 · 경계 오류율 · 고유명사/숫자 오류율 · 헛소리 삽입률.

경계 오류율이 이 저장소의 핵심 숫자다. 청크를 자른 시각 ±2초 안에서 생긴 오류만 따로 센다
(사용자가 겪은 '사이 단어가 사라지는' 문제를 직접 재는 숫자).

정렬된 연산마다 녹음 시각을 붙여야 경계 안인지 안다.
- 받아쓰기 낱말이 있는 연산(맞음 · 바꿈 · 끼어듦)은 그 낱말의 시각.
- 정답에 시각이 있으면(합성 한국어 자료) 정답 시각을 쓴다.
- 정답에 시각이 없고 빠진 낱말(D)이면 앞뒤 연산 시각 사이에 고르게 놓는다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .align import Op, align, edit_counts
from .words import RefToken, Word

NUMBER_TAGS = {"CARDINAL", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "YEAR", "DATE", "TIME"}
ENTITY_TAGS = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
               "ABBREVIATION", "WEBSITE", "ALPHANUMERIC"}
_DIGIT = re.compile(r"\d")
HALLUCINATION_RUN = 4  # 정답에 없는 낱말이 이만큼 이어지면 헛소리로 센다


def error_rate(ref: Sequence[str], hyp: Sequence[str], ops: Sequence[Op] | None = None) -> dict:
    ops = align(ref, hyp) if ops is None else ops
    c = edit_counts(ops)
    n = len(ref)
    return {"ref": n, "S": c["S"], "D": c["D"], "I": c["I"],
            "rate": round((c["S"] + c["D"] + c["I"]) / n, 5) if n else None}


def op_times(ops: Sequence[Op], ref: Sequence[RefToken], hyp: Sequence[Word]) -> np.ndarray:
    """연산마다 녹음 시각(초). 정해지지 않으면 앞뒤로 채운다."""
    t = np.full(len(ops), np.nan)
    for k, (_op, i, j) in enumerate(ops):
        if i is not None and ref[i].start is not None:
            r = ref[i]
            t[k] = (r.start + (r.end if r.end is not None else r.start)) / 2
        elif j is not None:
            t[k] = hyp[j].mid
    known = np.nonzero(~np.isnan(t))[0]
    if len(known) == 0:
        return t
    missing = np.nonzero(np.isnan(t))[0]
    t[missing] = np.interp(missing, known, t[known])
    return t


def near(times: np.ndarray, points: Sequence[float], radius: float) -> np.ndarray:
    """시각마다 points 중 하나와 radius 안에 있는가."""
    if len(points) == 0:
        return np.zeros(len(times), dtype=bool)
    p = np.sort(np.asarray(points, dtype=float))
    idx = np.searchsorted(p, times)
    left = np.abs(times - p[np.clip(idx - 1, 0, len(p) - 1)])
    right = np.abs(times - p[np.clip(idx, 0, len(p) - 1)])
    return np.minimum(left, right) <= radius


def region_error(ops: Sequence[Op], times: np.ndarray, mask: np.ndarray) -> dict:
    """mask 안 연산만으로 오류율: (바꿈 + 빠짐 + 끼어듦) ÷ 정답 낱말 수."""
    kinds = np.array([op for op, _, _ in ops])
    ref_words = int(np.isin(kinds[mask], ["C", "S", "D"]).sum())
    errors = int(np.isin(kinds[mask], ["S", "D", "I"]).sum())
    return {"ref": ref_words, "errors": errors, "rate": round(errors / ref_words, 5) if ref_words else None}


def tagged_error(ops: Sequence[Op], ref: Sequence[RefToken]) -> dict:
    """꼬리표 달린 정답 낱말 중 틀리거나 빠진 비율. 꼬리표가 없는 자료는 숫자가 든 낱말을 숫자로 본다."""
    groups = {"number": 0, "entity": 0}
    wrong = {"number": 0, "entity": 0}
    any_tags = any(r.tags for r in ref)
    for op, i, _ in ops:
        if i is None:
            continue
        tags = set(ref[i].tags)
        is_num = bool(tags & NUMBER_TAGS) if any_tags else bool(_DIGIT.search(ref[i].text))
        is_ent = bool(tags & ENTITY_TAGS)
        for g, hit in (("number", is_num), ("entity", is_ent)):
            if hit:
                groups[g] += 1
                wrong[g] += op in ("S", "D")
    return {g: {"ref": groups[g], "errors": wrong[g],
                "rate": round(wrong[g] / groups[g], 5) if groups[g] else None} for g in groups}


def hallucinations(ops: Sequence[Op], n_ref: int, min_run: int = HALLUCINATION_RUN) -> dict:
    """정답에 없는 낱말이 min_run 개 넘게 이어진 덩어리. 무음 · 대기음에서 지어낸 문장을 잡는다."""
    runs, words, cur = 0, 0, 0
    for op, _, _ in [*ops, ("C", None, None)]:
        if op == "I":
            cur += 1
            continue
        if cur >= min_run:
            runs += 1
            words += cur
        cur = 0
    return {"runs": runs, "words": words, "per_1k_ref_words": round(words / n_ref * 1000, 3) if n_ref else None}


def confidence(ops: Sequence[Op], hyp: Sequence[Word]) -> dict:
    """단어 확률이 맞음 · 틀림을 얼마나 가르나: AUC = 맞은 낱말 하나와 틀린 낱말 하나를 뽑았을 때 맞은 쪽 확률이 더 높을
    확률(0.5 = 못 가름, 1 = 완벽). 겹친 구간 투표(O2)가 확률에 기대므로, 이 값이 낮으면 O2 를 쓰지 않는다."""
    right, wrong = [], []
    for op, _, j in ops:
        if j is None or hyp[j].prob is None:
            continue
        (right if op == "C" else wrong).append(hyp[j].prob)
    if not right or not wrong:
        return {"auc": None, "right": len(right), "wrong": len(wrong)}
    values = np.concatenate([right, wrong])
    ranks = np.empty(len(values))
    order = values.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):  # 같은 값은 평균 순위
        same = values == v
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    u = ranks[: len(right)].sum() - len(right) * (len(right) + 1) / 2
    return {"auc": round(float(u / (len(right) * len(wrong))), 4), "right": len(right), "wrong": len(wrong),
            "mean_prob_right": round(float(np.mean(right)), 4), "mean_prob_wrong": round(float(np.mean(wrong)), 4)}


def evaluate(ref: Sequence[RefToken], hyp: Sequence[Word], *, cut_edges: Sequence[float] = (),
             seams: Sequence[float] = (), radius: float = 2.0, normalize=None) -> dict:
    """한 녹음의 점수. cut_edges = 녹음을 자른 모든 청크 가장자리, seams = 결과를 이어 붙인 자리."""
    norm = normalize or (lambda s: [s])
    ref_t: list[RefToken] = []
    for r in ref:
        for part in norm(r.text):
            ref_t.append(RefToken(part, r.start, r.end, r.tags))
    hyp_w: list[Word] = []
    for w in hyp:
        for part in norm(w.text):
            hyp_w.append(Word(part, w.start, w.end, w.prob, w.chunk, w.agreed))
    ops = align([r.text for r in ref_t], [w.text for w in hyp_w])
    times = op_times(ops, ref_t, hyp_w)
    at_cut = near(times, cut_edges, radius)
    at_seam = near(times, seams, radius)
    return {
        "overall": error_rate([r.text for r in ref_t], [w.text for w in hyp_w], ops),
        "near_cut_edges": region_error(ops, times, at_cut),
        "away_from_cut_edges": region_error(ops, times, ~at_cut),
        "near_seams": region_error(ops, times, at_seam),
        "tagged": tagged_error(ops, ref_t),
        "hallucination": hallucinations(ops, len(ref_t)),
        "confidence": confidence(ops, hyp_w),
    }

"""검수 0층: 낱말마다 '틀렸을 위험' 점수 — 사람이(또는 다음 층이) 어디부터 볼지 정한다.

단어 확률 하나로는 맞은 낱말과 틀린 낱말을 잘 못 가른다(실측 AUC 0.72~0.82, reports/real_results.md ③).
여기서는 받아쓰기를 다시 돌리지 않고 이미 있는 것만으로 신호를 더 모은다.

| 특징 | 뜻 |
|---|---|
| risk_prob · risk_prob_nb | 1 − 단어 확률, 앞뒤 1낱말 중 가장 낮은 확률의 1 − |
| disagree · no_second | O1 은 모든 지점을 창 두 개가 받아 적는다. 쓰지 않은 쪽 창이 같은 자리에 같은 낱말을 적었나 |
| alt_disagree | (선택) 다른 모델 받아쓰기가 같은 자리에 같은 낱말을 적었나 — 같은 모델 두 창보다 다양성이 크다 |
| has_digit · capital_mid · all_caps · has_latin | 숫자 · 문장 가운데 대문자(고유명사 후보) · 약어 · 한국어 글 속 영문 |
| near_glossary · in_glossary | 용어집 말과 비슷하지만 같지는 않음(오인식된 이름 후보) · 용어집 말 그대로 |
| n_chars · log_dur_per_char · repeat | 글자 수 · 글자당 길이(너무 짧거나 길면 의심) · 앞 낱말과 같은 낱말 |

라벨은 정답과 정렬해 그 낱말이 바뀜(S) · 끼어듦(I)이면 1. 빠진 낱말(D)은 받아쓰기에 자리가 없어 이 점수로 직접 잡지
못한다(빠짐 비율은 따로 보고). 점수 모델은 표준화한 특징의 L2 로지스틱 회귀(numpy 뉴턴법)이고, 통화 하나씩 빼고
학습해 그 통화를 점수 매긴다(leave-one-call-out) — 같은 통화로 학습하고 시험하면 숫자가 부풀기 때문이다.
"""

from __future__ import annotations

import bisect
import difflib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .align import align
from .chunking import Chunk
from .words import RefToken, Word

FEATURES = (
    "risk_prob", "risk_prob_nb", "disagree", "no_second", "alt_disagree", "has_digit", "capital_mid", "all_caps",
    "has_latin", "near_glossary", "in_glossary", "n_chars", "log_dur_per_char", "repeat",
)
SAME_WORD_SEC = 1.0  # 다른 창 · 다른 모델이 같은 낱말을 이 시각 차 안에 적었으면 '같은 자리에 같은 낱말'
NEAR_GLOSSARY = 0.7  # 용어집 말과 글자 유사도가 이 이상이고 1 미만이면 near_glossary

_DIGIT = re.compile(r"\d")
_LATIN = re.compile(r"[A-Za-z]")
_SENT_END = re.compile(r"[.?!]$")


def key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


# ---------- 다른 창 · 다른 모델이 같게 적었나 ----------

class _Index:
    """낱말들을 가운데 시각으로 찾기."""

    def __init__(self, words: Sequence[Word]):
        self.words = sorted(words, key=lambda w: w.mid)
        self.mids = [w.mid for w in self.words]

    def has(self, k: str, t: float, tol: float) -> bool:
        lo = bisect.bisect_left(self.mids, t - tol)
        hi = bisect.bisect_right(self.mids, t + tol)
        return any(key(self.words[i].text) == k for i in range(lo, hi))


def second_window_support(words: Sequence[Word], chunk_results: Sequence[tuple[Chunk, Sequence[Word]]],
                          tol: float = SAME_WORD_SEC) -> list[bool | None]:
    """결과 낱말마다: 그 시각을 함께 받아 적은 다른 창(들)이 같은 낱말을 적었나. 다른 창이 없으면 None.

    O1 은 녹음 처음 · 끝 반 창을 빼면 모든 지점이 창 두 개에 들어간다. 결과에는 그 지점이 창 가운데에 있는 쪽만
    쓰므로, 쓰지 않은 쪽은 공짜 두 번째 의견이다(같은 모델이라 다양성은 작다 — O2 투표가 이득이 없던 이유).
    """
    indexes = {c.index: (c, _Index(ws)) for c, ws in chunk_results if not c.skip}
    out: list[bool | None] = []
    for w in words:
        others = [ix for idx, (c, ix) in indexes.items() if idx != w.chunk and c.start <= w.mid < c.end]
        if not others:
            out.append(None)
            continue
        k = key(w.text)
        out.append(all(ix.has(k, w.mid, tol) for ix in others))
    return out


def alt_support(words: Sequence[Word], alt: Sequence[Word] | None, tol: float = SAME_WORD_SEC) -> list[bool | None]:
    """다른 모델 받아쓰기(alt)가 같은 자리에 같은 낱말을 적었나."""
    if alt is None:
        return [None] * len(words)
    ix = _Index(alt)
    return [ix.has(key(w.text), w.mid, tol) for w in words]


# ---------- 특징 ----------

def _glossary_tokens(glossary: Sequence[str]) -> list[str]:
    return sorted({key(t) for term in glossary for t in term.split() if len(key(t)) >= 3})


def _best_ratio(k: str, vocab: Sequence[str]) -> float:
    if len(k) < 3 or not vocab:
        return 0.0
    best = 0.0
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(k)
    for v in vocab:
        if not 0.5 <= len(v) / len(k) <= 2.0:
            continue
        sm.set_seq1(v)
        if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
            continue
        best = max(best, sm.ratio())
    return best


def features(words: Sequence[Word], *, support: Sequence[bool | None] | None = None,
             alt: Sequence[bool | None] | None = None, glossary: Sequence[str] = ()) -> np.ndarray:
    """(낱말 수 × len(FEATURES)) 특징 행렬. support = second_window_support, alt = alt_support 결과."""
    n = len(words)
    support = support if support is not None else [None] * n
    alt = alt if alt is not None else [None] * n
    vocab = _glossary_tokens(glossary)
    vocab_set = set(vocab)
    probs = np.array([0.5 if w.prob is None else float(w.prob) for w in words])
    x = np.zeros((n, len(FEATURES)))
    col = {name: k for k, name in enumerate(FEATURES)}
    for i, w in enumerate(words):
        k = key(w.text)
        prev = words[i - 1] if i else None
        nb = probs[max(0, i - 1): i + 2]
        x[i, col["risk_prob"]] = 1 - probs[i]
        x[i, col["risk_prob_nb"]] = 1 - nb.min()
        x[i, col["disagree"]] = support[i] is False
        x[i, col["no_second"]] = support[i] is None
        x[i, col["alt_disagree"]] = alt[i] is False
        x[i, col["has_digit"]] = bool(_DIGIT.search(w.text))
        stripped = w.text.strip("\"'([")
        sentence_start = prev is None or bool(_SENT_END.search(prev.text))
        x[i, col["capital_mid"]] = bool(stripped[:1].isupper()) and not sentence_start and stripped != "I"
        x[i, col["all_caps"]] = len(k) >= 2 and k.isalpha() and stripped.upper() == stripped and k.isascii()
        x[i, col["has_latin"]] = bool(_LATIN.search(w.text))
        if vocab:
            exact = k in vocab_set
            x[i, col["in_glossary"]] = exact
            x[i, col["near_glossary"]] = (not exact) and _best_ratio(k, vocab) >= NEAR_GLOSSARY
        x[i, col["n_chars"]] = len(k)
        x[i, col["log_dur_per_char"]] = math.log(max(w.end - w.start, 0.01) / max(1, len(k)))
        x[i, col["repeat"]] = prev is not None and key(prev.text) == k and bool(k)
    return x


# ---------- 라벨 ----------

@dataclass
class Labels:
    wrong: np.ndarray  # 낱말마다 1(바꿈 · 끼어듦) · 0(맞음) · -1(정규화하면 사라지는 낱말 — 평가에서 뺀다)
    tagged: np.ndarray  # 그 낱말에 정렬된 정답 낱말이 고유명사 · 숫자 꼬리표를 가졌나
    deletions: int  # 받아쓰기에 자리가 없는 빠진 정답 낱말 수(이 점수로 직접 못 잡음)
    ref_words: int


def word_labels(ref: Sequence[RefToken], words: Sequence[Word], normalize: Callable[[str], list[str]],
                tag_set: frozenset[str] | set[str] = frozenset()) -> Labels:
    ref_t: list[tuple[str, tuple[str, ...]]] = [(p, r.tags) for r in ref for p in normalize(r.text)]
    hyp_t: list[str] = []
    owner: list[int] = []
    for i, w in enumerate(words):
        for p in normalize(w.text):
            hyp_t.append(p)
            owner.append(i)
    wrong = np.full(len(words), -1, dtype=np.int64)
    for i in set(owner):
        wrong[i] = 0
    tagged = np.zeros(len(words), dtype=bool)
    dels = 0
    for op, i, j in align([t for t, _ in ref_t], hyp_t):
        if j is None:
            dels += 1
            continue
        w = owner[j]
        if op in ("S", "I"):
            wrong[w] = 1
        if i is not None and set(ref_t[i][1]) & set(tag_set):
            tagged[w] = True
    return Labels(wrong, tagged, dels, len(ref_t))


# ---------- 점수 모델 ----------

@dataclass
class RiskModel:
    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    extra: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, names: Sequence[str] = FEATURES, l2: float = 1.0,
            iters: int = 30) -> RiskModel:
        """L2 로지스틱 회귀(뉴턴법). 표준편차가 0 인 특징(그 자료에 없는 신호)은 0 으로 둔다."""
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        z = np.hstack([np.ones((len(x), 1)), (x - mean) / scale])
        beta = np.zeros(z.shape[1])
        reg = np.full(z.shape[1], l2)
        reg[0] = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(z @ beta, -30, 30)))
            grad = z.T @ (p - y) + reg * beta
            hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(reg) + 1e-9 * np.eye(z.shape[1])
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.abs(step).max() < 1e-7:
                break
        return cls(tuple(names), mean, scale, beta[1:], float(beta[0]))

    def score(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(((x - self.mean) / self.scale) @ self.coef + self.intercept, -30, 30)))

    def weights(self) -> dict[str, float]:
        """표준화한 특징 기준 가중치 — 어느 신호가 위험을 올리나."""
        return {n: round(float(c), 4) for n, c in zip(self.names, self.coef, strict=True)}

    def to_dict(self) -> dict:
        return {"names": list(self.names), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> RiskModel:
        return cls(tuple(d["names"]), np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]),
                   float(d["intercept"]), d.get("extra", {}))


# ---------- 평가 ----------

def rank_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """틀린 낱말 하나와 맞은 낱말 하나를 뽑았을 때 틀린 쪽 점수가 더 높을 확률(같으면 반)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2  # 같은 값은 평균 순위(1부터)
    ranks[order] = np.repeat(avg, counts)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return round(float(u / (n_pos * n_neg)), 4)


def catch_at_word_budget(y: np.ndarray, s: np.ndarray, fractions: Sequence[float] = (0.05, 0.1, 0.2)) -> dict:
    """위험 순으로 낱말 상위 f 만 볼 때 틀린 낱말 중 잡히는 비율."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(s), kind="mergesort")
    total = int(y.sum())
    out = {}
    for f in fractions:
        k = int(math.ceil(f * len(y)))
        out[f"{f:g}"] = round(float(y[order[:k]].sum() / total), 4) if total else None
    return out


def review_seconds(words: Sequence[Word], picked: np.ndarray, pad: float = 1.0, join: float = 2.0) -> float:
    """고른 낱말들을 사람이 들어 볼 구간(앞뒤 pad 초, join 초 안으로 붙은 것은 한 구간)의 총 길이."""
    spans = sorted((max(0.0, words[i].start - pad), words[i].end + pad) for i in np.flatnonzero(picked))
    total, cur = 0.0, None
    for a, b in spans:
        if cur and a <= cur[1] + join:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                total += cur[1] - cur[0]
            cur = (a, b)
    if cur:
        total += cur[1] - cur[0]
    return total


def catch_at_time_budget(words: Sequence[Word], y: np.ndarray, s: np.ndarray, duration: float,
                         fractions: Sequence[float] = (0.1, 0.2, 0.3), pad: float = 1.0, join: float = 2.0) -> dict:
    """사람이 녹음 길이의 f 만큼만 들을 수 있을 때(위험 높은 낱말 주변부터) 틀린 낱말 중 잡히는 비율.

    검수 시간이 곧 비용이라, '상위 몇 % 낱말' 보다 이 숫자가 실제 쓰임에 가깝다.
    """
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(s), kind="mergesort")
    total = int(y.sum())
    out = {}
    for f in fractions:
        lo, hi = 0, len(order)
        while lo < hi:  # 듣는 시간이 f · duration 을 넘지 않는 가장 많은 낱말 수
            mid = (lo + hi + 1) // 2
            picked = np.zeros(len(words), dtype=bool)
            picked[order[:mid]] = True
            if review_seconds(words, picked, pad, join) <= f * duration:
                lo = mid
            else:
                hi = mid - 1
        out[f"{f:g}"] = round(float(y[order[:lo]].sum() / total), 4) if total else None
    return out


@dataclass
class CallData:
    call_id: str
    words: list[Word]
    x: np.ndarray
    labels: Labels
    duration: float

    @property
    def mask(self) -> np.ndarray:
        return self.labels.wrong >= 0


def leave_one_call_out(calls: Sequence[CallData], columns: Sequence[str] = FEATURES, l2: float = 1.0
                       ) -> dict[str, np.ndarray]:
    """통화마다: 나머지 통화로 학습한 모델의 점수. columns 로 쓸 특징을 고른다(비교용)."""
    idx = [FEATURES.index(c) for c in columns]
    out = {}
    for held in calls:
        train = [c for c in calls if c.call_id != held.call_id]
        if not train:
            raise ValueError("통화가 둘 이상이어야 한다")
        x = np.vstack([c.x[c.mask][:, idx] for c in train])
        y = np.concatenate([c.labels.wrong[c.mask] for c in train]).astype(float)
        model = RiskModel.fit(x, y, columns, l2=l2)
        out[held.call_id] = model.score(held.x[:, idx])
    return out


def summarize(calls: Sequence[CallData], scores: dict[str, np.ndarray]) -> dict:
    """모은 점수로 AUC · 낱말 예산 · 시간 예산 · 고유명사/숫자 오류 잡는 비율."""
    y = np.concatenate([c.labels.wrong[c.mask] for c in calls])
    s = np.concatenate([scores[c.call_id][c.mask] for c in calls])
    tagged = np.concatenate([c.labels.tagged[c.mask] for c in calls])
    time_budget: dict[str, list[float]] = {}
    tagged_time: dict[str, list[float]] = {}
    weights, tagged_weights = [], []
    for c in calls:
        m = c.mask
        ws = [w for w, keep in zip(c.words, m, strict=True) if keep]
        yy, ss = c.labels.wrong[m], scores[c.call_id][m]
        tg = c.labels.tagged[m]
        weights.append(int(yy.sum()))
        for f, v in catch_at_time_budget(ws, yy, ss, c.duration).items():
            time_budget.setdefault(f, []).append(v if v is not None else 0.0)
        yt = yy.astype(bool) & tg  # 고유명사 · 숫자 오류만
        tagged_weights.append(int(yt.sum()))
        for f, v in catch_at_time_budget(ws, yt, ss, c.duration).items():
            tagged_time.setdefault(f, []).append(v if v is not None else 0.0)
    w = np.array(weights, dtype=float)
    wt = np.array(tagged_weights, dtype=float)
    return {
        "words": int(len(y)), "errors": int(y.sum()),
        "deletions": int(sum(c.labels.deletions for c in calls)),
        "auc": rank_auc(y, s),
        "auc_per_call": {c.call_id: rank_auc(c.labels.wrong[c.mask], scores[c.call_id][c.mask]) for c in calls},
        "catch_at_word_budget": catch_at_word_budget(y, s),
        # 통화마다 '녹음 길이의 f 만 듣기' 로 잡은 비율을 그 통화 오류 수로 가중 평균
        "catch_at_time_budget": {f: round(float(np.average(v, weights=w)), 4) if w.sum() else None
                                 for f, v in time_budget.items()},
        "tagged_catch_at_time_budget": {f: round(float(np.average(v, weights=wt)), 4) if wt.sum() else None
                                        for f, v in tagged_time.items()},
        "tagged_errors": int((y.astype(bool) & tagged).sum()),
    }

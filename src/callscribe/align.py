"""정답과 받아쓰기를 낱말(또는 글자) 단위로 정렬한다: 맞음(C) · 바꿈(S) · 빠짐(D) · 끼어듦(I).

어닝콜 한 건은 정답이 1만 낱말 안팎이라 표 전체(1만 × 1만)를 잡으면 메모리가 모자라다. 그래서
- 한 줄(정답 낱말 하나)씩 numpy 로 계산하고(끼어듦은 '누적 최솟값' 으로 한 번에),
- 긴 글은 대각선 근처 띠(band)만 계산하며, 고른 경로가 띠 가장자리에 닿으면 띠를 두 배로 넓혀 다시 한다
  (띠 밖에 더 나은 경로가 있을 수 있다는 뜻이라서).
작은 입력에서는 띠 없이 전체를 계산하고, 테스트에서 단순한 편집 거리 계산과 결과를 맞춘다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FULL_LIMIT = 4_000_000  # 정답 × 받아쓰기 칸 수가 이보다 작으면 띠 없이 전체 계산
DEFAULT_BAND = 1500
_INF = np.int64(1 << 40)

Op = tuple[str, int | None, int | None]  # (종류, 정답 위치, 받아쓰기 위치)


def align(ref: Sequence[str], hyp: Sequence[str], band: int | None = None) -> list[Op]:
    """최소 편집 정렬. 같은 비용이면 맞음/바꿈 → 빠짐 → 끼어듦 순으로 고른다."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return [("I", None, j) for j in range(m)]
    if m == 0:
        return [("D", i, None) for i in range(n)]
    vocab: dict[str, int] = {}
    r = np.fromiter((vocab.setdefault(t, len(vocab)) for t in ref), dtype=np.int64, count=n)
    h = np.fromiter((vocab.setdefault(t, len(vocab)) for t in hyp), dtype=np.int64, count=m)
    w = band if band is not None else (None if n * m <= FULL_LIMIT else DEFAULT_BAND)
    while True:
        ops = _dp(r, h, w)
        if ops is not None:
            return ops
        w = w * 2  # 경로가 띠 가장자리에 닿았다 → 넓혀 다시


def edit_counts(ops: Sequence[Op]) -> dict[str, int]:
    out = {"C": 0, "S": 0, "D": 0, "I": 0}
    for op, _, _ in ops:
        out[op] += 1
    return out


def _dp(r: np.ndarray, h: np.ndarray, w: int | None) -> list[Op] | None:
    n, m = len(r), len(h)
    if w is not None:
        w = max(w, int(np.ceil(m / n)) + 2, int(np.ceil(n / m)) + 2)

    def bounds(i: int) -> tuple[int, int]:
        if w is None:
            return 0, m
        c = i * m / n
        return max(0, int(c - w)), min(m, int(c + w) + 1)

    lo, hi = bounds(0)
    prev, prev_lo, prev_hi = np.arange(lo, hi + 1, dtype=np.int64), lo, hi
    bps: list[np.ndarray] = [np.full(hi - lo + 1, 2, dtype=np.uint8)]
    los, his = [lo], [hi]
    for i in range(1, n + 1):
        lo, hi = bounds(i)
        cols = np.arange(lo, hi + 1)
        up = _take(prev, prev_lo, prev_hi, cols) + 1
        diag = _take(prev, prev_lo, prev_hi, cols - 1)
        valid = cols >= 1
        diag[valid] += (h[cols[valid] - 1] != r[i - 1])
        tmp = np.minimum(diag, up)
        choice = np.where(diag <= up, 0, 1).astype(np.uint8)
        k = np.arange(len(cols), dtype=np.int64)
        cur = k + np.minimum.accumulate(tmp - k)  # 같은 줄에서 왼쪽으로부터 끼어듦
        bp = np.where(cur < tmp, 2, choice).astype(np.uint8)
        bps.append(bp)
        los.append(lo)
        his.append(hi)
        prev, prev_lo, prev_hi = cur, lo, hi

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if w is not None and i > 0 and ((j == los[i] and los[i] > 0) or (j == his[i] and his[i] < m)):
            return None
        if i == 0:
            ops.append(("I", None, j - 1))
            j -= 1
            continue
        b = bps[i][j - los[i]]
        if b == 0:
            ops.append(("C" if r[i - 1] == h[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def _take(row: np.ndarray, lo: int, hi: int, cols: np.ndarray) -> np.ndarray:
    out = np.full(len(cols), _INF, dtype=np.int64)
    ok = (cols >= lo) & (cols <= hi)
    out[ok] = row[cols[ok] - lo]
    return out

"""녹음을 어떻게 자를까: 청크 계획.

청크마다 두 구간이 있다.
- [start, end)       모델에게 줄 소리 구간
- [own_start, own_end)  결과에 반영할 구간. 모든 청크의 own 구간을 이으면 녹음 전체를 **정확히 한 번** 덮는다
                        (불변조건 1 — 겹친 구간이 두 번 들어가거나 경계 구간이 빠지면 안 됨, check_partition).

| 이름 | 계획 |
|---|---|
| B0 | fixed: 60초 고정, 겹침 없음(지금 wspr) |
| B1 | single: 통째로 모델에 맡김(모델 자체의 긴 오디오 처리) |
| O1 · O2 | overlapped: 창을 절반씩 겹쳐(1+2, 2+3 …). own 경계는 겹친 구간의 가운데 |
| V1 | vad: 2초 넘는 무음은 받아 적지 않고(skip) 비워 두며, 말소리 구간은 조용한 지점에서 30초 안으로 자름 |
| V2 | vad + pad: V1 의 각 청크를 양옆으로 조금 넓혀 받아 적고, own 은 V1 구간 그대로 |
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float
    own_start: float
    own_end: float
    skip: bool = False  # 무음이라 받아 적지 않는 구간(결과는 비어 있다 — 대기음에서 지어낸 말을 막는다)

    @property
    def edges(self) -> tuple[float, float]:
        return self.start, self.end


def single(duration: float) -> list[Chunk]:
    return [Chunk(0, 0.0, duration, 0.0, duration)]


def fixed(duration: float, size: float = 60.0) -> list[Chunk]:
    n = max(1, math.ceil(duration / size - 1e-9))
    return [Chunk(k, k * size, min(duration, (k + 1) * size), k * size, min(duration, (k + 1) * size))
            for k in range(n)]


def overlapped(duration: float, size: float = 30.0, hop: float | None = None) -> list[Chunk]:
    """창 길이 size, 간격 hop(기본 size/2 → 모든 지점이 두 창에 들어감). 마지막 창은 녹음 끝에 맞춰 당긴다."""
    hop = size / 2 if hop is None else hop
    if not 0 < hop <= size:
        raise ValueError("0 < hop <= size")
    if duration <= size:
        return single(duration)
    starts = []
    s = 0.0
    while s + size < duration - 1e-9:
        starts.append(s)
        s += hop
    starts.append(max(0.0, duration - size))
    starts = sorted(set(round(x, 6) for x in starts))
    ends = [min(duration, x + size) for x in starts]
    chunks = []
    for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
        own_a = 0.0 if k == 0 else (a + ends[k - 1]) / 2
        own_b = duration if k == len(starts) - 1 else (starts[k + 1] + b) / 2
        chunks.append(Chunk(k, a, b, own_a, own_b))
    return chunks


def frame_energy(audio: np.ndarray, sr: int, frame: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """프레임별 소리 크기(dBFS). 모델 없이 쓰는 단순한 무음 찾기용(내려받는 VAD 모델 없이)."""
    n = max(1, int(sr * frame))
    k = len(audio) // n
    x = audio[: k * n].astype(np.float64).reshape(k, n)
    rms = np.sqrt(np.mean(x * x, axis=1)) + 1e-10
    return (np.arange(k) + 0.5) * frame, 20 * np.log10(rms)


MIN_PIECE = 0.5  # 이보다 짧은 말소리 조각은 따로 받아 적지 않는다
QUIET_DB = 3.0  # 고를 수 있는 구간에서 가장 조용한 값보다 이만큼 안쪽이면 '쉼' 으로 본다


def _best_pause(times: np.ndarray, db: np.ndarray, sel: np.ndarray) -> float:
    """고를 수 있는 프레임(sel) 안에서 가장 긴 쉼의 가운데. 길이가 같으면 가장 늦은 쉼(청크를 max_len 가까이 채움)."""
    seg = db[sel]
    quiet = np.concatenate([[False], seg <= seg.min() + QUIET_DB, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    runs = list(zip(edges[::2], edges[1::2], strict=True))  # [시작, 끝) — sel 안 위치
    start, end = max(runs, key=lambda r: (r[1] - r[0], r[0]))
    return float(times[sel[(start + end - 1) // 2]])


def silent_spans(times: np.ndarray, db: np.ndarray, silence_db: float = -50.0, min_silence: float = 2.0,
                 margin: float = 0.2) -> list[tuple[float, float]]:
    """min_silence 초 넘게 silence_db 보다 조용한 구간. 앞뒤 margin 은 말소리 쪽에 남긴다."""
    if len(times) == 0:
        return []
    frame = float(times[1] - times[0]) if len(times) > 1 else 0.03
    quiet = np.concatenate([[False], db < silence_db, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    spans = []
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        s, e = times[a] - frame / 2 + margin, times[b - 1] + frame / 2 - margin
        if e - s >= min_silence - 2 * margin:
            spans.append((float(s), float(e)))
    return spans


def vad(duration: float, times: np.ndarray, db: np.ndarray, max_len: float = 30.0, min_len: float = 5.0,
        pad: float = 0.0, silence_db: float = -50.0, min_silence: float = 2.0) -> list[Chunk]:
    """긴 무음은 skip 청크로 비워 두고, 말소리 구간은 [min_len, max_len] 뒤 사이의 가장 조용한 프레임에서 자른다.

    pad > 0 이면(V2) 받아 적는 구간만 양옆으로 넓히고(무음 쪽으로는 넓히지 않음), own 구간은 자른 구간 그대로 둔다.
    """
    pieces: list[tuple[float, float, bool]] = []
    pos = 0.0
    for s, e in [*silent_spans(times, db, silence_db, min_silence), (duration, duration)]:
        s, e = max(s, pos), min(e, duration)
        if 0 < s - pos < MIN_PIECE and e > s and (not pieces or pieces[-1][2]):
            s = pos  # 무음 앞 아주 짧은 조각은 무음에 붙인다
        if s - pos >= MIN_PIECE or (s > pos and e <= s):  # 말소리 구간을 max_len 안으로
            a = pos
            while s - a > max_len:
                sel = np.nonzero((times >= a + min_len) & (times <= a + max_len))[0]
                cut = _best_pause(times, db, sel) if len(sel) else a + max_len
                pieces.append((a, cut, False))
                a = cut
            pieces.append((a, s, False))
        elif s > pos and pieces and pieces[-1][2]:
            pieces[-1] = (pieces[-1][0], s, True)
        if e > s:
            pieces.append((s, e, True))
        pos = max(pos, e)
    chunks = []
    for k, (a, b, skip) in enumerate(pieces):
        prev_skip = k > 0 and pieces[k - 1][2]
        next_skip = k + 1 < len(pieces) and pieces[k + 1][2]
        lo = a if (skip or prev_skip) else max(0.0, a - pad)
        hi = b if (skip or next_skip) else min(duration, b + pad)
        chunks.append(Chunk(k, lo, hi, a, b, skip))
    return chunks


def check_partition(chunks: list[Chunk], duration: float, tol: float = 1e-6) -> list[str]:
    """불변조건 1: own 구간이 [0, duration) 을 빈틈 · 겹침 없이 나누고, 각 own 은 받아 적는 구간 안에 있다."""
    problems = []
    if not chunks:
        return ["청크 없음"]
    if abs(chunks[0].own_start) > tol:
        problems.append(f"처음이 0 이 아님: {chunks[0].own_start}")
    if abs(chunks[-1].own_end - duration) > tol:
        problems.append(f"끝이 녹음 길이와 다름: {chunks[-1].own_end} ≠ {duration}")
    for a, b in zip(chunks, chunks[1:], strict=False):
        if abs(a.own_end - b.own_start) > tol:
            problems.append(f"청크 {a.index}·{b.index} 사이 {'겹침' if a.own_end > b.own_start else '빈틈'}")
    for c in chunks:
        if c.own_start < c.start - tol or c.own_end > c.end + tol or c.own_end <= c.own_start:
            problems.append(f"청크 {c.index} own 구간이 받아 적는 구간 밖")
    return problems


def cut_edges(chunks: list[Chunk], duration: float) -> list[float]:
    """녹음을 자른 모든 지점(녹음 처음 · 끝 제외)."""
    pts = {round(x, 6) for c in chunks for x in c.edges}
    return sorted(p for p in pts if 1e-6 < p < duration - 1e-6)


def seams(chunks: list[Chunk]) -> list[float]:
    """결과를 이어 붙인 지점(own 경계)."""
    return [c.own_end for c in chunks[:-1]]

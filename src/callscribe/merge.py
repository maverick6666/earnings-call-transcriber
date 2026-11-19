"""청크 결과 이어 붙이기.

- merge_own (B0 · V1 · O1 · V2): 청크마다 자기 own 구간 안(낱말 가운데 시각 기준) 낱말만 쓴다.
  겹쳐 자른 경우(O1) own 경계가 겹친 구간 가운데라, 한 창의 가장자리에서 잘린 낱말은 다른 창의 가운데에서 받아 적힌
  것으로 채워진다 — 사용자 아이디어(1+2, 2+3)의 기본형.
- merge_vote (O2): 두 창이 모두 받아 적은 구간을 정렬해 고른다
  (Whisper-Streaming 의 LocalAgreement · ROVER 와 같은 계열).
    두 창이 같게 적은 낱말     → 확정(agreed=True)
    서로 다르게 적은 낱말     → 단어 확률이 높은 쪽, 같으면 그 시각이 창 가운데에 더 가까운 쪽
    한쪽만 적은 낱말          → 그 시각이 그 창의 가장자리에서 더 먼(=그 창이 더 믿을 만한) 경우만 남긴다
- 두 청크가 같은 낱말을 이음매 양쪽에 하나씩 넣는 경우를 막으려고, 이음매 근처의 같은 낱말 중복은 하나만 남긴다.
"""

from __future__ import annotations

from collections.abc import Sequence

from .align import align
from .chunking import Chunk
from .words import Word

ChunkResult = tuple[Chunk, Sequence[Word]]
SEAM_DUP_TOL = 0.3  # 이음매 양쪽 같은 낱말의 가운데 시각 차가 이보다 작으면 중복으로 본다


def _key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _tag(w: Word, chunk: Chunk, agreed: bool | None = None) -> Word:
    return Word(w.text, w.start, w.end, w.prob, chunk.index, agreed if agreed is not None else w.agreed)


def merge_own(results: Sequence[ChunkResult]) -> list[Word]:
    out: list[Word] = []
    last = len(results) - 1
    for k, (chunk, words) in enumerate(results):
        for w in words:
            if chunk.own_start <= w.mid < chunk.own_end or (k == last and w.mid >= chunk.own_end):
                out.append(_tag(w, chunk))
    return dedupe_seams(sorted(out, key=lambda w: (w.start, w.end)), [c.own_end for c, _ in results[:-1]])


def dedupe_seams(words: list[Word], seam_points: Sequence[float], tol: float = SEAM_DUP_TOL) -> list[Word]:
    """이음매 근처에서 서로 다른 청크가 같은 낱말을 하나씩 넣었으면 하나만 남긴다(확률이 높은 쪽)."""
    if not seam_points:
        return words
    out: list[Word] = []
    for w in words:
        if out:
            p = out[-1]
            same = _key(p.text) == _key(w.text) and p.chunk != w.chunk and abs(p.mid - w.mid) < tol
            if same and any(abs(s - w.mid) < 2.0 for s in seam_points):
                if (w.prob or 0) > (p.prob or 0):
                    out[-1] = w
                continue
        out.append(w)
    return out


def _edge_distance(chunk: Chunk, t: float) -> float:
    return min(t - chunk.start, chunk.end - t)


def merge_vote(results: Sequence[ChunkResult]) -> list[Word]:
    """겹친 구간을 두 창의 결과로 골라 잇는다. 구간 = 창 가장자리들로 나눈 조각, 조각마다 덮는 창은 하나 또는 둘."""
    chunks = [c for c, _ in results]
    words_of = {c.index: list(ws) for c, ws in results}
    pts = sorted({x for c in chunks for x in c.edges})
    out: list[Word] = []
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        cover = [c for c in chunks if c.start <= a + 1e-9 and c.end >= b - 1e-9]
        if not cover:
            continue
        # 셋 이상이 덮으면 조각 가운데가 가장 안쪽인 두 창만
        mid = (a + b) / 2
        cover = sorted(cover, key=lambda c: -_edge_distance(c, mid))[:2]
        pieces = [[w for w in words_of[c.index] if a <= w.mid < b or (b == pts[-1] and w.mid >= b)] for c in cover]
        if len(cover) == 1:
            out.extend(_tag(w, cover[0]) for w in pieces[0])
        else:
            out.extend(_vote(cover[0], pieces[0], cover[1], pieces[1]))
    out.sort(key=lambda w: (w.start, w.end))
    return dedupe_seams(out, pts[1:-1])


def _vote(ca: Chunk, wa: list[Word], cb: Chunk, wb: list[Word]) -> list[Word]:
    ops = align([_key(w.text) for w in wa], [_key(w.text) for w in wb])
    out: list[Word] = []

    def prob(w: Word) -> float:
        return 0.5 if w.prob is None else w.prob

    for op, i, j in ops:
        if op == "C":
            x, y = wa[i], wb[j]
            best = x if prob(x) >= prob(y) else y
            out.append(Word(best.text, (x.start + y.start) / 2, (x.end + y.end) / 2, max(prob(x), prob(y)),
                            ca.index if best is x else cb.index, True))
        elif op == "S":
            x, y = wa[i], wb[j]
            if prob(x) != prob(y):
                pick, chunk = (x, ca) if prob(x) > prob(y) else (y, cb)
            else:
                pick, chunk = (x, ca) if _edge_distance(ca, x.mid) >= _edge_distance(cb, y.mid) else (y, cb)
            out.append(_tag(pick, chunk, agreed=False))
        elif op == "D":  # A 만 적음
            x = wa[i]
            if _edge_distance(ca, x.mid) > _edge_distance(cb, x.mid):
                out.append(_tag(x, ca, agreed=False))
        else:  # B 만 적음
            y = wb[j]
            if _edge_distance(cb, y.mid) > _edge_distance(ca, y.mid):
                out.append(_tag(y, cb, agreed=False))
    return out

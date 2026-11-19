"""청크 결과 이어 붙이기.

- merge_own (B0 · V1 · O1 · V2): 청크마다 자기 own 구간 안(낱말 가운데 시각 기준) 낱말만 쓴다.
  겹쳐 자른 경우(O1) own 경계가 겹친 구간 가운데라, 한 창의 가장자리에서 잘린 낱말은 다른 창의 가운데에서 받아 적힌
  것으로 채워진다 — 사용자 아이디어(1+2, 2+3)의 기본형.
- merge_vote (O2): 창을 차례로 이으며, 두 창이 모두 받아 적은 겹친 구간을 낱말로 정렬해 고른다
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


VOTE_MARGIN = 2.0  # 겹친 구간 양옆으로 이만큼 더 넓혀 정렬한다(두 창이 같은 낱말에 매긴 시각이 어긋나도 잡히게)
SAME_PLACE_SEC = 1.0  # 글이 다른 두 낱말(바꿈)은 가운데 시각 차가 이 안이어야 같은 소리의 두 후보로 보고 고른다
SAME_WORD_SEC = 3.0  # 글이 같은 두 낱말(맞음)은 강한 증거라 두 창의 시각 차를 이만큼까지 허용한다


def merge_vote(results: Sequence[ChunkResult]) -> list[Word]:
    """창을 차례로 잇는다: 지금까지 결과의 꼬리(다음 창과 겹친 구간)와 다음 창의 머리를 낱말로 정렬해 고른다.

    두 창은 같은 낱말에 몇 초씩 다른 시각을 매기기도 한다(실제 모델에서 확인). 그래서 시각으로 조각을 나눠 따로 고르면
    같은 문장이 두 번 들어가거나 순서가 섞인다. 겹친 구간 전체를 글로 정렬하고, 결과는 정렬 순서 그대로 둔다.
    """
    ordered = sorted(results, key=lambda r: r[0].start)
    chunk_of = {c.index: c for c, _ in ordered}
    out = [_tag(w, ordered[0][0]) for w in ordered[0][1]]
    for (prev, _), (cur, words) in zip(ordered[:-1], ordered[1:], strict=False):
        lo, hi = cur.start, prev.end
        cur_words = [_tag(w, cur) for w in words]
        if hi <= lo:  # 겹치지 않으면 그냥 잇는다
            out.extend(cur_words)
            continue
        cut = next((k for k, w in enumerate(out) if w.mid >= lo - VOTE_MARGIN), len(out))
        head = [w for w in cur_words if w.mid <= hi + VOTE_MARGIN]
        rest = [w for w in cur_words if w.mid > hi + VOTE_MARGIN]
        out = out[:cut] + _vote(out[cut:], head, chunk_of, prev, cur) + rest
    return _monotonic(out)


def _vote(tail: list[Word], head: list[Word], chunk_of: dict[int, Chunk], prev: Chunk, cur: Chunk) -> list[Word]:
    ops = align([_key(w.text) for w in tail], [_key(w.text) for w in head])
    out: list[Word] = []

    def prob(w: Word) -> float:
        return 0.5 if w.prob is None else w.prob

    def own(w: Word) -> Chunk:
        return chunk_of.get(w.chunk, prev)

    def keep_one_side(w: Word, from_tail: bool) -> None:
        """한쪽만 적은 낱말: 그 시각에 그 창이 다른 창보다 안쪽이면 남긴다(O1 과 같은 '가운데 창을 믿기')."""
        mine, other = (own(w), cur) if from_tail else (cur, prev)
        if _edge_distance(mine, w.mid) > _edge_distance(other, w.mid):
            out.append(Word(w.text, w.start, w.end, w.prob, w.chunk, False))

    for op, i, j in ops:
        x = tail[i] if i is not None else None
        y = head[j] if j is not None else None
        # 짝지어진 두 낱말이 같은 자리(시각)에 있지 않으면 같은 소리의 두 후보가 아니다.
        # ROVER 처럼 같은 자리끼리만 고른다.
        # 두 창이 문장을 다르게 끊거나 숫자를 다르게 적을 때(시월 ↔ 10월) 엉뚱한 낱말끼리 짝지어지는 것을 막는다.
        if op in ("C", "S") and abs(x.mid - y.mid) > (SAME_WORD_SEC if op == "C" else SAME_PLACE_SEC):
            keep_one_side(x, True)
            keep_one_side(y, False)
        elif op == "C":
            best = x if prob(x) >= prob(y) else y
            out.append(Word(best.text, best.start, best.end, max(prob(x), prob(y)), best.chunk, True))
        elif op == "S":
            if prob(x) != prob(y):
                pick = x if prob(x) > prob(y) else y
            else:
                pick = x if _edge_distance(own(x), x.mid) >= _edge_distance(cur, y.mid) else y
            out.append(Word(pick.text, pick.start, pick.end, pick.prob, pick.chunk, False))
        elif op == "D":
            keep_one_side(x, True)
        else:
            keep_one_side(y, False)
    out.sort(key=lambda w: w.start)
    return out


def _monotonic(words: list[Word]) -> list[Word]:
    """정렬 순서를 지키면서 시작 시각이 거꾸로 가지 않게 다듬는다(두 창의 시각 차이 때문)."""
    out: list[Word] = []
    for w in words:
        if out and w.start < out[-1].start:
            start = out[-1].start
            w = Word(w.text, start, max(w.end, start), w.prob, w.chunk, w.agreed, w.original)
        out.append(w)
    return out

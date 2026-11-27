"""뭉개진 창 고치기 — O1R = O1(30초 창 · 15초 간격) + 이 단계.

한국어 실제 실적 콜(삼성SDS)에서 30초 창 일부가 내용을 통째로 버리고 '기상캐스터 배혜지' · '시청해주셔서 감사합니다'
같은 방송 자막 헛소리 두세 낱말만 냈다. 창마다 앞 내용 없이 따로 받아 적기 때문으로 보인다. merge_own 은 창마다 맡은
가운데 15초(own)만 쓰므로, 그 15초가 결과에서 통째로 빠진다(2025년 3분기 통화는 발표 글자의 17%).

찾기 — 창이 맡은 구간에 말소리(Silero VAD, backends.speech_energy)가 MIN_SPEECH 초 넘게 있는데, 그 구간 글자 수 ÷
    말소리 초가 이 통화 창들 가운데값의 COLLAPSE_RATIO 배보다 작으면 뭉개진 창이다.
고치기 — 아래를 차례로 시도해, 맡은 구간 글자율이 가운데값의 ACCEPT_RATIO 배 이상인 첫 결과를 쓴다.
    1) 창을 5초 뒤로 옮겨 다시  2) 앞 창 글 끝 PROMPT_CHARS 자를 힌트로  3) 창을 5초 앞으로 옮겨 다시
    4) 통화 설명(hint)을 힌트로. 모두 모자라면 맡은 구간 글자가 가장 많은 결과(원래보다 많을 때만).
    옮긴 창도 맡은 구간(가운데 15초)을 다 덮는다. 힌트는 그 창에만 준다.
실험(삼성SDS 2개 통화, 뭉개진 창 28개, B1 받아쓰기와 맡은 구간 글자 유사도 0.9 이상을 '살아남' 으로): 옮기기 23개,
    앞 창 힌트 13개, 통화 설명 10개가 살아났고 넷 중 하나로는 28개 모두 살아났다. 어느 하나도 혼자서는 모자라다.

다시 받아쓰기는 runner 가 넘겨주는 transcribe(start, end, prompt) 로 한다 — 저장본은 runner 가 맡는다(옮긴 창은 보통
청크와 같은 이름, 힌트를 준 결과는 이름에 힌트 해시를 붙임).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .chunking import Chunk
from .normalize import normalize_ko_chars
from .words import Word

MIN_SPEECH = 5.0  # 맡은 구간 말소리가 이보다 짧으면 판단하지 않는다(초)
COLLAPSE_RATIO = 0.3
ACCEPT_RATIO = 0.6
MAX_RATIO = 2.5  # 이보다 글자가 많으면 되풀이 고리(같은 말 반복)로 보고 쓰지 않는다
SHIFT = 5.0
PROMPT_CHARS = 200
NONSPEECH_DB = -70.0  # backends.speech_energy 가 말소리 아닌 곳에 덮는 값

Energy = tuple[np.ndarray, np.ndarray]
Transcribe = Callable[[float, float, str | None], list[Word]]  # (시작, 끝, 힌트) -> 낱말


def speech_seconds(energy: Energy, a: float, b: float) -> float:
    times, db = energy
    frame = float(times[1] - times[0]) if len(times) > 1 else 0.03
    return float(((times >= a) & (times < b) & (db > NONSPEECH_DB)).sum() * frame)


def own_chars(words: Sequence[Word], chunk: Chunk) -> int:
    """맡은 구간(낱말 가운데 시각 기준) 글자 수 — 띄어쓰기 · 문장부호 빼고(영어도 글자로 센다)."""
    return sum(len(normalize_ko_chars(w.text)) for w in words if chunk.own_start <= w.mid < chunk.own_end)


def _rates(results: Sequence[tuple[Chunk, Sequence[Word]]], energy: Energy) -> list[float | None]:
    out: list[float | None] = []
    for chunk, words in results:
        sec = speech_seconds(energy, chunk.own_start, chunk.own_end)
        out.append(own_chars(words, chunk) / sec if sec >= MIN_SPEECH and not chunk.skip else None)
    return out


def collapsed(results: Sequence[tuple[Chunk, Sequence[Word]]], energy: Energy) -> tuple[list[int], float]:
    """뭉개진 창 번호들과 이 통화의 글자율 가운데값(말소리 1초당 글자)."""
    rates = _rates(results, energy)
    known = [r for r in rates if r is not None]
    if not known:
        return [], 0.0
    median = float(np.median(known))
    return [k for k, r in enumerate(rates) if r is not None and r < COLLAPSE_RATIO * median], median


def repair(results: list[tuple[Chunk, list[Word]]], transcribe: Transcribe, energy: Energy, duration: float, *,
           hint: str = "", can_prompt: bool = True) -> tuple[list[tuple[Chunk, list[Word]]], list[dict]]:
    """뭉개진 창의 낱말을 바꿔 끼운다. 돌려주는 기록: 창마다 시도한 것 · 고른 것 · 글자율."""
    bad, median = collapsed(results, energy)
    out = list(results)
    log = []
    for k in bad:
        chunk, words = out[k]
        sec = speech_seconds(energy, chunk.own_start, chunk.own_end)
        prev = " ".join(w.text for w in out[k - 1][1])[-PROMPT_CHARS:] if k > 0 else ""
        attempts: list[tuple[str, float, float, str | None]] = [
            ("shift+5", chunk.start + SHIFT, min(duration, chunk.end + SHIFT), None)]
        if can_prompt and prev:
            attempts.append(("prev_text", chunk.start, chunk.end, prev))
        attempts.append(("shift-5", max(0.0, chunk.start - SHIFT), chunk.end - SHIFT, None))
        if can_prompt and hint:
            attempts.append(("call_hint", chunk.start, chunk.end, hint))
        tried = []
        best: tuple[str, list[Word], int] = ("original", words, own_chars(words, chunk))
        chosen = None
        for name, a, b, prompt in attempts:
            if not (a <= chunk.own_start and b >= chunk.own_end):
                continue  # 녹음 처음 · 끝에서는 옮긴 창이 맡은 구간을 다 덮지 못한다
            got = transcribe(a, b, prompt)
            n = own_chars(got, chunk)
            tried.append({"attempt": name, "rate": round(n / sec, 3)})
            if n / sec > MAX_RATIO * median:
                continue
            if n > best[2]:
                best = (name, got, n)
            if n / sec >= ACCEPT_RATIO * median:
                chosen = name
                break
        out[k] = (chunk, best[1])
        log.append({"window": [round(chunk.start, 3), round(chunk.end, 3)], "speech_sec": round(sec, 2),
                    "rate_before": round(own_chars(words, chunk) / sec, 3), "median_rate": round(median, 3),
                    "tried": tried, "used": best[0], "accepted": chosen is not None})
    return out, log


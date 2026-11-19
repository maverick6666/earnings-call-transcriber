"""모델 없이 방식의 '작동 원리' 를 확인하는 시험대.

진짜 받아쓰기 모델 대신, 알려진 실패 모양을 그대로 흉내 내는 가짜 모델(FakeASR)을 쓴다.
- 창 가장자리에 걸친 낱말은 사라지거나 엉뚱한 조각이 된다(경계 손실 — 사용자가 겪은 문제).
- 창 가장자리 가까이(기본 1초)는 틀릴 확률이 높다(앞뒤 문맥이 잘려서).
- 창 안에 긴 무음이 있으면 가끔 없는 문장을 지어낸다(Careless Whisper, FAccT 2024 의 관찰).
- 고유명사 · 숫자는 조금 더 자주 틀린다.
이 숫자들은 **실제 모델의 성능이 아니다.** 방식(B0 · O1 · O2 · V1 · V2)이 이런 실패에 어떻게 대응하는지, 그리고 점수
계산기가 그 차이를 잡아내는지 확인하는 용도다. 실제 비교는 Earnings-21 · Zeroth-Korean 을 받은 뒤 진짜 모델로 한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .words import RefToken, Word

COMMON = (
    "we", "the", "our", "and", "of", "to", "in", "a", "revenue", "quarter", "growth", "margin", "demand",
    "customers", "year", "compared", "operating", "guidance", "expect", "strong", "segment", "second", "half",
    "outlook", "pricing", "inventory", "cost",
)
ENTITIES = ("Samsung", "HBM", "Galaxy", "Hynix", "Exynos", "Foundry", "TSMC", "Nvidia", "Harman", "OLED")
NUMBERS = ("4.2", "17", "2025", "30", "1.8", "12", "6.5", "100", "45", "3")
HALLUCINATION = ("thank", "you", "for", "watching", "please", "subscribe")


def garble(text: str) -> str:
    """잘못 알아들은 낱말: 정규화(소문자 · 문장부호 제거) 뒤에도 원래 낱말과 다르게."""
    return text[::-1].lower() + "x"


@dataclass
class Timeline:
    words: list[RefToken]  # 정답(시각 · 꼬리표 포함)
    silences: list[tuple[float, float]]  # 2초 넘는 무음 구간
    duration: float

    def energy(self, frame: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        """말소리가 있는 곳은 −20 dB, 낱말 사이는 −45 dB, 무음은 −60 dB 인 가짜 소리 크기(VAD 시험용)."""
        t = (np.arange(int(self.duration / frame)) + 0.5) * frame
        db = np.full(len(t), -45.0)
        for w in self.words:
            db[(t >= w.start) & (t < w.end)] = -20.0
        for a, b in self.silences:
            db[(t >= a) & (t < b)] = -60.0
        return t, db


def make_call(seed: int, n_words: int = 2500, lead_silence: float = 0.0) -> Timeline:
    """한 통화: 문장(6~18낱말) 사이 0.3~1.2초 쉼, 가끔 3~8초 긴 쉼. lead_silence 는 처음 대기음(경계 시나리오)."""
    rng = np.random.default_rng(seed)
    words: list[RefToken] = []
    silences: list[tuple[float, float]] = []
    t = 0.5 + lead_silence
    if lead_silence > 2:
        silences.append((0.0, lead_silence))
    while len(words) < n_words:
        for _ in range(int(rng.integers(6, 19))):
            u = rng.random()
            if u < 0.07:
                text, tags = str(rng.choice(ENTITIES)), ("ORG",)
            elif u < 0.13:
                text, tags = str(rng.choice(NUMBERS)), ("CARDINAL",)
            else:
                text, tags = str(rng.choice(COMMON)), ()
            d = float(rng.uniform(0.18, 0.55))
            words.append(RefToken(text, round(t, 3), round(t + d, 3), tags))
            t += d + float(rng.uniform(0.02, 0.12))
        if rng.random() < 0.04:
            gap = float(rng.uniform(3, 8))
            silences.append((t, t + gap))
            t += gap
        else:
            t += float(rng.uniform(0.3, 1.2))
    return Timeline(words, silences, round(t + 0.5, 3))


@dataclass
class FakeASR:
    timeline: Timeline
    seed: int = 0
    p_base: float = 0.03  # 평소 틀릴 확률
    p_delete: float = 0.01
    p_edge: float = 0.30  # 창 가장자리 가까이 추가로 틀릴 확률
    edge_zone: float = 1.0
    p_tagged: float = 0.05  # 고유명사 · 숫자 추가
    p_fragment: float = 0.5  # 가장자리에 걸친 낱말이 엉뚱한 조각으로 나올 확률(아니면 사라짐)
    p_hallucinate: float = 0.5  # 창 안 긴 무음에서 지어낼 확률
    # 자른 곳이 쉼(낱말 없음)이면 가장자리 불이익을 이만큼으로 줄인다. 가정이라 보고서에서 두 값을 견준다.
    pause_edge_factor: float = 1.0
    # 단어 확률이 맞고 틀림을 알려 주는가. True 면 맞은 낱말 0.8~0.99 · 틀린 낱말 0.3~0.7, False 면 둘 다 0.5~0.99.
    # O2 투표가 확률에 기대므로 이 가정에 따라 결과가 달라진다 — 보고서에서 두 값을 견준다.
    informative_prob: bool = True
    name: str = "fake-asr"
    version: str = "1"

    def __post_init__(self):
        self._starts = np.array([w.start for w in self.timeline.words])
        self._ends = np.array([w.end for w in self.timeline.words])

    def _p(self, rng: np.random.Generator, correct: bool, lo: float, hi: float) -> float:
        if not self.informative_prob:
            return float(rng.uniform(0.5, 0.99))
        return float(rng.uniform(lo, hi))

    def _in_pause(self, t: float, gap: float = 0.15) -> bool:
        """그 시각 ±gap 안에 걸친 낱말이 없으면 쉼에서 자른 것."""
        return not bool(np.any((self._starts < t + gap) & (self._ends > t - gap)))

    def transcribe(self, start: float, end: float) -> list[Word]:
        h = hashlib.sha256(f"{self.seed}|{start:.3f}|{end:.3f}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        at_start, at_end = start > 1e-6, end < self.timeline.duration - 1e-6  # 녹음 처음 · 끝은 잘린 곳이 아니다
        f_start = self.pause_edge_factor if at_start and self._in_pause(start) else 1.0
        f_end = self.pause_edge_factor if at_end and self._in_pause(end) else 1.0
        out: list[Word] = []
        for w in self.timeline.words:
            if w.end <= start or w.start >= end:
                continue
            cut = (w.start < start and at_start) or (w.end > end and at_end)
            if cut:
                if rng.random() < self.p_fragment:
                    a, b = max(w.start, start), min(w.end, end)
                    out.append(Word(garble(w.text[: max(1, len(w.text) // 2)]), a, b, self._p(rng, False, 0.2, 0.5)))
                continue
            d_start = w.start - start if at_start else np.inf
            d_end = end - w.end if at_end else np.inf
            edge = 0.0
            if min(d_start, d_end) < self.edge_zone:
                edge = self.p_edge * (f_start if d_start <= d_end else f_end)
            p_err = self.p_base + edge + (self.p_tagged if w.tags else 0.0)
            u = rng.random()
            if u < self.p_delete:
                continue
            if u < self.p_delete + p_err:
                out.append(Word(garble(w.text), w.start, w.end, self._p(rng, False, 0.3, 0.7)))
            else:
                out.append(Word(w.text, w.start, w.end, self._p(rng, True, 0.8, 0.99)))
        for a, b in self.timeline.silences:
            a2, b2 = max(a, start), min(b, end)
            if b2 - a2 >= 2.0 and rng.random() < self.p_hallucinate:
                step = (b2 - a2) / (len(HALLUCINATION) + 1)
                for k, text in enumerate(HALLUCINATION):
                    s = a2 + step * (k + 0.5)
                    out.append(Word(text, s, s + step * 0.8, self._p(rng, False, 0.5, 0.9)))
        return sorted(out, key=lambda w: w.start)

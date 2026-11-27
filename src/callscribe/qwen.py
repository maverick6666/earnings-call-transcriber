"""Qwen3-ASR: 맥락(배경 글 · 이름 목록)을 넣을 수 있는 음성 LLM 을 같은 평가 틀에 붙인다.

- 받아쓰기: `Qwen3ASRModel.transcribe(audio, context=..., language=..., return_time_stamps=True)`.
  context 는 시스템 프롬프트로 들어간다(qwen-asr 0.0.6 `_build_messages`).
- 낱말 시각: Qwen3-ForcedAligner. 그런데 시각 맞추기 도구는 낱말을 글자 · 숫자 · 작은따옴표만 남기고 다듬는다
  ('4.2%' → '42', '$1.5' → '15'). 그 글을 그대로 쓰면 숫자가 망가지므로, **낱말 글은 받아쓰기 원문에서 가져오고
  시각만** 맞춘 조각에서 옮겨 붙인다(`words_from_alignment`).
- 단어 확률은 주지 않는다(prob=None) — 0층 위험 점수의 확률 특징은 이 모델에서 비어 있다.

qwen-asr · torch 는 무거워서 기본 의존성에 넣지 않는다(`pip install -e ".[qwen]"`, 별도 환경 권장).
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .words import Word

SAMPLE_RATE = 16_000


def _kept(ch: str) -> bool:
    """시각 맞추기 도구(qwen-asr)가 남기는 글자: 글자 · 숫자 · 작은따옴표."""
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def words_from_alignment(text: str, items: Sequence[tuple[str, float, float]], offset: float = 0.0,
                         clip_end: float | None = None) -> list[Word]:
    """받아쓰기 원문(띄어쓰기 낱말)에 맞춘 조각들의 시각을 붙인다.

    원문 낱말을 다듬은 글자 줄과 조각 글자 줄을 글자 단위로 맞춰, 낱말마다 그 글자들이 든 조각의 처음 시작 ~
    마지막 끝을 쓴다. 다듬으면 글자가 남지 않는 낱말('%', '—')은 앞 낱말 끝 시각(없으면 다음 낱말 시작)에 붙인다.
    """
    words = text.split()
    if not words:
        return []
    hyp_chars, hyp_word = [], []
    for wi, w in enumerate(words):
        for ch in w:
            if _kept(ch):
                hyp_chars.append(ch.lower())
                hyp_word.append(wi)
    ali_chars, ali_item = [], []
    for ii, (t, _, _) in enumerate(items):
        for ch in t:
            ali_chars.append(ch.lower())
            ali_item.append(ii)
    span: list[list[float] | None] = [None] * len(words)
    sm = difflib.SequenceMatcher(None, hyp_chars, ali_chars, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            wi, ii = hyp_word[a + k], ali_item[b + k]
            s, e = float(items[ii][1]), float(items[ii][2])
            if span[wi] is None:
                span[wi] = [s, e]
            else:
                span[wi][0] = min(span[wi][0], s)
                span[wi][1] = max(span[wi][1], e)
    # 시각이 없는 낱말: 앞 낱말 끝(없으면 다음 낱말 시작)에 길이 0 으로
    for wi in range(len(words)):
        if span[wi] is None:
            prev = next((span[j][1] for j in range(wi - 1, -1, -1) if span[j] is not None), None)
            nxt = next((span[j][0] for j in range(wi + 1, len(words)) if span[j] is not None), None)
            t = prev if prev is not None else (nxt if nxt is not None else 0.0)
            span[wi] = [t, t]
    out, last = [], 0.0
    for w, (s, e) in zip(words, span, strict=True):
        s = max(s, last)  # 시각 순서 유지(불변조건: 결과 낱말은 시각 순)
        e = max(e, s)
        if clip_end is not None:
            s, e = min(s, clip_end), min(e, clip_end)
        out.append(Word(w, round(offset + s, 3), round(offset + e, 3), None))
        last = s
    return out


@dataclass
class QwenASR:  # pragma: no cover - 모델이 있어야 돈다
    """runner.Transcriber 와 같은 모양. 한 창씩도 되지만 `transcribe_many` 로 여러 창을 한 번에 넣는 것이 빠르다."""

    audio: np.ndarray | None = None
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_name: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str | None = None  # "English" · "Korean"
    context: str = ""
    batch_size: int = 8
    max_new_tokens: int = 512
    name: str = "qwen3-asr"
    version: str = field(init=False, default="")

    def __post_init__(self):
        import importlib.metadata

        import torch
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens, forced_aligner=self.aligner_name,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"))
        self.version = (f"qwen-asr {importlib.metadata.version('qwen-asr')} · {self.model_name.split('/')[-1]}"
                        f" + {self.aligner_name.split('/')[-1]} · bf16")

    @property
    def settings(self) -> dict:
        return {"language": self.language, "context": self.context, "max_new_tokens": self.max_new_tokens}

    def transcribe_many(self, spans: Sequence[tuple[float, float]]) -> list[list[Word]]:
        if self.audio is None:
            raise RuntimeError("audio 를 먼저 넣어야 한다")
        clips = [(self.audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)], SAMPLE_RATE) for a, b in spans]
        res = self._model.transcribe(audio=clips, context=[self.context] * len(clips),
                                     language=[self.language] * len(clips), return_time_stamps=True)
        out = []
        for (a, b), r in zip(spans, res, strict=True):
            items = [(it.text, it.start_time, it.end_time) for it in (r.time_stamps or [])]
            out.append(words_from_alignment(r.text, items, offset=a, clip_end=b - a))
        return out

    def transcribe(self, start: float, end: float) -> list[Word]:
        return self.transcribe_many([(start, end)])[0]

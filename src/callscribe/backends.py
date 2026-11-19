"""진짜 받아쓰기 모델. faster-whisper 와 모델 파일이 있어야 한다(내려받기는 사용자 허락 뒤).

B1(모델 자체의 긴 오디오 처리)은 청크 하나(녹음 전체)를 이 클래스에 넘기면 된다 — faster-whisper 가 안에서 30초 창으로
앞 문장을 이어 받으며 처리한다. 나머지 방식은 runner 가 자른 구간을 하나씩 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .words import Word

SAMPLE_RATE = 16_000


@dataclass
class FasterWhisperASR:
    audio: np.ndarray  # 16 kHz 모노 float32
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    beam_size: int = 5
    hotwords: str | None = None  # +P: 회사별 용어집
    condition_on_previous_text: bool = True
    name: str = "faster-whisper"
    version: str = field(init=False, default="")

    def __post_init__(self):
        try:
            import faster_whisper
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - 설치 전에는 여기서 멈춘다
            raise RuntimeError('faster-whisper 가 없다: pip install -e ".[asr]" (모델은 처음 실행 때 내려받음)') from e
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.version = f"faster-whisper {faster_whisper.__version__} · {self.model_size} · {self.compute_type}"

    def transcribe(self, start: float, end: float) -> list[Word]:  # pragma: no cover - 모델이 있어야 돈다
        clip = self.audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        segments, _ = self._model.transcribe(
            clip, language=self.language, beam_size=self.beam_size, word_timestamps=True, vad_filter=False,
            hotwords=self.hotwords, condition_on_previous_text=self.condition_on_previous_text)
        out = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    out.append(Word(text, round(start + w.start, 3), round(start + w.end, 3), float(w.probability)))
        return out

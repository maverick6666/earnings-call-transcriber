"""진짜 받아쓰기 모델. faster-whisper 와 모델 파일이 있어야 한다(내려받기는 사용자 허락 뒤).

B1(모델 자체의 긴 오디오 처리)은 청크 하나(녹음 전체)를 이 클래스에 넘기면 된다 — faster-whisper 가 안에서 30초 창으로
앞 문장을 이어 받으며 처리한다. 나머지 방식은 runner 가 자른 구간을 하나씩 넘긴다.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .words import Word

SAMPLE_RATE = 16_000


def use_pip_cuda_libs() -> list[str]:
    """Windows: pip 로 받은 nvidia-cublas · nvidia-cudnn 의 DLL 폴더를 CTranslate2 가 찾게 한다(CUDA 전체 설치 없이)."""
    if sys.platform != "win32":
        return []
    found = []
    for base in map(Path, sys.path):
        root = base / "nvidia"
        if root.is_dir():
            for bin_dir in root.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                found.append(str(bin_dir))
    return found


def speech_energy(audio: np.ndarray, frame: float = 0.03, nonspeech_db: float = -70.0,
                  min_silence_ms: int = 500, pad_ms: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """V1 · V2 용 소리 크기: 프레임별 dBFS 에, Silero VAD 가 말소리가 아니라고 본 곳은 nonspeech_db 로 덮는다.

    에너지만 보면 대기 음악 · 잡음도 '소리' 라서 건너뛰지 못한다.
    Silero VAD(faster-whisper 에 들어 있음, 따로 받지 않음)로 말소리 구간을 찾고, 그 밖은 아주 조용하게 표시해
    chunking.vad 가 건너뛰게(2초 넘으면) · 자르게(짧은 쉼) 한다.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    from .chunking import frame_energy

    times, db = frame_energy(audio, SAMPLE_RATE, frame)
    segs = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms))
    speech = np.zeros(len(times), dtype=bool)
    for seg in segs:
        speech[(times >= seg["start"] / SAMPLE_RATE) & (times < seg["end"] / SAMPLE_RATE)] = True
    return times, np.where(speech, db, nonspeech_db)


def load_audio(path: str | Path) -> np.ndarray:
    """mp3 · flac · wav → 16 kHz 모노 float32 (faster-whisper 가 함께 설치하는 PyAV 로 디코딩)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


@dataclass
class FasterWhisperASR:
    audio: np.ndarray | None = None  # 16 kHz 모노 float32. 모델은 한 번 올리고 녹음마다 바꿔 끼운다
    model_size: str = "large-v3-turbo"  # 평가에서 정한 기본값(large-v3 와 정확도가 같고 3배 빠름)
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    beam_size: int = 5
    hotwords: str | None = None  # +P: 회사별 용어집
    condition_on_previous_text: bool = True
    download_root: str | None = None  # 모델 파일 폴더(없으면 Hugging Face 기본 위치)
    name: str = "faster-whisper"
    version: str = field(init=False, default="")
    padded_retries: list = field(init=False, default_factory=list)  # 끝에 무음을 붙여 다시 받아쓴 조각

    def __post_init__(self):
        use_pip_cuda_libs()
        try:
            import faster_whisper
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - 설치 전에는 여기서 멈춘다
            raise RuntimeError('faster-whisper 가 없다: pip install -e ".[asr]" (모델은 처음 실행 때 내려받음)') from e
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type,
                                   download_root=self.download_root)
        self.version = f"faster-whisper {faster_whisper.__version__} · {self.model_size} · {self.compute_type}"

    @property
    def settings(self) -> dict:
        """받아쓰기 결과를 바꾸는 설정(저장본 열쇠 · 실행 기록에 들어감)."""
        return {"language": self.language, "beam_size": self.beam_size, "hotwords": self.hotwords,
                "condition_on_previous_text": self.condition_on_previous_text}

    def transcribe(self, start: float, end: float) -> list[Word]:
        if self.audio is None:
            raise RuntimeError("audio 를 먼저 넣어야 한다")
        clip = self.audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        try:
            return self._words(clip, start)
        except IndexError:
            # faster-whisper 1.2.1 의 단어 시각 맞추기가 조각 맨 끝의 아주 짧은 구간에서 넘어진다.
            # 끝에 1초 무음을 붙여 다시 받아쓰고, 붙인 무음 안의 낱말은 버린다(이 경우만, 기록에 남김).
            self.padded_retries.append((round(start, 3), round(end, 3)))
            padded = np.concatenate([clip, np.zeros(SAMPLE_RATE, dtype=clip.dtype)])
            return [w for w in self._words(padded, start) if w.start < end]

    def _words(self, clip: np.ndarray, start: float) -> list[Word]:  # pragma: no cover
        segments, _ = self._model.transcribe(
            clip, language=self.language, beam_size=self.beam_size, word_timestamps=True, vad_filter=False,
            hotwords=self.hotwords, condition_on_previous_text=self.condition_on_previous_text)
        out = []
        for seg in segments:  # 받아쓰기는 여기서 돈다(segments 는 게으른 생성기)
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    out.append(Word(text, round(start + w.start, 3), round(start + w.end, 3), float(w.probability)))
        return out

"""방식 하나로 녹음 한 건을 받아 적는다: 청크 계획 → 청크마다 받아쓰기(저장본 있으면 건너뜀) → 이어 붙이기 → 검사.

- 같은 녹음 + 같은 설정 → 같은 결과(불변조건 5). 설정 · 모델 이름과 버전 · 녹음 식별값을 실행 기록에 남기고,
  결과 해시도 함께 적는다.
- 받아쓰기 도중 멈췄다가 다시 돌리면, 끝난 청크는 저장본을 읽고 다시 받아 적지 않는다(상태 경계 시나리오).
  저장본 열쇠는 (설정 해시, 녹음 식별값, 청크 구간)이라 설정이 바뀌면 섞이지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import chunking, merge
from .words import Word

STRATEGIES = ("B0", "B1", "O1", "O2", "V1", "V2")


class Transcriber(Protocol):
    name: str
    version: str

    def transcribe(self, start: float, end: float) -> list[Word]: ...


@dataclass(frozen=True)
class RunConfig:
    strategy: str
    fixed_size: float = 60.0  # B0
    window: float = 30.0  # O1 · O2 창 길이
    hop: float | None = None  # O1 · O2 간격(기본 window/2 = 1+2, 2+3)
    vad_max: float = 30.0  # V1 · V2
    vad_min: float = 5.0
    vad_pad: float = 1.0  # V2
    silence_db: float = -50.0
    min_silence: float = 2.0
    extra: dict = field(default_factory=dict)  # 모델 설정(예: beam_size, language) — 해시에 들어간다

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy 는 {STRATEGIES} 중 하나")

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class RunResult:
    words: list[Word]
    chunks: list[chunking.Chunk]
    manifest: dict
    transcribed: int
    from_cache: int


def plan(cfg: RunConfig, duration: float, energy: tuple[np.ndarray, np.ndarray] | None = None) -> list[chunking.Chunk]:
    s = cfg.strategy
    if s == "B0":
        return chunking.fixed(duration, cfg.fixed_size)
    if s == "B1":
        return chunking.single(duration)
    if s in ("O1", "O2"):
        return chunking.overlapped(duration, cfg.window, cfg.hop)
    if energy is None:
        raise ValueError("V1 · V2 는 소리 크기(energy)가 필요하다")
    times, db = energy
    return chunking.vad(duration, times, db, cfg.vad_max, cfg.vad_min, cfg.vad_pad if s == "V2" else 0.0,
                        cfg.silence_db, cfg.min_silence)


def run(cfg: RunConfig, asr: Transcriber, duration: float, audio_id: str, *, cache_dir: Path | None = None,
        energy: tuple[np.ndarray, np.ndarray] | None = None) -> RunResult:
    chunks = plan(cfg, duration, energy)
    problems = chunking.check_partition(chunks, duration)
    if problems:
        raise AssertionError(f"청크 계획이 녹음을 정확히 한 번 덮지 않는다: {problems}")
    key = f"{cfg.digest()}-{hashlib.sha256(f'{audio_id}|{asr.name}|{asr.version}'.encode()).hexdigest()[:12]}"
    store = (cache_dir / key) if cache_dir else None
    if store:
        store.mkdir(parents=True, exist_ok=True)
    results, done, cached = [], 0, 0
    for c in chunks:
        if c.skip:
            results.append((c, []))
            continue
        path = store / f"{c.start:.3f}-{c.end:.3f}.json" if store else None
        if path and path.exists():
            words = [Word(**w) for w in json.loads(path.read_text(encoding="utf-8"))]
            cached += 1
        else:
            words = asr.transcribe(c.start, c.end)
            done += 1
            if path:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps([w.to_dict() for w in words], ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)  # 다 쓴 뒤에만 이름을 바꿔, 중간에 멈춰도 반쯤 쓴 저장본이 남지 않게
        results.append((c, words))
    words = merge.merge_vote(results) if cfg.strategy == "O2" else merge.merge_own(results)
    check_output(words, chunks)
    manifest = {
        "config": cfg.to_dict(), "config_hash": cfg.digest(), "transcriber": {"name": asr.name, "version": asr.version},
        "audio_id": audio_id, "duration": duration, "chunks": len(chunks),
        "skipped_chunks": sum(c.skip for c in chunks),
        "output_hash": hashlib.sha256(json.dumps([w.to_dict() for w in words]).encode()).hexdigest()[:16],
    }
    return RunResult(words, chunks, manifest, done, cached)


def check_output(words: list[Word], chunks: list[chunking.Chunk]) -> None:
    """결과 검사: 시각 순서, 모든 낱말에 시각, 받아 적지 않은(skip) 구간에 낱말 없음."""
    for a, b in zip(words, words[1:], strict=False):
        if b.start < a.start - 1e-9:
            raise AssertionError("결과 낱말이 시각 순서가 아니다")
    for w in words:
        if w.start is None or w.end is None or w.end < w.start:
            raise AssertionError(f"시각이 없는 낱말: {w}")
    for c in chunks:
        if c.skip and any(c.own_start <= w.mid < c.own_end for w in words):
            raise AssertionError(f"무음으로 비워 둔 구간 {c.own_start:.1f}~{c.own_end:.1f}초에 낱말이 있다")

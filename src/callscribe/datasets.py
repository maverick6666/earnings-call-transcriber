"""평가 자료 읽기. 자료 자체는 저장소에 없다(내려받기는 사용자 허락 뒤).

Earnings-21 (Rev, CC BY-SA 4.0) — 실제 어닝콜 44건 · 39시간.
    media/<id>.mp3, transcripts/nlp_references/<id>.nlp, transcripts/wer_tags/<id>.wer_tag.json
    .nlp 는 한 줄에 낱말 하나, 구분자 '|', 머리줄 `token|speaker|ts|endTs|punctuation|case|tags|wer_tags`.
    tags 는 ['5:TIME'] 처럼 '번호:종류', wer_tags 는 번호만이고 종류는 wer_tag.json({"5": {"entity_type": "TIME"}}).
    ts · endTs 는 비어 있을 수 있다 → 그러면 점수 계산이 받아쓰기 시각으로 위치를 잡는다(metrics.op_times).

Zeroth-Korean (OpenSLR 40, CC BY 4.0) — 낭독 문장. 긴 오디오는 문장을 쉼과 함께 이어 붙여 만든다(build_long_audio):
    정답이 문장 단위 시각까지 정확히 있어 경계 효과를 잴 수 있다. 문장 안 글자 시각은 고르게 나눈 근사다.
    폴더 구조는 LibriSpeech 식(`*.trans.txt` 에 '문장번호 글') 으로 보고 읽는다 — 내려받은 뒤 다시 확인한다.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .normalize import normalize_ko_chars
from .words import RefToken

NLP_HEADER = ["token", "speaker", "ts", "endTs", "punctuation", "case", "tags", "wer_tags"]


def _float(x: str) -> float | None:
    """빈 칸 · 깨진 값('.' — Eval-10 밖 몇몇 파일에 있다)은 시각 없음으로 본다."""
    try:
        return float(x)
    except ValueError:
        return None


def read_nlp(path: Path, wer_tags_path: Path | None = None) -> list[RefToken]:
    """Earnings-21 .nlp → 정답 낱말(시각 · 종류 꼬리표). 종류는 tags 칸을 먼저, 없으면 wer_tag.json 으로."""
    id_to_type: dict[str, str] = {}
    if wer_tags_path and wer_tags_path.exists():
        id_to_type = {k: v.get("entity_type", "") for k, v in json.loads(wer_tags_path.read_text("utf-8")).items()}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("|")
    if header != NLP_HEADER:
        raise ValueError(f"NLP 머리줄이 다르다: {header}")
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("|")
        if len(cols) != len(NLP_HEADER):
            raise ValueError(f"칸 수가 다르다: {line!r}")
        row = dict(zip(NLP_HEADER, cols, strict=True))
        tags: list[str] = []
        for item in _list(row["tags"]):
            tags.append(item.split(":", 1)[1] if ":" in item else item)
        if not tags:
            tags = [id_to_type[i] for i in _list(row["wer_tags"]) if id_to_type.get(i)]
        out.append(RefToken(row["token"], _float(row["ts"]), _float(row["endTs"]), tuple(sorted(set(tags)))))
    return out


def _list(cell: str) -> list[str]:
    cell = cell.strip()
    if not cell or cell == "[]":
        return []
    return [str(x) for x in ast.literal_eval(cell)]


def read_trans(paths: Iterable[Path]) -> dict[str, str]:
    """LibriSpeech 식 '*.trans.txt' 들 → {문장번호: 글}."""
    out = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                uid, _, text = line.strip().partition(" ")
                out[uid] = text.strip()
    return out


def build_long_audio(utterances: list[tuple[str, np.ndarray, str]], sr: int, seed: int = 0,
                     gap: tuple[float, float] = (0.3, 1.2), long_gap_rate: float = 0.05,
                     long_gap: tuple[float, float] = (3.0, 6.0), lead_silence: float = 0.0
                     ) -> tuple[np.ndarray, list[RefToken], list[tuple[float, float, str]]]:
    """문장 녹음을 쉼과 함께 이어 붙인 긴 오디오 · 글자 단위 정답(시각 포함) · 문장 위치표."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead_silence * sr), dtype=np.float32)]
    ref: list[RefToken] = []
    spans = []
    t = lead_silence
    for uid, audio, text in utterances:
        d = len(audio) / sr
        chars = normalize_ko_chars(text)
        for k, ch in enumerate(chars):  # 문장 안 글자 시각은 고르게 나눈 근사
            a = t + d * k / max(1, len(chars))
            ref.append(RefToken(ch, round(a, 3), round(a + d / max(1, len(chars)), 3)))
        spans.append((round(t, 3), round(t + d, 3), uid))
        parts.append(audio.astype(np.float32))
        g = float(rng.uniform(*long_gap)) if rng.random() < long_gap_rate else float(rng.uniform(*gap))
        parts.append(np.zeros(int(g * sr), dtype=np.float32))
        t += d + int(g * sr) / sr
    return np.concatenate(parts), ref, spans

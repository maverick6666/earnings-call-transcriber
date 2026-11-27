"""평가 자료 읽기. 자료 자체는 저장소에 없다(내려받기는 사용자 허락 뒤).

Earnings-21 (Rev, CC BY-SA 4.0) — 실제 어닝콜 44건 · 39시간.
    media/<id>.mp3, transcripts/nlp_references/<id>.nlp, transcripts/wer_tags/<id>.wer_tag.json
    .nlp 는 한 줄에 낱말 하나, 구분자 '|', 머리줄 `token|speaker|ts|endTs|punctuation|case|tags|wer_tags`.
    tags 는 ['5:TIME'] 처럼 '번호:종류', wer_tags 는 번호만이고 종류는 wer_tag.json({"5": {"entity_type": "TIME"}}).
    ts · endTs 는 비어 있을 수 있다 → 그러면 점수 계산이 받아쓰기 시각으로 위치를 잡는다(metrics.op_times).

Zeroth-Korean (OpenSLR 40, CC BY 4.0) — 낭독 문장. 긴 오디오는 문장을 쉼과 함께 이어 붙여 만든다(build_long_audio):
    정답이 문장 단위 시각까지 정확히 있어 경계 효과를 잴 수 있다. 문장 안 글자 시각은 고르게 나눈 근사다.
    폴더 구조는 LibriSpeech 식(`*.trans.txt` 에 '문장번호 글') 으로 보고 읽는다 — 내려받은 뒤 다시 확인한다.

한국 실적 콜(삼성SDS IR 페이지) — 녹음 + 회사가 낸 스크립트. 스크립트는 정답 전사문이 아니라 회사 글이다:
    발표 부분은 거의 말한 그대로, 질의응답은 다듬어 줄인 글이고, 사회자 안내는 없다. 화자 꼬리표는 말한 것이 아니라 뺀다
    (read_kr_script). 시각이 없어 점수 계산이 받아쓰기 시각으로 위치를 잡는다.
"""

from __future__ import annotations

import ast
import json
import re
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


# 스크립트의 화자 꼬리표: 'Q. 증권사, 이름 위원' · 'A1. IR팀장, 이름 상무' · '[ 물류사업부 ]'
# · 끝인사 앞 'IR팀장, 이름 상무'
_KR_LABEL = re.compile(r"(?:Q|A\d*)\.\s.*|\[.*\]|[^,]{2,40},\s*[가-힣]{2,4}\s*"
                       r"(?:상무|전무|부사장|사장|부회장|이사|위원|팀장|실장|센터장|본부장|연구원|애널리스트)")
_KR_QA = re.compile(r"\[\s*Q\s*&\s*A.*\]")
_KR_TITLE = re.compile(r".*컨퍼런스\s*콜")
# 숫자 + 붙은 단위('3조' · '3,913억원' · '0.3%p' · '23,300여개사'). 뒤 조사('으로' 등)는 숫자 꼬리표에 넣지 않는다.
KR_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?(?:%p|%|여|조|억|만|천|백|원|개사|개|명|배|년|분기|월|일|위|건|곳|p)*")


def read_kr_script(text: str) -> list[tuple[str, str]]:
    """회사 스크립트 글 → [(부분, 문단)]. 부분은 'talk'(발표) · 'qa'(질의응답). 첫 줄 제목과 화자 꼬리표는 뺀다.
    PDF 에서 뽑은 글은 줄이 문장 중간에서 끊겨 있어 빈 줄 · 꼬리표 사이를 한 문단으로 잇는다."""
    out: list[tuple[str, str]] = []
    section, buf = "talk", []
    lines = [line.strip() for line in text.splitlines()]
    first = next((k for k, line in enumerate(lines) if line), len(lines))
    if first < len(lines) and _KR_TITLE.fullmatch(lines[first]):
        lines[first] = ""

    def flush() -> None:
        if buf:
            out.append((section, " ".join(buf)))
            buf.clear()

    for line in lines:
        if _KR_QA.fullmatch(line):
            flush()
            section = "qa"
        elif not line or _KR_LABEL.fullmatch(line):
            flush()
        else:
            buf.append(line)
    flush()
    return out


def kr_reference(paragraphs: Iterable[tuple[str, str]], entities: dict[str, str]) -> list[RefToken]:
    """문단 → 정답 낱말. 고유명사(entities: 글 → 종류, 예 {'삼성SDS': 'ORG'})와 숫자(KR_NUMBER)는 떼어 꼬리표를 달고,
    질의응답 부분 낱말에는 'QA' 꼬리표를 더한다. 고유명사는 긴 것부터 맞춘다('삼성SDS' 가 'SDS' 보다 먼저)."""
    names = sorted((n for n in entities if n.strip()), key=len, reverse=True)
    pattern = re.compile("|".join([*(re.escape(n) for n in names), KR_NUMBER.pattern]))
    out: list[RefToken] = []
    for section, text in paragraphs:
        extra = ("QA",) if section == "qa" else ()
        pos = 0
        for m in pattern.finditer(text):
            out.extend(RefToken(t, tags=extra) for t in text[pos:m.start()].split())
            out.append(RefToken(m.group(), tags=(entities.get(m.group(), "CARDINAL"), *extra)))
            pos = m.end()
        out.extend(RefToken(t, tags=extra) for t in text[pos:].split())
    return out

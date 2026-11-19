"""+P: 회사별 용어집 → hotwords.

어닝콜에서 가장 비싼 오류는 회사명 · 제품명 · 사업부 · 숫자다. 그 회사가 낸 공시 · 사업보고서 글(한국 회사는 DART,
Earnings-21 의 미국 회사는 공개 자료)에서 자주 나오는 고유명사 후보를 뽑아 모델의 hotwords 에 넣는다.
`initial_prompt` 는 첫 창에만 영향을 주지만 hotwords 는 창마다 디코더 앞에 들어간다(faster-whisper).

뽑기 규칙(모델 없이, 설명 가능하게)
- 영어: 대문자로 시작하는 낱말 1~3개 묶음, 대문자 약어(HBM, OLED), 글자 + 숫자 섞인 말(HBM3E, A17)
- 한국어: 2글자 넘는 한글 · 영문 · 숫자 덩어리 중 흔한 말(STOPWORDS_KO)을 뺀 것
- 두 번 이상 나온 것만, 많이 나온 순서로, 글자 수 한도(MAX_CHARS) 안에서
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .align import align
from .normalize import normalize_en
from .words import RefToken, Word

MAX_CHARS = 400  # hotwords 는 디코더 앞에 들어가 창의 토큰 한도를 나눠 쓴다 — 짧게
_EN_NAME = re.compile(
    r"\b(?:[A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+){0,2}"  # 대문자로 시작하는 1~3낱말 이름
    r"|[A-Z]{2,}[0-9A-Z]*"  # 약어
    r"|[A-Za-z]+[0-9][0-9A-Za-z]*)\b"  # 글자 + 숫자(HBM3E)
)
_KO_TERM = re.compile(r"[가-힣A-Za-z0-9]{2,}")
STOPWORDS_EN = {"The", "We", "Our", "This", "That", "And", "In", "For", "As", "It", "On", "Thank", "Yes", "So", "Q"}
STOPWORDS_KO = {"그리고", "있습니다", "합니다", "하였습니다", "당사는", "당사의", "있는", "대한", "위한", "통해",
                "따라", "경우", "이상", "이하", "기준", "사업", "보고서", "주식", "회사"}


def extract_terms(texts: Iterable[str], language: str = "en", min_count: int = 2, max_terms: int = 60) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        if language == "en":
            for m in _EN_NAME.finditer(text):
                term = m.group(0).strip()
                if term not in STOPWORDS_EN and not term.isdigit():
                    counts[term] += 1
        else:
            for m in _KO_TERM.finditer(text):
                term = m.group(0)
                if term not in STOPWORDS_KO and not term.isdigit():
                    counts[term] += 1
    terms = [t for t, c in counts.most_common() if c >= min_count]
    return terms[:max_terms]


def hotwords(terms: Sequence[str], max_chars: int = MAX_CHARS) -> str:
    out, size = [], 0
    for t in terms:
        if size + len(t) + 1 > max_chars:
            break
        out.append(t)
        size += len(t) + 1
    return " ".join(out)


def term_recall(ref: Sequence[RefToken], hyp: Sequence[Word], terms: Sequence[str]) -> dict:
    """정답에 나온 용어집 낱말 중 맞게 받아 적은 비율(+P 효과를 직접 재는 숫자)."""
    vocab = {p for t in terms for p in normalize_en(t)}
    r = [p for tok in ref for p in normalize_en(tok.text)]
    h = [p for w in hyp for p in normalize_en(w.text)]
    hits = total = 0
    for op, i, _ in align(r, h):
        if i is not None and r[i] in vocab:
            total += 1
            hits += op == "C"
    return {"ref": total, "correct": hits, "recall": round(hits / total, 5) if total else None}

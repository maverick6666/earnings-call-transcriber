"""점수를 매기기 전 글자 고르기. 정답과 받아쓰기에 똑같이 쓴다.

- 영어(WER): 소문자 · 앞뒤 문장부호 제거 · 하이픈으로 이은 말은 나눈다(Earnings-21 낱말 규칙과 같게). 숫자는 그대로 둔다
  — '20 percent' 와 'twenty percent' 를 같게 보는 숫자 정규화는 하지 않고, 숫자 오류는 따로 센다.
- 한국어(CER): 유니코드 NFC, 띄어쓰기 · 문장부호를 빼고 글자 단위로 비교한다(띄어쓰기 차이를 오류로 세지 않기 위해).
"""

from __future__ import annotations

import re
import unicodedata

_EDGE_PUNCT = re.compile(r"^[^\w%$]+|[^\w%$']+$")
_KO_DROP = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_en_token(token: str) -> list[str]:
    t = unicodedata.normalize("NFC", token).lower().replace("’", "'")
    out = []
    for part in t.split("-"):
        part = _EDGE_PUNCT.sub("", part)
        if part:
            out.append(part)
    return out


def normalize_en(text: str) -> list[str]:
    return [p for tok in text.split() for p in normalize_en_token(tok)]


def normalize_ko_chars(text: str) -> list[str]:
    return list(_KO_DROP.sub("", unicodedata.normalize("NFC", text)))

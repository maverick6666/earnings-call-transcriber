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


# Whisper 논문의 영어 평가 관례(EnglishTextNormalizer)에서 점수에 영향이 큰 것만 가볍게 따른다:
# 군말 지우기 · 구어 줄임말 풀기 · 숫자 뒤 % 붙이기. 정답과 받아쓰기에 똑같이 쓴다.
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "mm", "mhm", "ah", "er", "erm", "eh", "hm"}
CONTRACTIONS = {"gonna": ["going", "to"], "wanna": ["want", "to"], "gotta": ["got", "to"]}
_NUMBER = re.compile(r"^[$€£]?\d[\d.,]*$")


def normalize_en_conventional(text: str) -> list[str]:
    """낱말 하나 단위: 군말을 지우고 구어 줄임말을 푼다. 숫자 뒤 % 붙이기는 앞뒤 낱말을 봐야 해서 merge_percent 로."""
    out: list[str] = []
    for tok in normalize_en(text):
        if tok not in FILLERS:
            out.extend(CONTRACTIONS.get(tok, [tok]))
    return out


def merge_percent(items: list, text=lambda x: x.text, join=None) -> list:
    """'60' '%' · '60' 'percent' → '60%' (앞 숫자 낱말에 붙인다). items 는 text 속성이 있는 낱말들."""
    out: list = []
    for x in items:
        t = text(x)
        if t in ("%", "percent") and out and _NUMBER.match(text(out[-1])):
            out[-1] = join(out[-1], text(out[-1]) + "%")
            continue
        out.append(x)
    return out


def normalize_ko_chars(text: str) -> list[str]:
    return list(_KO_DROP.sub("", unicodedata.normalize("NFC", text)))


# 한국어 관례 정규화: 정답(Zeroth-Korean)은 숫자를 한글로 적는다('이천십육년' · '삼월'). 받아쓰기가 '2016년' · '3월' 로
# 적으면 같은 말인데 글자 오류가 된다. 아라비아 숫자를 한자어 수 읽기로 바꿔 양쪽을 맞춘다(영어의 숫자 관례와 같은 뜻).
# 고유어 수('네 킬로미터' · '일곱 시간')는 맞추지 못한다 — 한자어 읽기만.
_KO_DIGIT = "영일이삼사오육칠팔구"
_KO_SMALL = ("", "십", "백", "천")
_KO_BIG = ("", "만", "억", "조", "경")
_KO_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_KO_UNITS = (("%", "퍼센트"), ("km", "킬로미터"), ("kg", "킬로그램"), ("cm", "센티미터"), ("mm", "밀리미터"))


def sino_korean(n: int) -> str:
    """정수 → 한자어 수 읽기. 2016 → 이천십육, 1210 → 천이백십, 10000 → 만, 100000000 → 일억."""
    if n == 0:
        return "영"
    groups, big = [], 0
    while n:
        g = n % 10000
        if g:
            s = "".join(("" if d == 1 and i else _KO_DIGIT[d]) + _KO_SMALL[i]
                        for i in range(3, -1, -1) if (d := g // 10 ** i % 10))
            groups.append(("" if g == 1 and big == 1 else s) + _KO_BIG[big])
        n //= 10000
        big += 1
    return "".join(reversed(groups))


def spell_ko_numbers(text: str) -> str:
    """'2016년 1,210원 4.5% 3km' → '이천십육년 천이백십원 사점오퍼센트 삼킬로미터'."""
    def number(m: re.Match) -> str:
        whole, _, frac = m.group().replace(",", "").partition(".")
        return sino_korean(int(whole)) + ("점" + "".join(_KO_DIGIT[int(d)] for d in frac) if frac else "")

    out = _KO_NUMBER.sub(number, unicodedata.normalize("NFC", text))
    for unit, read in _KO_UNITS:
        out = re.sub(rf"(?<=[영일이삼사오육칠팔구십백천만억조경점]){re.escape(unit)}", read, out, flags=re.I)
    return out


def normalize_ko_conventional(text: str) -> list[str]:
    return normalize_ko_chars(spell_ko_numbers(text))

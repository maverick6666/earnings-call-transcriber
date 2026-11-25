"""검수 1층: 고유명사 제한 교정 — LLM 은 '제안' 만 하고, 받아들일지는 규칙이 정한다.

LLM 에게 받아쓰기를 통째로 고쳐 쓰게 하면 맞는 말까지 바꾸거나 없는 말을 지어낸다(Ma et al. 2023 · ClozeGER ·
RLLM-CF). 그래서 RECOVER(arXiv 2603.16411)처럼 범위를 고유명사로 좁히고, 고칠 수 있는 말을 용어집으로 묶는다.

    구간(약 45초) → 용어집 후보 찾기(글자 · 발음 유사) → 후보가 있는 구간만 제안 받기 → 규칙 검사 → 적용

규칙(validate)
1. 바꿀 말(new)은 용어집에 있어야 한다. 표기는 용어집 것을 쓴다.
2. 원래 말(old)은 그 구간 받아쓰기에 실제로 있어야 한다(낱말 단위로 찾고, 띄어쓰기 차이는 허용 — 'cy tiva').
3. 대소문자 · 띄어쓰기만 다른 바꿈은 버린다(점수에 안 잡히고 표기만 흔든다).
4. old 와 new 가 글자 · 발음으로 충분히 비슷해야 한다(MIN_SIMILARITY, 3글자 이하 약어는 SHORT_MIN_SIMILARITY).
   'citeva → Cytiva' 는 되고 'star → Cytiva' 는 안 된다.
5. old 가 이미 용어집에 있는 다른 이름이면 버린다(맞는 이름을 다른 이름으로 바꾸는 사고).
6. 모양만 바꾸는 편집은 버린다 — 낱말 더하기 · 빼기('financial → Voya Financial'), 붙은 조각 더하기 · 빼기
   ('Healthcare's → Healthcare Options'), 복수 · 소유격('Affimed's → Affimed'), 쉼표 · 마침표를 넘는 old
   ('Johnny, uh, so. → John Hess'), 받아 적은 숫자 · 로마 숫자 바꾸기('Phase III'). 겹치는 낱말이 있으면 나머지끼리
   비슷해야 한다('Eversource in → Eversource Energy' 는 'in' 과 'Energy' 를 비교해 버림).
7. 흔한 낱말(common — 다른 통화들에 두루 나오는 낱말)이 걸린 바꿈은 0층 위험 표시(risky)가 그 자리에 있어야 한다.
   old 가 모두 흔한 낱말('as → ASA', 'at → AT&T', 'in video → Nvidia')이거나 new 가 흔한 낱말 하나('operating →
   Operations')인 경우다. 정답 글에 걸어 본 실험에서 과교정 대부분이 이 모양이었다(reports/review_layer.md).
8. old 는 MAX_OLD 낱말 이하. 겹치는 편집은 먼저 온 것만.

바뀐 낱말은 시각을 원래 낱말 것에서 받고, 원래 글을 `original` 에 남긴다(불변조건 2).
제안자는 둘: FuzzyProposer(LLM 없이 가장 비슷한 후보로 바꿈 — 기준선) · LLMProposer(문맥을 보고 고름).
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .detect import key, word_labels
from .llm import LLM
from .words import RefToken, Word

MIN_SIMILARITY = 0.6
SHORT_MIN_SIMILARITY = 0.85
MAX_OLD = 4
SEGMENT_SEC = 45.0
MAX_CANDIDATES = 30
MIN_CANDIDATE_SCORE = 0.55

_HANGUL = re.compile(r"[가-힣]")
_AFFIX = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_PH_RULES = (("sch", "sk"), ("ph", "f"), ("ck", "k"), ("qu", "kw"), ("wh", "w"), ("kn", "n"), ("gh", "g"),
             ("x", "ks"))


# ---------- 글자 · 발음 유사도 ----------

def jamo(text: str) -> str:
    """한글 음절을 자모로 푼다('현대' → 'ㅎㅕㄴㄷㅐ'). 한 글자 차이를 덜 엄하게 본다."""
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_L[code // 588] + _V[(code % 588) // 28] + _T[code % 28].strip())
        else:
            out.append(ch)
    return "".join(out)


def phonetic_en(k: str) -> str:
    """아주 가벼운 영어 발음 열쇠: 비슷한 소리 글자를 합치고 모음을 'a' 하나로('citeva' · 'cytiva' → 'satava')."""
    s = k
    for a, b in _PH_RULES:
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k").replace("q", "k").replace("z", "s").replace("y", "i")
    s = re.sub(r"[aeiou]+", "a", s)  # 모음은 자리만 남긴다(지우면 'Phil' · 'Paul' 이 같아진다)
    return re.sub(r"(.)\1+", r"\1", s)


def _forms(text: str) -> tuple[str, str]:
    k = key(text)
    if _HANGUL.search(k):
        j = jamo(k)
        return j, j
    return k, phonetic_en(k)


def similarity(a: str, b: str) -> float:
    """글자 유사도와 발음 열쇠 유사도의 평균(0~1). 한국어는 자모로 비교한다."""
    (ca, pa), (cb, pb) = _forms(a), _forms(b)
    if not ca or not cb:
        return 0.0
    char = difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio()
    ph = difflib.SequenceMatcher(None, pa, pb, autojunk=False).ratio() if pa and pb else char
    return (char + ph) / 2


# ---------- 구간 · 후보 ----------

def segments(words: Sequence[Word], max_sec: float = SEGMENT_SEC, min_sec: float = 20.0) -> list[tuple[int, int]]:
    """낱말 [a, b) 구간들. min_sec 을 넘긴 뒤 문장이 끝나면 자르고, 길어도 max_sec 에서 자른다."""
    out, a = [], 0
    for i, w in enumerate(words):
        if i == a:
            continue
        length = w.end - words[a].start
        ends_sentence = bool(re.search(r"[.?!。]$", words[i - 1].text))
        if length >= max_sec or (length >= min_sec and ends_sentence):
            out.append((a, i))
            a = i
    if a < len(words):
        out.append((a, len(words)))
    return out


@dataclass(frozen=True)
class Candidate:
    term: str  # 용어집 표기
    score: float  # 구간 안 가장 비슷한 낱말 묶음과의 유사도(1 = 그대로 있음)
    old: str  # 그 낱말 묶음(받아쓰기 그대로)
    present: bool  # 용어집 말이 구간에 이미 그대로 있나


def _ngrams(seg: Sequence[Word], max_n: int) -> list[tuple[int, int, str, str, str]]:
    keys = [key(w.text) for w in seg]
    out = []
    for s in range(len(seg)):
        if not keys[s]:
            continue
        for n in range(1, max_n + 1):
            if s + n > len(seg):
                break
            joined = "".join(keys[s:s + n])
            char, ph = _forms(joined)
            out.append((s, s + n, joined, char, ph))
    return out


def find_candidates(seg: Sequence[Word], glossary: Sequence[str], k: int = MAX_CANDIDATES,
                    min_score: float = MIN_CANDIDATE_SCORE) -> list[Candidate]:
    """용어집 말마다 구간에서 가장 비슷한 낱말 묶음(용어 낱말 수 ± 1)을 찾아 점수 순으로 k 개."""
    if not seg or not glossary:
        return []
    max_n = min(MAX_OLD, max(len(t.split()) for t in glossary) + 1)
    grams = _ngrams(seg, max_n)
    sm_c = difflib.SequenceMatcher(autojunk=False)
    sm_p = difflib.SequenceMatcher(autojunk=False)
    out = []
    for term in dict.fromkeys(glossary):
        tk = key(term)
        if not tk:
            continue
        tc, tp = _forms(term)
        sm_c.set_seq2(tc)
        sm_p.set_seq2(tp)
        n_term = len(term.split())
        best = (0.0, None)
        for s, e, joined, char, ph in grams:
            if not (max(1, n_term - 1) <= e - s <= n_term + 1) or not 0.6 <= len(joined) / len(tk) <= 1.67:
                continue
            if joined == tk:
                best = (1.0, (s, e))
                break
            sm_c.set_seq1(char)
            if sm_c.quick_ratio() < 0.45:
                continue
            sm_p.set_seq1(ph)
            score = (sm_c.ratio() + (sm_p.ratio() if ph and tp else sm_c.ratio())) / 2
            if score > best[0]:
                best = (score, (s, e))
        score, span = best
        if span is not None and score >= min_score:
            old = " ".join(w.text for w in seg[span[0]:span[1]])
            out.append(Candidate(term, round(score, 4), old, score >= 1.0))
    out.sort(key=lambda c: (-c.score, c.term))
    return out[:k]


# ---------- 제안 ----------

@dataclass(frozen=True)
class Edit:
    old: str
    new: str
    reason: str = ""


class Proposer(Protocol):
    name: str

    def propose(self, seg: Sequence[Word], candidates: Sequence[Candidate]) -> list[Edit]: ...


@dataclass
class FuzzyProposer:
    """LLM 없는 기준선: 그대로 있지 않은 후보 중 유사도가 min_score 이상이면 바꾼다. LLM 이 문맥으로 얼마나 더
    잘 고르는지(또는 덜 망치는지) 견주는 데 쓴다."""
    min_score: float = 0.8
    name: str = "fuzzy"

    def propose(self, seg: Sequence[Word], candidates: Sequence[Candidate]) -> list[Edit]:
        return [Edit(c.old, c.term, f"similarity {c.score:.2f}") for c in candidates
                if not c.present and c.score >= self.min_score]


SYSTEM_PROMPT = """You review an automatic transcript of a company earnings call. Your only job is to fix proper names \
(people, companies, products, brands, tickers) that the speech recognizer misheard.

Rules:
- Propose an edit only when a phrase in TRANSCRIPT is very likely a mishearing of a name listed in CANDIDATES.
- "new" must be copied exactly from CANDIDATES. "old" must be copied exactly from TRANSCRIPT (1 to 4 words).
- Never change grammar, punctuation, casing, filler words, numbers or any other words.
- A name that already appears correctly needs no edit. If you are unsure, propose nothing.

Respond with JSON only: {"edits": [{"old": "...", "new": "...", "reason": "..."}]}"""


def build_prompt(seg: Sequence[Word], candidates: Sequence[Candidate], context: str = "") -> str:
    lines = []
    if context:
        lines += ["CONTEXT:", context, ""]
    lines += ["CANDIDATES (names known for this call):", *[f"- {c.term}" for c in candidates], "",
              "TRANSCRIPT:", " ".join(w.text for w in seg)]
    return "\n".join(lines)


def parse_edits(text: str) -> list[Edit]:
    """LLM 답에서 JSON 을 꺼낸다(코드 울타리 · 앞뒤 말 허용). 모양이 틀리면 빈 목록 — 제안이 없는 것과 같다."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return []
    try:
        obj = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    items = obj.get("edits", []) if isinstance(obj, dict) else []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and isinstance(it.get("old"), str) and isinstance(it.get("new"), str):
            out.append(Edit(it["old"].strip(), it["new"].strip(), str(it.get("reason", ""))[:200]))
    return out


@dataclass
class LLMProposer:
    llm: LLM
    context: str = ""  # 회사 이름 · 분기 같은 한 줄(선택)
    system: str = SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return f"llm:{self.llm.name}"

    def propose(self, seg: Sequence[Word], candidates: Sequence[Candidate]) -> list[Edit]:
        return parse_edits(self.llm.complete(self.system, build_prompt(seg, candidates, self.context)))


# ---------- 검사 · 적용 ----------

@dataclass
class Decision:
    edit: Edit
    ok: bool
    why: str
    new: str = ""  # 용어집 표기
    spans: list[tuple[int, int]] = field(default_factory=list)  # 구간 안 낱말 위치 [a, b)


def find_spans(seg: Sequence[Word], old: str) -> list[tuple[int, int]]:
    target = key(old)
    keys = [key(w.text) for w in seg]
    out, s = [], 0
    while s < len(seg):
        hit = None
        if keys[s]:
            joined = ""
            for e in range(s, min(len(seg), s + MAX_OLD + 1)):
                joined += keys[e]
                if joined == target:
                    hit = (s, e + 1)
                    break
                if not target.startswith(joined):
                    break
        if hit:
            out.append(hit)
            s = hit[1]
        else:
            s += 1
    return out


def common_vocabulary(docs: Iterable[Iterable[str]], min_docs: int = 3) -> set[str]:
    """min_docs 개 넘는 문서(다른 통화 전사문)에 나오는 낱말(key). 회사 고유의 이름이 아니라 흔한 낱말로 본다."""
    df: dict[str, int] = {}
    for doc in docs:
        for k in {key(t) for t in doc} - {""}:
            df[k] = df.get(k, 0) + 1
    return {k for k, n in df.items() if n >= min_docs}


_INNER_PUNCT = re.compile(r"[.?!,;:]")
_ROMAN = re.compile(r"[IVX]+[A-Z]?")


def _numberish(token: str) -> bool:
    core = _AFFIX.match(token).group(2)
    return bool(re.search(r"\d", core)) or bool(_ROMAN.fullmatch(core))


def _check(old: str, new: str, canon: dict[str, str], common: Set[str], min_sim: float,
           shape_rules: bool = True) -> tuple[str, bool]:
    """규칙 2 · 5~8 과 유사도. (이유, 흔한 낱말이라 위험 근거가 필요한가). 이유가 'ok' 면 통과.
    shape_rules=False 면 규칙 6(모양만 바꾸는 편집)을 끈다 — 견주기용."""
    old_toks = [t for t in old.split() if key(t)]
    ko, kn = key(old), key(new)
    if not old_toks:
        return "empty_old", False
    if ko == kn:
        return "case_or_spacing_only", False
    if ko in canon:
        return "old_is_known_name", False
    if len(old_toks) > MAX_OLD:
        return "old_too_long", False
    if not shape_rules:
        if similarity(old, new) < (SHORT_MIN_SIMILARITY if len(kn) <= 3 else min_sim):
            return "not_similar", False
        old_k, new_k = [key(t) for t in old_toks], [key(t) for t in new.split() if key(t)]
        return "ok", bool(common) and (all(k in common for k in old_k) or (len(new_k) == 1 and new_k[0] in common))
    if any(_INNER_PUNCT.search(_AFFIX.match(t).group(2) + _AFFIX.match(t).group(3)) for t in old_toks[:-1]):
        return "crosses_punctuation", False  # 이름 하나가 쉼표 · 마침표를 넘어가지 않는다('Johnny, uh, so.')
    if ko in (kn + "s", kn + "es") or kn in (ko + "s", ko + "es"):
        return "inflection_only", False  # 'Affimed's → Affimed', 'gigawatt → Gigawatts'
    old_k, new_k = [key(t) for t in old_toks], [key(t) for t in new.split() if key(t)]
    if set(old_k) <= set(new_k) or set(new_k) <= set(old_k):
        return "adds_or_drops_words", False
    for x in {ko, ko[:-1] if ko.endswith("s") and len(ko) > 3 else ko}:
        a, b = sorted((x, kn), key=len)
        if a != b and a in b and len(b) - len(a) >= 2:
            return "adds_or_drops_part", False  # 'Healthcare's → Healthcare Options', 'banner → Bay Banners'
    shared = set(old_k) & set(new_k)
    res_old = [t for t in old_toks if key(t) not in shared]
    res_new = [t for t in new.split() if key(t) and key(t) not in shared]
    if any(_numberish(t) for t in res_old):
        return "touches_number", False  # 받아 적은 숫자 · 로마 숫자는 건드리지 않는다('Phase III')
    ro, rn = " ".join(res_old), " ".join(res_new)
    if similarity(ro, rn) < (SHORT_MIN_SIMILARITY if len(key(rn)) <= 3 else min_sim):
        return "not_similar", False  # 겹치는 낱말이 있으면 나머지끼리 비교('Eversource in → Eversource Energy')
    res_old_k, res_new_k = [key(t) for t in res_old], [key(t) for t in res_new]
    needs_risk = bool(common) and (all(k in common for k in res_old_k)
                                   or (len(res_new_k) == 1 and res_new_k[0] in common))
    return "ok", needs_risk


def validate(seg: Sequence[Word], edits: Sequence[Edit], glossary: Sequence[str],
             min_sim: float = MIN_SIMILARITY, *, common: Set[str] = frozenset(),
             risky: np.ndarray | None = None, risk_for_all: bool = False,
             shape_rules: bool = True) -> list[Decision]:
    """risky = 구간 낱말마다 0층 위험 표시(없으면 흔한 낱말이 걸린 바꿈은 모두 버린다 — 규칙 7).
    risk_for_all = 흔한 낱말이 아니어도 모든 바꿈에 위험 표시를 요구한다('Concho → Conoco' 처럼 둘 다 실제 이름인
    경우는 글만 봐서는 못 가린다 — 정답 글 실험의 남은 과교정이 대부분 이 모양)."""
    canon: dict[str, str] = {}
    for t in glossary:
        canon.setdefault(key(t), t)
    out: list[Decision] = []
    taken: set[int] = set()
    for ed in edits:
        new = canon.get(key(ed.new))
        if new is None:
            out.append(Decision(ed, False, "not_in_glossary"))
            continue
        why, needs_risk = _check(ed.old, new, canon, common, min_sim, shape_rules)
        if why != "ok":
            out.append(Decision(ed, False, why))
            continue
        spans = [sp for sp in find_spans(seg, ed.old) if not taken & set(range(*sp))]
        if not spans:
            out.append(Decision(ed, False, "old_not_found"))
            continue
        if needs_risk or risk_for_all:
            spans = [sp for sp in spans if risky is not None and risky[sp[0]:sp[1]].any()]
            if not spans:
                out.append(Decision(ed, False, "common_words_without_risk"))
                continue
        for sp in spans:
            taken |= set(range(*sp))
        out.append(Decision(ed, True, "ok", new, spans))
    return out


def _carry_affixes(old_words: Sequence[Word], new: str) -> list[str]:
    """원래 낱말의 앞 문장부호(첫 낱말) · 뒤 문장부호(끝 낱말)를 새 낱말에 옮긴다('citeva,' → 'Cytiva,')."""
    toks = new.split()
    pre = _AFFIX.match(old_words[0].text).group(1)
    suf = _AFFIX.match(old_words[-1].text).group(3)
    toks[0] = pre + toks[0]
    toks[-1] = toks[-1] + suf
    return toks


def apply_spans(words: Sequence[Word], changes: Sequence[tuple[int, int, str]]
                ) -> tuple[list[Word], list[tuple[int, int, int, int]]]:
    """changes = [(a, b, new)] 낱말 위치(전체 기준). 결과 낱말과 [(원래 a, b, 결과 a, b)] 대응표."""
    out: list[Word] = []
    mapping = []
    pos = 0
    for a, b, new in sorted(changes):
        if a < pos:
            continue  # 겹침
        out.extend(words[pos:a])
        old = words[a:b]
        toks = _carry_affixes(old, new)
        start, end = old[0].start, old[-1].end
        step = (end - start) / len(toks)
        o0 = len(out)
        for n, t in enumerate(toks):
            original = " ".join(w.text for w in old) if n == 0 else ""  # 더한 낱말은 원래 낱말 없음 = 빈 글
            out.append(Word(t, round(start + step * n, 3), round(start + step * (n + 1), 3), None, old[0].chunk,
                            None, original))
        mapping.append((a, b, o0, len(out)))
        pos = b
    out.extend(words[pos:])
    return out, mapping


def correct_entities(words: Sequence[Word], glossary: Sequence[str], proposer: Proposer, *,
                     risky: np.ndarray | None = None, ask_only_risky: bool = False,
                     segment_sec: float = SEGMENT_SEC,
                     max_candidates: int = MAX_CANDIDATES, min_candidate: float = MIN_CANDIDATE_SCORE,
                     min_sim: float = MIN_SIMILARITY, common: Set[str] = frozenset(), risk_for_all: bool = False,
                     shape_rules: bool = True, progress: Callable[[int, int], None] | None = None
                     ) -> tuple[list[Word], dict]:
    """전체 받아쓰기에 고유명사 제한 교정을 건다.

    risky: 낱말마다 0층 위험 표시. 흔한 낱말이 걸린 바꿈의 근거로 쓴다(규칙 7).
    ask_only_risky: 위험 낱말이 든 후보만 제안 받는다 — 제안 받는 구간 수(= LLM 비용)를 줄인다.
    common: 흔한 낱말(common_vocabulary). 비어 있으면 규칙 7 을 쓰지 않는다.
    """
    words = list(words)
    segs = segments(words, segment_sec)
    changes: list[tuple[int, int, str]] = []
    log, asked = [], 0
    reasons: dict[str, int] = {}
    for n, (a, b) in enumerate(segs):
        seg = words[a:b]
        cands = find_candidates(seg, glossary, max_candidates, min_candidate)
        if risky is not None and ask_only_risky:
            seg_risky = risky[a:b]
            cands = [c for c in cands if c.present or any(seg_risky[s:e].any() for s, e in find_spans(seg, c.old))]
        if not any(not c.present for c in cands):
            continue
        asked += 1
        decisions = validate(seg, proposer.propose(seg, cands), glossary, min_sim, common=common,
                             risky=None if risky is None else risky[a:b], risk_for_all=risk_for_all,
                             shape_rules=shape_rules)
        for d in decisions:
            reasons[d.why] = reasons.get(d.why, 0) + 1
            log.append({"segment": [round(words[a].start, 2), round(words[b - 1].end, 2)], "old": d.edit.old,
                        "new": d.edit.new, "reason": d.edit.reason, "decision": d.why,
                        "occurrences": len(d.spans)})
            changes.extend((a + s, a + e, d.new) for s, e in d.spans)
        if progress:
            progress(n + 1, len(segs))
    out, mapping = apply_spans(words, changes)
    report = {"proposer": proposer.name, "segments": len(segs), "asked": asked,
              "proposed": sum(reasons.values()), "decisions": reasons, "applied_spans": len(mapping),
              "edits": log, "mapping": mapping}
    return out, report


def edit_outcomes(ref: Sequence[RefToken], before: Sequence[Word], after: Sequence[Word],
                  mapping: Sequence[tuple[int, int, int, int]], normalize: Callable[[str], list[str]]) -> dict:
    """바꾼 자리마다 정답과 견줘: fixed(틀림→맞음) · broke(맞음→틀림, 과교정) · still_wrong · neutral."""
    lb = word_labels(ref, before, normalize).wrong
    la = word_labels(ref, after, normalize).wrong
    out = {"fixed": 0, "broke": 0, "still_wrong": 0, "neutral": 0}
    for a, b, c, d in mapping:
        was_ok = bool(np.all(lb[a:b] <= 0))
        now_ok = bool(np.all(la[c:d] <= 0))
        out["fixed" if (not was_ok and now_ok) else "broke" if (was_ok and not now_ok)
            else "still_wrong" if not now_ok else "neutral"] += 1
    n = len(mapping)
    out["applied"] = n
    out["over_correction_rate"] = round(out["broke"] / n, 4) if n else None
    out["precision"] = round(out["fixed"] / n, 4) if n else None
    return out

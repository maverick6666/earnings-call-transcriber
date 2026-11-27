"""검수 1층의 '회사 자료로 확정' 제안자 — 회사가 통화 전에 낸 자료에 적힌 표기로만 고친다.

IR 교정자가 하는 일을 따른다: 회사가 쓰는 표기 목록(스타일 시트)을 먼저 만들고, 받아쓰기에서 그 이름을 다르게 적은
곳만 고친다. 자료는 통화 전에 공개된 것만 쓴다 — 그 분기 발표자료, 다시 듣기 페이지의 참석자 명단, 지난 분기 통화
스크립트. 그 통화의 스크립트(정답)는 보지 않는다.

    자료 글 → claude -p(도구 없음)가 이름을 자료에 적힌 그대로 뽑는다(names_from_materials)
    → 받아쓰기 전체와 이름 목록을 함께 주고 바꿀 곳을 고르게 한다(build_prompt)
    → 기계 확인(verify): new 가 목록 표기 그대로이고 자료 글에도 있음 · old 가 받아쓰기에 있음(조사 붙은 낱말 포함)
    → entity_fix 규칙(유사도 — 한글 · 영문은 소리 열쇠 — · 모양 · 조사 다시 붙이기)

검색 제안자(search_fix)와 달리 웹을 보지 않는다. 자료에 없는 이름(애널리스트 · 정부 기관 · 다른 회사)은 검색 제안자가
맡는다. 둘을 함께 쓰면 같은 old 는 자료 쪽을 먼저 쓴다(회사 표기가 우선).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .entity_fix import find_spans, key
from .search_fix import ClaudeSearch, Proposal, Trace, parse_proposals
from .words import Word

NAMES_SYSTEM = "You extract named entities from Korean business documents. Follow the user message. Do not use tools."
NAMES_PROMPT = """아래는 한 회사가 실적발표 컨퍼런스콜 전에 공개한 자료다
(발표 슬라이드, 다시 듣기 페이지의 일정 · 참석자, 지난 분기 컨퍼런스콜 스크립트).
이 글에 나오는 고유명사를 빠짐없이, 글에 적힌 철자 그대로 뽑아라.
- 넣을 것: 사람 이름(PERSON), 회사 · 기관 · 정부 부처 · 지자체 · 단체(ORG),
  제품 · 서비스 · 플랫폼 · 브랜드 이름(PRODUCT), 나라 · 도시 · 지명(GPE).
- 뺄 것: 직함과 회사 안 부서 이름, 일반 용어와 일반 약어(클라우드, 생성형 AI, 매출, AI, MSP, CSP, GPU, ERP).
- 글에 여러 번 나와도 한 번만 적는다. 띄어쓰기 · 대소문자 · 문장부호를 글과 똑같이 적는다. 같은 이름이 글에서 다르게
  적혔으면 둘 다 적는다. 조사는 떼고 이름만 적는다.

JSON 만 답한다: {"entities": [{"text": "...", "type": "PERSON|ORG|PRODUCT|GPE"}]}

자료:
"""

FIX_SYSTEM = "You correct proper names in Korean transcripts against a company style sheet. Follow the user message."
FIX_PROMPT = """아래 TRANSCRIPT 는 {company} 실적발표 컨퍼런스콜을 음성 인식으로 받아 적은 글이다.
NAMES 는 이 회사가 발표자료와 지난 컨퍼런스콜 스크립트에 쓴 이름 표기다(회사 표기 목록).

TRANSCRIPT 에서 NAMES 의 이름을 잘못 듣거나 다르게 적은 곳을 찾아 회사 표기로 고쳐라.
- 예: 영문으로 적는 이름을 한글로 적음('브리티 코파일럿' → 'Brity Copilot'), 음절을 잘못 들음('송혜구' → '송해구'),
  숫자로 적음('591' → '오구일').
- "old" 는 TRANSCRIPT 에서 그대로 옮긴 1~4 낱말이다. 이름 뒤에 붙은 조사는 빼고 이름 부분만 적는다.
- "new" 는 NAMES 에 있는 표기 하나를 글자 그대로 적는다. NAMES 에 없는 이름은 고치지 않는다.
- 소리와 문맥이 맞을 때만 고친다. 이미 회사 표기와 같은 이름, 일반 낱말, 숫자(금액 · 비율), 문법은 건드리지 않는다.
  확실하지 않으면 그대로 둔다.
- 같은 바꿈은 한 번만 적는다(받아쓰기에서 같은 old 가 여러 번 나와도).

JSON 만 답한다: {{"edits": [{{"old": "...", "new": "..."}}]}}

NAMES:
{names}

TRANSCRIPT:
{transcript}"""


def names_from_materials(text: str, runner: ClaudeSearch) -> list[str]:
    """자료 글의 고유명사(자료에 글자 그대로 있는 것만, 나온 순서대로)."""
    raw = runner.run(NAMES_PROMPT + text).text
    a, b = raw.find("{"), raw.rfind("}")
    try:
        items = json.loads(raw[a:b + 1]).get("entities", []) if a >= 0 else []
    except json.JSONDecodeError:
        return []
    names = [str(it.get("text", "")).strip() for it in items if isinstance(it, dict)]
    return list(dict.fromkeys(n for n in names if n and n in text))


def build_prompt(words: Sequence[Word], names: Sequence[str], company: str) -> str:
    return FIX_PROMPT.format(company=company, names="\n".join(names),
                             transcript=" ".join(w.text for w in words))


def verify(proposals: Sequence[Proposal], names: Sequence[str], materials: str, words: Sequence[Word]
           ) -> list[tuple[Proposal, str]]:
    """제안마다 (제안 — new 는 목록 표기로 맞춤, 'ok' 또는 버린 까닭)."""
    canon = {key(n): n for n in names}
    out = []
    for p in proposals:
        new = canon.get(key(p.new))
        if not p.old or not p.new.strip():
            out.append((p, "empty"))
        elif new is None or new not in materials:
            out.append((p, "not_in_names"))
        elif not find_spans(words, p.old):
            out.append((p, "old_not_in_transcript"))
        else:
            out.append((Proposal(p.old, new, "materials", ""), "ok"))
    return out


def propose(words: Sequence[Word], names: Sequence[str], materials: str, company: str, runner: ClaudeSearch
            ) -> tuple[list[tuple[Proposal, str]], Trace | None]:
    """세션 한 번: 받아쓰기 전체 + 이름 목록 → 제안 → 기계 확인. (확인 결과, 세션 기록)."""
    if not names:
        return [], None
    trace = runner.run(build_prompt(words, names, company))
    return verify(parse_proposals(trace.text), names, materials, words), trace

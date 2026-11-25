"""검수 1층의 '검색으로 확정' 제안자 — LLM 의 기억이 아니라 웹 검색 결과에 실제로 있는 철자로만 고친다.

    통화 전체 받아쓰기 → claude -p(도구는 WebSearch · WebFetch 만)가 잘못 들은 듯한 이름을 고르고 검색해 답한다
    → 기계 확인(verify) → 확정된 이름만 용어집이 되어 entity_fix 규칙 2~8(유사도 · 모양 · 흔한 낱말)을 그대로 거친다

용어집을 미리 만들 필요가 없다. 대신 LLM 이 '그럴듯한 다른 이름' 을 기억으로 지어내는 사고를 막아야 해서, 바꿀 말이
검색 결과에 있는지를 LLM 의 답이 아니라 도구가 돌려준 글로 확인한다.

기계 확인(verify)
1. 원래 말(old)이 받아쓰기에 있어야 한다(구간마다 entity_fix 규칙 2 가 다시 본다).
2. 근거 URL 이 이 세션의 검색 · 가져오기 결과에 실제로 나온 주소여야 한다.
3. 그 결과 글(제목 포함)에 바꿀 말(new)이 낱말 경계 그대로 있어야 한다(대소문자 · 문장부호만 무시).
4. 근거는 어닝콜 전사문이 아니어야 하고, 근거가 든 결과에 전사문 주소가 섞여 있어도 안 된다. 시험 자료(Earnings-21)는
   실제 통화라 그 통화의 전사문이 웹에 있다 — 거기서 베끼면 정답을 보고 고치는 셈이다. 새 통화를 받아 적을 때는
   전사문이 아직 없으므로, 이 규칙은 평가를 실제 쓰임에 맞추는 것이다(지난 분기 전사문까지 막아 조금 불리하게 잰다).
   검색 요약은 결과 여러 개를 한데 섞어 쓴 글이라 어느 주소에서 온 철자인지 가를 수 없다 — 그래서 결과 하나에라도
   전사문이 섞이면 그 결과 전체를 근거로 쓰지 않는다. 그때는 전사문이 아닌 페이지를 따로 가져와(WebFetch) 근거로 댄다.
5. 되풀이가 아니어야 한다. 검색 요약 · 가져오기 글은 도구 안의 모델이 쓴 글이라, LLM 이 검색어나 가져오기 지시에
   기억한 철자를 넣으면 그 철자를 그대로 되풀이할 수 있다. 그래서 바꿀 말이 그 도구 호출의 입력(검색어 · 지시)에
   있었다면 본문은 근거로 치지 않고, 검색 엔진이 돌려준 링크 제목에 있을 때만 인정한다.
세션이 전사문 주소를 본 것 자체도 `transcript_urls` 로 남긴다.

근거만 모자라 떨어진 제안(3~5)은 한 번 더 묻는다(`search_and_verify`): 그 제안만 모아 새 세션에 '전사문이 아닌
페이지를 직접 열어 확인하라' 고 하고, 그 답도 같은 기계 확인을 거친다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .entity_fix import Candidate, Edit, find_spans
from .words import Word

# 전사문 · 평가 자료를 싣는 곳. 그 밖의 곳은 주소나 제목에 'transcript' 가 있으면 전사문으로 본다.
TRANSCRIPT_HOSTS = ("seekingalpha.com", "fool.com", "rev.com", "github.com", "githubusercontent.com", "huggingface.co",
                    "kaggle.com", "insidermonkey.com", "alphastreet.com", "quartr.com", "gurufocus.com")
TOOLS = "WebSearch,WebFetch"

# 지시는 받아쓰기와 함께 표준 입력으로 넣는다 — 명령줄에는 줄바꿈 없는 한 줄만(Windows 명령줄 · .cmd 를 거쳐도 안전).
SYSTEM_PROMPT = "You verify the spelling of proper names in transcripts with web search. Follow the user message."
INSTRUCTIONS = f"""You check an automatic (speech recognition) transcript of a company earnings call for misheard \
proper names: people, companies, products, brands, places, tickers.

Work in two steps.
1. Read the whole TRANSCRIPT and pick the phrases that look like a misheard or misspelled proper name.
2. For each one, use WebSearch (and WebFetch if needed) to find how the name is really spelled, from sources about the \
company or the person: the company's own site, press releases, SEC filings, news, analyst or broker pages.

Rules
- Do not rely on memory. Propose a correction only when a search or fetch result you saw in this session shows the \
corrected spelling, and give that result's URL.
- Never use transcripts of earnings calls (this call or any other). Do not search for them, pass blocked_domains \
{json.dumps(list(TRANSCRIPT_HOSTS))} on every WebSearch, and ignore any transcript page that still shows up.
- Your answer is checked mechanically against the tool results, so make the evidence checkable:
  - A WebSearch result counts only if none of its links is a transcript, and the spelling is in a link title or, \
when your query did not contain that spelling, in the result text. Searching with the misheard words, the person's \
role or the company (for example "Monro CFO 2020") lets the results show the real spelling.
  - Otherwise WebFetch a page that is not a transcript (press release, SEC filing, company or investor-relations \
page, news article) with a prompt that does not contain the spelling, for example "List every person's full name \
and job title on this page, spelled exactly as written." Then give that page's URL.
- "old" must be copied exactly from TRANSCRIPT (1 to 4 words). "new" is the corrected name only.
- Change names only. Never change ordinary words, grammar, punctuation, numbers or filler words. A name that is \
already spelled correctly needs no edit. If unsure, leave it.
- Keep searches focused: one search can confirm several names (for example the company's management team, or the \
analysts who cover it).

Answer with JSON only, no other text:
{{"edits": [{{"old": "...", "new": "...", "url": "...", "evidence": "short quote showing the spelling"}}]}}"""


def build_prompt(words: Sequence[Word], company: str, quarter: str = "") -> str:
    when = f"fiscal quarter {quarter} earnings call, held in 2020" if quarter else "earnings call"
    return f"{INSTRUCTIONS}\n\nCOMPANY: {company} ({when})\n\nTRANSCRIPT:\n" + " ".join(w.text for w in words)


# 확인에 실패한 제안만 모아 한 번 더 묻는다 — 근거 페이지를 직접 열어(WebFetch) 확인하게. 받아쓰기는 다시 넣지 않는다.
RETRY_INSTRUCTIONS = f"""Earlier you proposed these spelling corrections for proper names in an automatic \
transcript of a company earnings call, but the evidence could not be checked: the search result you cited also \
listed earnings call transcripts, your query or prompt already contained the spelling, or the cited result did not \
show it.

For each item, confirm the spelling from one page that is not a transcript: WebFetch the cited URL if it is not a \
transcript, or find another page about the person or company (press release, SEC filing, company or \
investor-relations page, news article, analyst profile). Use a WebFetch prompt that does not contain the spelling, \
for example "List every person's full name and job title on this page, spelled exactly as written." Never use \
transcripts of earnings calls; pass blocked_domains {json.dumps(list(TRANSCRIPT_HOSTS))} on any WebSearch.

Keep "old" exactly as given. Drop an item if no page confirms it. Do not add new items.

Answer with JSON only, no other text:
{{"edits": [{{"old": "...", "new": "...", "url": "...", "evidence": "short quote showing the spelling"}}]}}"""


def build_retry_prompt(items: Sequence[Proposal], company: str) -> str:
    rows = [json.dumps({"old": p.old, "new": p.new, "url": p.url}, ensure_ascii=False) for p in items]
    return f"{RETRY_INSTRUCTIONS}\n\nCOMPANY: {company}\n\nITEMS:\n" + "\n".join(rows)


# ---------- claude -p 이 남긴 기록 읽기 ----------

_LINKS = re.compile(r"Links:\s*(\[.*?\])\s*(?:\n|$)", re.S)
_URL = re.compile(r"https?://[^\s\"'<>)\]]+")


@dataclass
class ToolResult:
    tool: str
    text: str  # 도구가 돌려준 글(검색은 결과 요약 · 제목, 가져오기는 페이지에서 뽑은 글)
    sources: list[tuple[str, str]]  # (주소, 제목)
    query: str = ""  # 그 도구 호출의 입력(검색어 · 가져오기 지시) — 되풀이 확인(verify 5)에 쓴다


@dataclass
class Trace:
    text: str = ""  # 마지막 답
    calls: list[dict] = field(default_factory=list)  # {"name", "input"}
    results: list[ToolResult] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    is_error: bool = False
    turns: int = 0


def _blocks(event: dict) -> list[dict]:
    content = (event.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_flatten(c.get("text", c.get("content", ""))) if isinstance(c, dict) else str(c)
                         for c in content)
    return "" if content is None else str(content)


def _sources(tool: str, tool_input: dict, text: str) -> list[tuple[str, str]]:
    if tool == "WebFetch":
        url = tool_input.get("url")
        return [(url, "")] if isinstance(url, str) else []
    m = _LINKS.search(text)
    if m:
        try:
            links = json.loads(m.group(1))
            return [(x["url"], str(x.get("title", ""))) for x in links if isinstance(x, dict) and "url" in x]
        except (json.JSONDecodeError, TypeError):
            pass
    return [(u, "") for u in dict.fromkeys(_URL.findall(text))]


def parse_stream(lines: Iterable[str]) -> Trace:
    """`claude -p --output-format stream-json --verbose` 의 줄들 → 도구 호출 · 도구 결과 · 마지막 답."""
    tr = Trace()
    pending: dict[str, tuple[str, dict]] = {}
    last_text = ""
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        if kind == "assistant":
            for b in _blocks(ev):
                if b.get("type") == "tool_use":
                    name, inp = str(b.get("name", "")), b.get("input") if isinstance(b.get("input"), dict) else {}
                    pending[str(b.get("id"))] = (name, inp)
                    tr.calls.append({"name": name, "input": inp})
                elif b.get("type") == "text" and b.get("text"):
                    last_text = b["text"]
        elif kind == "user":
            for b in _blocks(ev):
                if b.get("type") == "tool_result":
                    name, inp = pending.get(str(b.get("tool_use_id")), ("?", {}))
                    text = _flatten(b.get("content"))
                    query = inp.get("query") or inp.get("prompt") or ""
                    tr.results.append(ToolResult(name, text, _sources(name, inp, text), str(query)))
        elif kind == "result":
            tr.text = ev.get("result") if isinstance(ev.get("result"), str) else last_text
            tr.usage = ev.get("usage") or {}
            tr.cost_usd = ev.get("total_cost_usd")
            tr.is_error = bool(ev.get("is_error"))
            tr.turns = int(ev.get("num_turns") or 0)
    if not tr.text:
        tr.text = last_text
    return tr


# ---------- 기계 확인 ----------

def transcript_like(url: str, title: str = "") -> bool:
    host = (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in TRANSCRIPT_HOSTS):
        return True
    return "transcript" in url.lower() or "transcript" in title.lower()


def _url_key(url: str) -> str:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower().removeprefix("www.")
    return host + p.path.rstrip("/") + (f"?{p.query}" if p.query else "")


def _norm(text: str) -> str:
    return " " + " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split()) + " "


@dataclass(frozen=True)
class Proposal:
    old: str
    new: str
    url: str = ""
    evidence: str = ""


def parse_proposals(text: str) -> list[Proposal]:
    """답에서 JSON 을 꺼낸다(코드 울타리 · 앞뒤 말 허용). 모양이 틀리면 빈 목록 — 제안이 없는 것과 같다."""
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
            out.append(Proposal(it["old"].strip(), it["new"].strip(), str(it.get("url", "")).strip(),
                                str(it.get("evidence", ""))[:300]))
    return out


def transcript_urls(trace: Trace) -> list[str]:
    return list(dict.fromkeys(u for r in trace.results for u, t in r.sources if transcript_like(u, t)))


def verify(proposals: Sequence[Proposal], trace: Trace, words: Sequence[Word]) -> list[tuple[Proposal, str]]:
    """제안마다 (제안, 'ok' 또는 버린 까닭)."""
    out = []
    for p in proposals:
        target = _norm(p.new)
        if not p.old or not target.strip():
            why = "empty"
        elif not find_spans(words, p.old):
            why = "old_not_in_transcript"
        elif not p.url:
            why = "no_url"
        elif transcript_like(p.url):
            why = "source_is_transcript"
        else:
            seen = [r for r in trace.results if _url_key(p.url) in {_url_key(u) for u, _ in r.sources}]
            has = [r for r in seen if target in _norm(r.text + " " + _titles(r))]
            clean = [r for r in has if not any(transcript_like(u, t) for u, t in r.sources)]
            if not seen:
                why = "url_not_seen"
            elif not has:
                why = "not_in_search_results"
            elif not clean:
                why = "evidence_mixed_with_transcript"
            elif not any(_shows(r, target) for r in clean):
                why = "echo_of_query"
            else:
                why = "ok"
        out.append((p, why))
    return out


def _titles(r: ToolResult) -> str:
    return " ".join(t for _, t in r.sources)


def _shows(r: ToolResult, target: str) -> bool:
    """도구가 이 철자를 스스로 내놓았나(verify 5): 링크 제목에 있거나, 입력에 없던 철자가 본문에 있거나."""
    if target in _norm(_titles(r)):
        return True
    return target not in _norm(r.query) and target in _norm(_LINKS.sub(" ", r.text))


@dataclass
class FixedEditsProposer:
    """검색으로 확정한 바꿈 목록을 구간마다 그 구간에 든 것만 낸다(받아들일지는 entity_fix.validate 가 정한다)."""
    edits: Sequence[Edit]
    name: str = "search"

    def propose(self, seg: Sequence[Word], candidates: Sequence[Candidate]) -> list[Edit]:
        return [e for e in self.edits if find_spans(seg, e.old)]


# ---------- claude -p 부르기 ----------

_KEEP = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_GIT_BASH_PATH", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"}


def child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Claude Code 세션 안에서 부를 때 그 세션의 표시 변수를 지운다(평범한 터미널에서 부른 것과 같게)."""
    env = dict(os.environ if env is None else env)
    nested = "CLAUDECODE" in env
    for k in list(env):
        if k == "CLAUDECODE" or (k.startswith(("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_")) and k not in _KEEP):
            del env[k]
        elif nested and k == "ANTHROPIC_BASE_URL":
            del env[k]  # 데스크톱 앱이 제 세션에 건 프록시 — 안에서 띄운 claude 는 쓸 수 없다
    return env


Runner = Callable[[list[str], str, dict[str, str], float], str]


def _subprocess_runner(argv: list[str], stdin: str, env: dict[str, str], timeout: float) -> str:
    r = subprocess.run(argv, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                       timeout=timeout)
    if not (r.stdout or "").strip():
        raise RuntimeError(f"claude -p 가 아무것도 내지 않았다(종료 코드 {r.returncode}): {(r.stderr or '')[:300]}")
    return r.stdout


@dataclass
class ClaudeSearch:
    """claude -p(구독 로그인 그대로)로 검색 세션을 돌린다. 기록 전체를 저장해 다시 돌리면 호출 없이 같은 결과."""
    binary: str = "claude"
    model: str = "claude-sonnet-5-5"
    cache_dir: Path | None = None
    timeout: float = 1800.0
    runner: Runner = _subprocess_runner
    calls: int = 0
    hits: int = 0

    @property
    def name(self) -> str:
        return f"claude-p:{self.model}"

    def argv(self) -> list[str]:
        return [self.binary, "-p", "--output-format", "stream-json", "--verbose", "--model", self.model,
                "--tools", TOOLS, "--allowedTools", TOOLS, "--strict-mcp-config", "--setting-sources", "",
                "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT]

    def run(self, prompt: str) -> Trace:
        h = hashlib.sha256(json.dumps([self.model, SYSTEM_PROMPT, TOOLS, prompt], ensure_ascii=False).encode())
        path = self.cache_dir / f"{h.hexdigest()[:24]}.jsonl" if self.cache_dir else None
        if path and path.exists():
            self.hits += 1
            return parse_stream(path.read_text(encoding="utf-8").splitlines())
        self.calls += 1
        raw = self.runner(self.argv(), prompt, child_env(), self.timeout)
        trace = parse_stream(raw.splitlines())
        if trace.is_error:
            raise RuntimeError(f"claude -p 실패: {trace.text[:300]}")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw, encoding="utf-8")
        return trace


# ---------- 두 번에 걸친 검색 · 확인 ----------

# 근거만 모자라 떨어진 제안 — 페이지를 직접 열면 확인될 수 있다(받아쓰기에 없는 old 같은 것은 다시 묻지 않는다)
RETRYABLE = frozenset({"no_url", "source_is_transcript", "url_not_seen", "not_in_search_results",
                       "evidence_mixed_with_transcript", "echo_of_query"})


@dataclass
class SearchRound:
    trace: Trace
    checked: list[tuple[Proposal, str]]


def search_and_verify(search: ClaudeSearch, words: Sequence[Word], company: str, quarter: str = ""
                      ) -> list[SearchRound]:
    """1회: 통화 전체를 읽고 골라 검색. 2회(필요할 때만): 근거가 모자랐던 제안만 다시 확인.
    두 번째 답도 두 번째 세션의 도구 결과로만 확인하고, 처음에 없던 old 는 받지 않는다('not_asked')."""
    first = search.run(build_prompt(words, company, quarter))
    rounds = [SearchRound(first, verify(parse_proposals(first.text), first, words))]
    ok = {(p.old, p.new) for p, why in rounds[0].checked if why == "ok"}
    again = list(dict.fromkeys(p for p, why in rounds[0].checked if why in RETRYABLE and (p.old, p.new) not in ok))
    if again:
        second = search.run(build_retry_prompt(again, company))
        asked = {p.old for p in again}
        checked = [(p, verify([p], second, words)[0][1] if p.old in asked else "not_asked")
                   for p in parse_proposals(second.text)]
        rounds.append(SearchRound(second, checked))
    return rounds

"""검색 확정 제안자: claude -p 기록 읽기 · 기계 확인(검색 결과에 철자가 있나 · 전사문이 아닌가) · 저장본 · 환경 변수."""

import json

from callscribe.entity_fix import Edit, correct_entities
from callscribe.search_fix import (
    ClaudeSearch,
    FixedEditsProposer,
    Proposal,
    build_prompt,
    child_env,
    parse_proposals,
    parse_stream,
    transcript_like,
    transcript_urls,
    verify,
)
from callscribe.words import Word


def ws(text: str) -> list[Word]:
    return [Word(t, k, k + 0.5, 0.9, 0) for k, t in enumerate(text.split())]


SEARCH_TEXT = ('Web search results for query: "Monro Inc CFO"\n\n'
               'Links: [{"title":"Monro, Inc. Form 8-K","url":"https://www.sec.gov/Archives/x.htm"},'
               '{"title":"Monro Announces Leadership Transition","url":"https://www.globenewswire.com/n/1"}]\n\n'
               "Brian D'Ambrosia was Monro's Chief Financial Officer; Brett Ponton was CEO.")
MIXED_TEXT = ('Web search results for query: "Monro analysts"\n\n'
              'Links: [{"title":"Monro (MNRO) Q3 2020 Earnings Call Transcript","url":"https://www.fool.com/t/1"},'
              '{"title":"Monro stock","url":"https://example.com/mnro"}]\n\nAnalyst Scott Stember asked about tires.')


def stream(final: dict, results: list[tuple[str, dict, str]], is_error: bool = False) -> list[str]:
    lines = [json.dumps({"type": "system", "subtype": "init"})]
    for n, (tool, inp, text) in enumerate(results):
        lines.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"t{n}", "name": tool, "input": inp}]}}))
        lines.append(json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{n}", "content": [{"type": "text", "text": text}]}]}}))
    answer = json.dumps(final)
    lines.append(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}}))
    lines.append(json.dumps({"type": "result", "subtype": "success", "is_error": is_error, "result": answer,
                             "num_turns": len(results) + 1, "total_cost_usd": 0.12, "usage": {"output_tokens": 9}}))
    return lines


TRACE_LINES = stream({"edits": []}, [
    ("WebSearch", {"query": "Monro Inc CFO"}, SEARCH_TEXT),
    ("WebSearch", {"query": "Monro analysts"}, MIXED_TEXT),
    ("WebFetch", {"url": "https://ir.monro.com/team", "prompt": "names"}, "Maureen Mulholland, General Counsel"),
])


def test_parse_stream_collects_tool_results_sources_and_usage():
    tr = parse_stream(TRACE_LINES)
    assert [c["name"] for c in tr.calls] == ["WebSearch", "WebSearch", "WebFetch"]
    assert tr.results[0].sources[0] == ("https://www.sec.gov/Archives/x.htm", "Monro, Inc. Form 8-K")
    assert tr.results[2].sources == [("https://ir.monro.com/team", "")]  # 가져오기는 가져온 주소가 근거
    assert tr.cost_usd == 0.12 and tr.turns == 4 and not tr.is_error
    assert transcript_urls(tr) == ["https://www.fool.com/t/1"]


def test_transcript_like_by_host_path_or_title():
    assert transcript_like("https://seekingalpha.com/article/1-monro")
    assert transcript_like("https://www.nasdaq.com/articles/monro-q3-earnings-call-transcript")
    assert transcript_like("https://example.com/a", "Monro Q3 2020 Earnings Call Transcript")
    assert not transcript_like("https://www.sec.gov/Archives/x.htm", "Form 8-K")


def test_verify_accepts_only_spellings_the_tools_returned():
    tr = parse_stream(TRACE_LINES)
    words = ws("thanks Brian Dambrosia and Brett Ponten also Scott Stemmer and Maureen Mulhollen and Bob Smith")
    props = [
        Proposal("Brian Dambrosia", "Brian D'Ambrosia", "https://sec.gov/Archives/x.htm/"),  # 주소 모양만 다름
        Proposal("Brett Ponten", "Brett Ponton", "https://www.globenewswire.com/n/1"),
        Proposal("Scott Stemmer", "Scott Stember", "https://example.com/mnro"),  # 전사문과 섞인 결과에만 있음
        Proposal("Maureen Mulhollen", "Maureen Mulholland", "https://ir.monro.com/team"),
        Proposal("Bob Smith", "Bob Smyth", "https://www.sec.gov/Archives/x.htm"),  # 결과 글에 없는 철자 = 기억
        Proposal("Bob Smith", "Bob Smyth", "https://other.com/never-opened"),
        Proposal("Scott Stemmer", "Scott Stember", "https://www.fool.com/t/1"),
        Proposal("Jane Doe", "Jane Dough", "https://www.sec.gov/Archives/x.htm"),
    ]
    assert [why for _, why in verify(props, tr, words)] == [
        "ok", "ok", "evidence_mixed_with_transcript", "ok", "not_in_search_results", "url_not_seen",
        "source_is_transcript", "old_not_in_transcript"]


def test_spelling_must_match_on_word_boundaries():
    text = 'Links: [{"title":"Monro","url":"https://www.sec.gov/a"}]\n\nMonro United Kingdom operations.'
    tr = parse_stream(stream({"edits": []}, [("WebSearch", {"query": "q"}, text)]))
    # 'ted' 는 'United' 안에 있지만 낱말로는 없다 — 글자를 이어 붙여 찾으면 잘못 통과한다
    words = ws("we met Tad today")
    assert verify([Proposal("Tad", "Ted", "https://www.sec.gov/a")], tr, words)[0][1] == "not_in_search_results"


def test_fixed_edits_go_through_entity_rules():
    words = ws("our CFO Brian Dambrosio said the tires business is good")
    proposer = FixedEditsProposer([Edit("Brian Dambrosio", "Brian D'Ambrosia"), Edit("business", "Business Unit")])
    assert proposer.propose(words[:3], []) == []  # 구간에 old 가 없으면 내지 않는다
    out, rep = correct_entities(words, ["Brian D'Ambrosia", "Business Unit"], proposer)
    assert " ".join(w.text for w in out) == "our CFO Brian D'Ambrosia said the tires business is good"
    # 검색으로 확정한 이름이어도 entity_fix 규칙은 그대로 — 낱말을 더하기만 하는 바꿈은 버린다
    assert rep["decisions"] == {"ok": 1, "adds_or_drops_words": 1}


def test_parse_proposals_tolerates_fences_and_bad_items():
    text = 'Here:\n```json\n{"edits": [{"old": "a b", "new": "A B", "url": "u"}, {"old": 3}]}\n```'
    assert parse_proposals(text) == [Proposal("a b", "A B", "u", "")]
    assert parse_proposals("no json") == []


def test_claude_search_caches_whole_session_and_raises_on_error(tmp_path):
    seen = []

    def fake(argv, stdin, env, timeout):
        seen.append((argv, stdin))
        return "\n".join(TRACE_LINES)

    s = ClaudeSearch(cache_dir=tmp_path, runner=fake)
    prompt = build_prompt(ws("hello Monro"), "Monro Inc", "3")
    assert "COMPANY: Monro Inc (fiscal quarter 3 earnings call, held in 2020)\n\nTRANSCRIPT:\nhello Monro" in prompt
    a = s.run(prompt)
    b = s.run(prompt)
    assert len(seen) == 1 and s.hits == 1 and len(a.results) == len(b.results) == 3
    argv, stdin = seen[0]
    assert stdin == prompt  # 긴 받아쓰기는 명령줄이 아니라 표준 입력으로
    assert argv[argv.index("--tools") + 1] == "WebSearch,WebFetch"
    assert not any("\n" in x for x in argv)  # 명령줄에는 줄바꿈이 없다(지시는 표준 입력으로)

    def failing(argv, stdin, env, timeout):
        return "\n".join(stream({"edits": []}, [], is_error=True))

    try:
        ClaudeSearch(cache_dir=tmp_path, runner=failing).run("other prompt")
    except RuntimeError:
        pass
    else:
        raise AssertionError("실패한 세션은 오류로 멈춰야 한다")
    assert len(list(tmp_path.iterdir())) == 1  # 실패한 세션은 저장하지 않는다


def test_child_env_drops_session_markers_keeps_auth():
    env = child_env({"CLAUDECODE": "1", "CLAUDE_CODE_SIMPLE": "1", "CLAUDE_CODE_ENTRYPOINT": "x",
                     "ANTHROPIC_BASE_URL": "http://127.0.0.1:1", "CLAUDE_CODE_OAUTH_TOKEN": "t", "PATH": "p"})
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "t", "PATH": "p"}
    assert child_env({"ANTHROPIC_BASE_URL": "https://proxy"}) == {"ANTHROPIC_BASE_URL": "https://proxy"}

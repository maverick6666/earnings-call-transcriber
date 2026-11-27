"""한국 실적 콜: 회사 스크립트 → 정답 · 검색 확인의 한국어 처리 · 회사 스크립트 막기."""

import hashlib
import json

from callscribe.datasets import kr_reference, read_kr_script
from callscribe.search_fix import (
    SYSTEM_PROMPT,
    TOOLS,
    ClaudeSearch,
    Proposal,
    build_prompt,
    parse_stream,
    transcript_like,
    verify,
)
from callscribe.words import Word

SCRIPT = """삼성SDS 2025년 3분기 컨퍼런스 콜
안녕하십니까?
삼성SDS IR팀장 김민 상무입니다.

이번 분기 매출은 3조 3,913억원으로 전분기 대비 3.4%
감소했습니다. 영업이익률은 0.3%p 하락했습니다.
[ 클라우드 ]
안녕하십니까, 클라우드서비스사업부 이호준 부사장입니다.
[ Q&A 세션 ]
Q. 한화증권, 김소혜 위원
국가 AI 컴퓨팅센터 사업에 대해 설명 부탁드립니다.
A1. IR팀장, 김민 상무
A2. 클라우드서비스사업부, 이호준 부사장
OpenAI와는 두 가지를 진행 중입니다.
IR팀장, 김민 상무
감사합니다.
"""


def test_script_drops_title_and_speaker_labels_and_splits_qa():
    paras = read_kr_script(SCRIPT)
    assert paras == [
        ("talk", "안녕하십니까? 삼성SDS IR팀장 김민 상무입니다."),
        ("talk", "이번 분기 매출은 3조 3,913억원으로 전분기 대비 3.4% 감소했습니다. 영업이익률은 0.3%p 하락했습니다."),
        ("talk", "안녕하십니까, 클라우드서비스사업부 이호준 부사장입니다."),
        ("qa", "국가 AI 컴퓨팅센터 사업에 대해 설명 부탁드립니다."),
        ("qa", "OpenAI와는 두 가지를 진행 중입니다."),
        ("qa", "감사합니다."),
    ]


def test_reference_tags_names_longest_first_and_numbers_without_particles():
    ref = kr_reference(read_kr_script(SCRIPT), {"삼성SDS": "ORG", "SDS": "ORG", "김민": "PERSON", "OpenAI": "ORG"})
    tagged = [(r.text, r.tags) for r in ref if r.tags]
    assert ("삼성SDS", ("ORG",)) in tagged and ("SDS", ("ORG",)) not in tagged
    assert ("김민", ("PERSON",)) in tagged
    assert [t for t, tags in tagged if "CARDINAL" in tags] == ["3조", "3,913억원", "3.4%", "0.3%p"]
    assert ("OpenAI", ("ORG", "QA")) in tagged
    assert "으로" in [r.text for r in ref]  # 숫자 뒤 조사는 숫자 꼬리표 밖
    assert all("QA" in r.tags for r in ref[ref.index(next(r for r in ref if r.text == "국가")):])


def test_company_scripts_count_as_transcripts():
    sds = "https://image.samsungsds.com/kr/investor/ir_events/ear_rele/__icsFiles/afieldfile/2025/10/31/"
    assert transcript_like(sds + "Samsung%20SDS%202025Q3_Script_Kor.pdf")
    assert transcript_like(sds + "%EC%8B%A4%EC%A0%81%EB%B0%9C%ED%91%9C%20%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8.pdf")
    assert transcript_like("https://news.example.co.kr/a/1", "삼성SDS 컨콜 녹취록")
    deck = "SDS_3Q25%20%EC%8B%A4%EC%A0%81%EB%B0%9C%ED%91%9C%20%EC%8A%AC%EB%9D%BC%EC%9D%B4.pdf"  # 발표 슬라이드
    assert not transcript_like(sds + deck)
    assert not transcript_like("https://example.com/product-description")


def test_korean_names_are_checked_against_tool_results():
    words = [Word(t, k, k + 0.5) for k, t in enumerate(["이번", "분기", "이호중", "부사장이", "말했습니다"])]
    stream = [json.dumps(x, ensure_ascii=False) for x in (
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "WebSearch",
                                                       "input": {"query": "삼성SDS 클라우드서비스사업부장 부사장"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content":
            'Links: [{"title":"삼성SDS 클라우드서비스사업부장 이호준 부사장","url":"https://news.example.co.kr/1"}]\n'
            "요약"}]}},
        {"type": "result", "result": "{}", "num_turns": 2})]
    trace = parse_stream(stream)
    got = verify([Proposal("이호중", "이호준", "https://news.example.co.kr/1", ""),
                  Proposal("이호중", "이호선", "https://news.example.co.kr/1", "")], trace, words)
    assert [why for _, why in got] == ["ok", "not_in_search_results"]


def test_call_description_can_be_given_and_earnings21_prompt_is_unchanged():
    ws = [Word("안녕하십니까", 0, 1)]
    assert "COMPANY: Samsung SDS (2025Q3 call in Korean)\n\nTRANSCRIPT:\n안녕하십니까" in build_prompt(
        ws, "Samsung SDS", when="2025Q3 call in Korean")
    assert "(fiscal quarter 3 earnings call, held in 2020)" in build_prompt(ws, "Monro Inc", "3")


def test_runner_without_tools_and_search_cache_key_unchanged(tmp_path):
    seen = []

    def fake(argv, stdin, env, timeout):
        seen.append(argv)
        return json.dumps({"type": "result", "result": "{}", "num_turns": 1})

    ClaudeSearch(tools="", system_prompt="tag", runner=fake).run("x")
    assert seen[0][seen[0].index("--tools") + 1] == "" and "--allowedTools" not in seen[0]
    ClaudeSearch(cache_dir=tmp_path, runner=fake).run("hello")
    old_key = hashlib.sha256(json.dumps(["claude-sonnet-5-5", SYSTEM_PROMPT, TOOLS, "hello"]).encode()).hexdigest()
    assert (tmp_path / f"{old_key[:24]}.jsonl").exists()  # 저장해 둔 영어 검색 세션을 그대로 다시 쓴다

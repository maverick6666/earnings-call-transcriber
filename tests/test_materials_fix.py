"""회사 자료로 확정: 자료에 적힌 표기로만, 한글로 받아 적은 영문 이름은 소리로 견준다."""

import json

from callscribe import materials_fix
from callscribe.entity_fix import Edit, correct_entities, similarity
from callscribe.search_fix import ClaudeSearch, FixedEditsProposer
from callscribe.words import Word

MATERIALS = "참석패널 솔루션사업부장 송해구 부사장\nBrity Copilot 과 Cello Square 매출이 늘었습니다."


def answer(obj) -> callable:
    def fake(argv, stdin, env, timeout):
        fake.prompts.append(stdin)
        return json.dumps({"type": "result", "result": json.dumps(obj, ensure_ascii=False), "num_turns": 1})
    fake.prompts = []
    return fake


def test_names_are_kept_only_when_written_that_way_in_the_materials():
    runner = answer({"entities": [
        {"text": "송해구", "type": "PERSON"}, {"text": "Brity Copilot", "type": "PRODUCT"},
        {"text": "브리티 코파일럿", "type": "PRODUCT"}, {"text": "송해구", "type": "PERSON"}]})
    names = materials_fix.names_from_materials(MATERIALS, ClaudeSearch(tools="", runner=runner))
    assert names == ["송해구", "Brity Copilot"]  # 자료에 없는 표기 · 겹친 것은 버림


def test_proposals_must_use_a_name_from_the_list_and_old_must_be_in_the_transcript():
    words = [Word(t, k, k + 0.8) for k, t in enumerate(["송혜구", "부사장이", "브리티", "코파일럿을", "소개했습니다"])]
    runner = answer({"edits": [{"old": "송혜구", "new": "송해구"}, {"old": "브리티 코파일럿", "new": "brity copilot"},
                               {"old": "부사장이", "new": "부회장"}, {"old": "첼로", "new": "Cello Square"}]})
    names = ["송해구", "Brity Copilot", "Cello Square"]
    checked, trace = materials_fix.propose(words, names, MATERIALS, "삼성SDS", ClaudeSearch(tools="", runner=runner))
    assert [(p.new, why) for p, why in checked] == [
        ("송해구", "ok"), ("Brity Copilot", "ok"), ("부회장", "not_in_names"),
        ("Cello Square", "old_not_in_transcript")]
    assert "NAMES:\n송해구\nBrity Copilot\nCello Square" in runner.prompts[0]
    ok = [p for p, why in checked if why == "ok"]
    out, _ = correct_entities(words, [p.new for p in ok], FixedEditsProposer([Edit(p.old, p.new) for p in ok]))
    assert " ".join(w.text for w in out) == "송해구 부사장이 Brity Copilot을 소개했습니다"


def test_hangul_and_latin_spellings_are_compared_by_sound():
    assert similarity("브리티 코파일럿", "Brity Copilot") > 0.9
    assert similarity("엑센터", "AX센터") > 0.9
    assert similarity("클라우드", "Brity Copilot") < 0.6
    assert similarity("Citeva", "Cytiva") > 0.8  # 같은 글자 체계는 전처럼 글자 · 발음 열쇠로


def test_no_names_means_no_session():
    checked, trace = materials_fix.propose([Word("a", 0, 1)], [], MATERIALS, "x", ClaudeSearch(tools="", runner=None))
    assert checked == [] and trace is None

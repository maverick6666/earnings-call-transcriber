"""한국어 교정: 조사가 붙은 낱말 · 숫자로 받아 적힌 이름 · 정해 온 바꿈은 후보 찾기 없이."""

import json

from callscribe.entity_fix import Edit, attach_particle, correct_entities, match_spans, validate
from callscribe.search_fix import FixedEditsProposer, parse_proposals
from callscribe.words import Word


def ws(text: str, start: float = 0.0) -> list[Word]:
    return [Word(t, start + k, start + k + 0.8) for k, t in enumerate(text.split())]


def test_name_followed_by_a_particle_is_found_and_the_particle_is_kept():
    seg = ws("국가 구미씨의 데이터센터 구미씨스 구미씨")
    assert match_spans(seg, "구미씨") == [(1, 2, "의"), (4, 5, "")]  # '구미씨스' 의 '스' 는 조사가 아니다
    out, rep = correct_entities(seg, ["구미시"], FixedEditsProposer([Edit("구미씨", "구미시")]))
    assert [w.text for w in out] == ["국가", "구미시의", "데이터센터", "구미씨스", "구미시"]
    assert out[1].original == "구미씨의"


def test_particle_follows_the_new_names_final_consonant():
    assert attach_particle("송해구", "가") == "송해구가"
    assert attach_particle("오구일", "가") == "오구일이"  # 받침 있으면 '이'
    assert attach_particle("이호준", "는") == "이호준은"
    assert attach_particle("AX센터", "은") == "AX센터는"
    assert attach_particle("SDS", "가") == "SDS가"  # 라틴 글자 뒤는 읽는 소리를 몰라 그대로


def test_name_written_as_digits_only_when_the_digits_read_as_the_name():
    seg = ws("물류사업부장 591 부사장")
    ok = validate(seg, [Edit("591", "오구일")], ["오구일"])
    assert ok[0].ok and ok[0].spans == [(1, 2)]
    assert validate(seg, [Edit("591", "오백구십일")], ["오백구십일"])[0].ok  # 한자어 읽기
    assert validate(seg, [Edit("591", "오구이")], ["오구이"])[0].why == "touches_number"


def test_search_proposals_drop_a_particle_shared_by_old_and_new():
    answer = json.dumps({"edits": [
        {"old": "삼성 SS가", "new": "삼성 SDS가", "url": "u", "evidence": ""},
        {"old": "엑센터는", "new": "AX센터는", "url": "u", "evidence": ""},
        {"old": "경기두", "new": "경기도", "url": "u", "evidence": ""}]}, ensure_ascii=False)
    assert [(p.old, p.new) for p in parse_proposals(answer)] == [
        ("삼성 SS", "삼성 SDS"), ("엑센터", "AX센터"), ("경기두", "경기도")]  # '도' 는 이름의 일부


def test_fixed_edits_are_applied_even_when_the_right_name_is_already_in_the_segment():
    seg = ws("이호준 부사장이 답하고 이오준 부사장이 덧붙였습니다.")
    out, rep = correct_entities(seg, ["이호준"], FixedEditsProposer([Edit("이오준", "이호준")]))
    assert [w.text for w in out][3] == "이호준" and rep["asked"] == 1

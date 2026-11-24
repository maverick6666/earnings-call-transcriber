"""검수 1층: 후보 찾기 · 규칙 검사 · 적용(시각 · 원래 글) · 정답 대조 · LLM 답 읽기 · 저장본."""

import numpy as np

from callscribe.entity_fix import (
    Edit,
    FuzzyProposer,
    LLMProposer,
    apply_spans,
    common_vocabulary,
    correct_entities,
    edit_outcomes,
    find_candidates,
    jamo,
    parse_edits,
    segments,
    similarity,
    validate,
)
from callscribe.llm import CachedLLM
from callscribe.normalize import normalize_en
from callscribe.words import RefToken, Word

GLOSSARY = ["Cytiva", "Nvidia", "SK Hynix", "BMO", "Monro"]


def ws(text: str, t0: float = 0.0) -> list[Word]:
    return [Word(t, t0 + k, t0 + k + 0.5, 0.9, 0) for k, t in enumerate(text.split())]


def test_similarity_accepts_mishearings_and_rejects_unrelated():
    assert similarity("citeva", "Cytiva") >= 0.6 > similarity("star", "Cytiva")
    assert similarity("high nicks", "Hynix") > 0.6
    assert similarity("현대모비즈", "현대모비스") > 0.85 and jamo("현대") == "ㅎㅕㄴㄷㅐ"


def test_find_candidates_handles_split_words_and_marks_present():
    seg = ws("we sell to in video and Monro customers")
    cands = {c.term: c for c in find_candidates(seg, GLOSSARY)}
    assert cands["Nvidia"].old == "in video" and not cands["Nvidia"].present
    assert cands["Monro"].present and cands["Monro"].score == 1.0
    assert "SK Hynix" not in cands or cands["SK Hynix"].score < 0.6


def test_validate_rules():
    seg = ws("citeva and star grew while Monro, BMO and this held")
    decisions = validate(seg, [
        Edit("citeva", "cytiva"),  # 용어집 표기로 바뀌어야 함
        Edit("star", "Cytiva"),  # 안 비슷함
        Edit("grew", "Growth Corp"),  # 용어집에 없음
        Edit("Monro,", "Monro"),  # 문장부호만 다름
        Edit("BMO", "Monro"),  # 이미 아는 다른 이름
        Edit("this", "BMO"),  # 짧은 약어는 더 엄하게
        Edit("nowhere", "Nvidia"),  # 받아쓰기에 없음(유사도 검사 뒤라 not_similar 일 수도)
    ], GLOSSARY)
    why = [d.why for d in decisions]
    assert why[:6] == ["ok", "not_similar", "not_in_glossary", "case_or_spacing_only", "old_is_known_name",
                       "not_similar"]
    assert why[6] in ("old_not_found", "not_similar")
    assert decisions[0].new == "Cytiva" and decisions[0].spans == [(0, 1)]


def test_validate_finds_every_occurrence_without_overlap():
    seg = ws("in video beat and in video guided")
    (d,) = validate(seg, [Edit("in video", "Nvidia")], GLOSSARY)
    assert d.ok and d.spans == [(0, 2), (4, 6)]


def test_apply_keeps_time_original_and_punctuation():
    words = ws("we buy from in video, mostly")
    out, mapping = apply_spans(words, [(3, 5, "Nvidia")])
    assert [w.text for w in out] == ["we", "buy", "from", "Nvidia,", "mostly"]
    fixed = out[3]
    assert fixed.original == "in video," and fixed.start == words[3].start and fixed.end == words[4].end
    assert mapping == [(3, 5, 3, 4)]
    out2, _ = apply_spans(ws("the hynics ramp"), [(1, 2, "SK Hynix")])
    assert [w.text for w in out2] == ["the", "SK", "Hynix", "ramp"] and out2[2].original == ""
    assert out2[1].start < out2[2].start < out2[2].end <= 1.5


def test_correct_entities_with_fuzzy_baseline_and_outcomes():
    words = ws("revenue at citeva grew and in video demand stayed strong")
    ref = [RefToken(t) for t in ["revenue", "at", "Cytiva", "grew", "and", "Nvidia", "demand", "stayed", "strong"]]
    out, rep = correct_entities(words, GLOSSARY, FuzzyProposer(min_score=0.7))
    assert " ".join(w.text for w in out) == "revenue at Cytiva grew and Nvidia demand stayed strong"
    assert rep["asked"] == 1 and rep["decisions"]["ok"] == 2
    res = edit_outcomes(ref, words, out, rep["mapping"], normalize_en)
    assert res["fixed"] == 2 and res["broke"] == 0 and res["over_correction_rate"] == 0.0


def test_over_correction_is_counted():
    words = ws("our share price")
    ref = [RefToken(t) for t in ["our", "share", "price"]]
    out, rep = correct_entities(words, ["Shure"], FuzzyProposer(min_score=0.8))
    assert [w.text for w in out] == ["our", "Shure", "price"]  # 기준선은 흔한 낱말도 바꿔 버린다
    res = edit_outcomes(ref, words, out, rep["mapping"], normalize_en)
    assert res["broke"] == 1 and res["over_correction_rate"] == 1.0


def test_risky_mask_limits_what_is_sent():
    words = ws("our share price")
    out, rep = correct_entities(words, ["Shure"], FuzzyProposer(min_score=0.8), risky=np.zeros(3, dtype=bool),
                                ask_only_risky=True)
    assert rep["asked"] == 0 and [w.text for w in out] == ["our", "share", "price"]


def test_common_words_need_risk_evidence():
    common = common_vocabulary([["our", "share", "price", "in", "video"], ["share", "in", "video"]], min_docs=2)
    assert common == {"share", "in", "video"}
    words = ws("our share price and in video sales")
    no_risk = np.zeros(7, dtype=bool)
    out, rep = correct_entities(words, ["Shure", "Nvidia"], FuzzyProposer(min_score=0.7), risky=no_risk,
                                common=common)
    assert [w.text for w in out] == [w.text for w in words]
    assert rep["decisions"] == {"common_words_without_risk": 2}
    risk = no_risk.copy()
    risk[5] = True  # 'video' 가 위험 표시 → 'in video → Nvidia' 만 받아들임
    out, rep = correct_entities(words, ["Shure", "Nvidia"], FuzzyProposer(min_score=0.7), risky=risk, common=common)
    assert " ".join(w.text for w in out) == "our share price and Nvidia sales"


def test_adding_or_dropping_words_is_rejected():
    seg = ws("Voya financial and Eversource grew")
    why = [d.why for d in validate(seg, [Edit("financial", "Voya Financial"), Edit("Eversource", "Eversource Gas")],
                                   ["Voya Financial", "Eversource Gas"])]
    assert why == ["adds_or_drops_words", "adds_or_drops_words"]


def test_segments_cover_everything_once():
    words = ws(" ".join(f"w{k}." if k % 7 == 6 else f"w{k}" for k in range(200)))
    segs = segments(words, max_sec=45, min_sec=20)
    assert segs[0][0] == 0 and segs[-1][1] == 200
    assert all(a == b for (_, a), (b, _) in zip(segs, segs[1:], strict=False))
    assert all(words[b - 1].end - words[a].start <= 46 for a, b in segs)


def test_parse_edits_tolerates_fences_and_garbage():
    text = 'Sure.\n```json\n{"edits": [{"old": "citeva", "new": "Cytiva", "reason": "name"}, {"old": 3}]}\n```'
    assert parse_edits(text) == [Edit("citeva", "Cytiva", "name")]
    assert parse_edits("no json here") == [] and parse_edits("{broken") == []


class FakeLLM:
    name = "fake"

    def __init__(self, answer: str):
        self.answer, self.calls = answer, 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        assert "CANDIDATES" in user and "TRANSCRIPT" in user and "Cytiva" in user
        return self.answer


def test_llm_proposer_with_cache(tmp_path):
    fake = FakeLLM('{"edits": [{"old": "citeva", "new": "Cytiva"}, {"old": "grew", "new": "Grew Inc"}]}')
    llm = CachedLLM(fake, tmp_path)
    words = ws("revenue at citeva grew")
    out, rep = correct_entities(words, GLOSSARY, LLMProposer(llm))
    assert [w.text for w in out] == ["revenue", "at", "Cytiva", "grew"]
    assert rep["decisions"] == {"ok": 1, "not_in_glossary": 1}
    correct_entities(words, GLOSSARY, LLMProposer(llm))  # 같은 입력 → 저장본
    assert fake.calls == 1 and llm.hits == 1

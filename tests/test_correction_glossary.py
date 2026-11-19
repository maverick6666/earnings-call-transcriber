"""+L 교정 검사(지어낸 문장 버리기 · 원래 낱말 남기기) · +P 용어집."""

from callscribe.correction import apply_correction
from callscribe.glossary import extract_terms, hotwords, term_recall
from callscribe.words import RefToken, Word


def ws(text: str) -> list[Word]:
    return [Word(t, float(k), k + 0.5, 0.9) for k, t in enumerate(text.split())]


def test_replacement_keeps_time_and_original():
    out, rep = apply_correction(ws("our h b m revenue grew"), "our HBM revenue grew")
    texts = [w.text for w in out]
    assert texts == ["our", "HBM", "revenue", "grew"] and rep["replaced"] == 1 and rep["deleted"] == 2
    fixed = out[1]
    assert fixed.original in ("h", "b", "m") and fixed.start >= 1.0  # 시각은 원래 낱말 것


def test_sentence_level_addition_is_rejected():
    out, rep = apply_correction(ws("revenue grew strongly"),
                                "revenue grew strongly and we expect record margins next year")
    assert [w.text for w in out] == ["revenue", "grew", "strongly"]
    assert rep["rejected_insertions"] == 7 and rep["rejected_runs"] == ["and we expect record margins next year"]


def test_short_split_beside_a_fix_is_accepted_with_interpolated_time():
    out, rep = apply_correction(ws("the hbmthreee ramp"), "the HBM 3E ramp")
    assert [w.text for w in out] == ["the", "HBM", "3E", "ramp"] and rep["inserted"] == 1
    # 둘 중 하나는 바꾼 낱말(원래 글 남김), 하나는 더한 낱말(원래 글 없음 = ""); 시각은 앞뒤 사이
    assert sorted([out[1].original, out[2].original]) == ["", "hbmthreee"]
    assert out[0].end <= out[1].start <= out[2].start <= out[3].start


def test_unchanged_text_is_identity():
    words = ws("guidance for the second half")
    out, rep = apply_correction(words, "guidance for the second half")
    assert out == words and rep["kept"] == 5


def test_extract_terms_and_hotwords():
    docs = ["Samsung Foundry shipped HBM3E to Nvidia.", "HBM3E demand from Nvidia grew. Samsung Foundry expanded.",
            "The Galaxy line was flat."]
    terms = extract_terms(docs)
    assert {"HBM3E", "Nvidia", "Samsung Foundry"} <= set(terms) and "Galaxy" not in terms  # 한 번만 나온 말은 뺌
    assert len(hotwords(terms, max_chars=12)) <= 12


def test_extract_korean_terms():
    docs = ["당사는 HBM 과 파운드리 사업을 합니다.", "파운드리 매출과 HBM 판매가 늘었습니다."]
    terms = extract_terms(docs, language="ko")
    assert "파운드리" in terms and "HBM" in terms and "당사는" not in terms


def test_term_recall():
    ref = [RefToken(t) for t in ("Nvidia", "bought", "HBM3E")]
    hyp = [Word("in", 0, 1), Word("video", 1, 2), Word("bought", 2, 3), Word("HBM3E", 3, 4)]
    assert term_recall(ref, hyp, ["Nvidia", "HBM3E"]) == {"ref": 2, "correct": 1, "recall": 0.5}

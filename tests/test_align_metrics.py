"""정렬 · 정규화 · 점수."""

import random

import numpy as np
import pytest

from callscribe.align import align, edit_counts
from callscribe.metrics import confidence, error_rate, evaluate, hallucinations, near, op_times, tagged_error
from callscribe.normalize import normalize_en, normalize_ko_chars
from callscribe.words import RefToken, Word


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def test_alignment_matches_edit_distance_and_reconstructs_both_sides():
    rng = random.Random(7)
    for _ in range(400):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 14))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 14))]
        ops = align(a, b)
        c = edit_counts(ops)
        assert c["S"] + c["D"] + c["I"] == levenshtein(a, b)
        assert [a[i] for _, i, _ in ops if i is not None] == a
        assert [b[j] for _, _, j in ops if j is not None] == b


def test_banded_alignment_equals_full_alignment():
    rng = random.Random(3)
    ref = [f"w{rng.randint(0, 50)}" for _ in range(600)]
    hyp = [w for w in ref if rng.random() > 0.05] + ["x"] * 5
    full = edit_counts(align(ref, hyp))
    # 일부러 아주 좁은 띠로 시작해도 가장자리에 닿으면 넓혀서 같은 답을 낸다
    assert edit_counts(align(ref, hyp, band=3)) == full


@pytest.mark.parametrize(("text", "tokens"), [
    ("Good morning, everyone.", ["good", "morning", "everyone"]),
    ("year-over-year growth of 4.2%", ["year", "over", "year", "growth", "of", "4.2%"]),
    ("We're  “excited”", ["we're", "excited"]),
])
def test_normalize_en(text, tokens):
    assert normalize_en(text) == tokens


def test_normalize_ko_drops_spaces_and_punctuation():
    assert normalize_ko_chars("매출은 3조, 늘었다.") == list("매출은3조늘었다")


def test_error_rate_counts():
    r = error_rate(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
    assert (r["S"], r["D"], r["I"], r["rate"]) == (1, 0, 1, 0.5)


def test_deleted_words_get_interpolated_times():
    ref = [RefToken(t) for t in ("a", "b", "c", "d")]
    hyp = [Word("a", 0.0, 1.0), Word("d", 3.0, 4.0)]
    ops = align([r.text for r in ref], [w.text for w in hyp])
    t = op_times(ops, ref, hyp)
    assert t.tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5])


def test_near_points():
    assert near(np.array([0.0, 9.0, 10.5, 30.0]), [10.0, 20.0], 2.0).tolist() == [False, True, True, False]


def test_tagged_errors_use_tags_or_digits():
    ref = [RefToken("Samsung", tags=("ORG",)), RefToken("grew"), RefToken("17", tags=("CARDINAL",))]
    ops = align(["Samsung", "grew", "17"], ["Samsun", "grew", "17"])
    t = tagged_error(ops, ref)
    assert t["entity"] == {"ref": 1, "errors": 1, "rate": 1.0} and t["number"]["errors"] == 0
    untagged = [RefToken("revenue"), RefToken("4.2")]
    assert tagged_error(align(["revenue", "4.2"], ["revenue", "4"]), untagged)["number"]["rate"] == 1.0


def test_hallucination_runs_need_four_inserted_words_in_a_row():
    ops = align(["a", "b"], ["a", "thank", "you", "for", "watching", "b", "x"])
    h = hallucinations(ops, 2)
    assert (h["runs"], h["words"]) == (1, 4)


def test_boundary_error_isolates_errors_near_cut_edges():
    ref = [RefToken(f"w{k}", float(k), k + 0.5) for k in range(40)]
    hyp = [Word(f"w{k}", float(k), k + 0.5) for k in range(40) if k not in (19, 20)]  # 20초 자른 곳 양옆이 빠짐
    ev = evaluate(ref, hyp, cut_edges=[20.0])
    assert ev["overall"]["D"] == 2
    assert ev["near_cut_edges"]["errors"] == 2 and ev["away_from_cut_edges"]["errors"] == 0


def test_confidence_auc():
    hyp = [Word("a", 0, 1, 0.9), Word("x", 1, 2, 0.3), Word("c", 2, 3, 0.8), Word("y", 3, 4, 0.85)]
    ops = align(["a", "b", "c", "d"], [w.text for w in hyp])
    c = confidence(ops, hyp)
    # 맞음 {0.9, 0.8} · 틀림 {0.3, 0.85}: 4쌍 중 3쌍에서 맞은 쪽이 높다
    assert c["auc"] == 0.75 and (c["right"], c["wrong"]) == (2, 2)

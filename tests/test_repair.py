"""O1R: 말소리가 있는데 글자가 거의 없는 창(뭉개진 창)을 찾아 옮기기 · 힌트로 다시 받아 적는다."""

import numpy as np

from callscribe import runner
from callscribe.repair import collapsed
from callscribe.words import Word

DURATION = 120.0


def speech_everywhere(duration: float = DURATION):
    times = np.arange(0, duration, 0.03)
    return times, np.full(len(times), -20.0)


class WindowASR:
    """0.5초마다 '가나다'. 정해 둔 (구간, 힌트) 는 헛소리 두 낱말만(뭉개진 창), loop 는 같은 말을 잔뜩."""

    def __init__(self, broken=(), loops=()):
        self.broken = set(broken)
        self.loops = set(loops)
        self.calls = []
        self.name, self.version = "fake-window", "1"

    def transcribe(self, start, end, prompt=None):
        self.calls.append((start, end, prompt is not None))
        key = (round(start, 3), round(end, 3), prompt is not None)
        if key in self.broken:
            return [Word("기상캐스터", start + 1.0, start + 1.5), Word("배혜지", start + 1.5, start + 2.0)]
        text = "가나다라마바사아자차" if key in self.loops else "가나다"
        return [Word(text, t, t + 0.4) for t in np.arange(start, end - 0.4, 0.5)]


def test_collapsed_window_is_found_and_refilled_from_the_first_attempt_that_works():
    asr = WindowASR(broken={(15.0, 45.0, False), (20.0, 50.0, False)})  # 원래 창과 5초 옮긴 창이 뭉개짐
    plain = runner.run(runner.RunConfig("O1"), asr, DURATION, "c", energy=speech_everywhere())
    assert not [w for w in plain.words if 22.5 <= w.mid < 37.5]  # O1 은 맡은 15초가 빈다
    res = runner.run(runner.RunConfig("O1R"), asr, DURATION, "c", energy=speech_everywhere(), hint="통화 설명")
    (fix,) = res.manifest["repairs"]
    assert fix["window"] == [15.0, 45.0] and fix["accepted"] and fix["used"] == "prev_text"
    assert [t["attempt"] for t in fix["tried"]] == ["shift+5", "prev_text"]
    assert len([w for w in res.words if 22.5 <= w.mid < 37.5]) == 30
    assert asr.calls[-1] == (15.0, 45.0, True)  # 힌트는 그 창에만


def test_repetition_loop_is_not_taken_as_a_repair():
    asr = WindowASR(broken={(15.0, 45.0, False)}, loops={(20.0, 50.0, False)})
    res = runner.run(runner.RunConfig("O1R"), asr, DURATION, "c", energy=speech_everywhere())
    (fix,) = res.manifest["repairs"]
    assert fix["used"] == "prev_text"
    assert all(w.text == "가나다" for w in res.words if 22.5 <= w.mid < 37.5)


def test_healthy_call_is_left_alone_and_silence_is_not_collapse():
    times, db = speech_everywhere()
    db[(times >= 22.5) & (times < 37.5)] = -70.0  # 말소리 없는 구간 — 글자가 없어도 뭉개진 게 아니다
    asr = WindowASR(broken={(15.0, 45.0, False)})
    res = runner.run(runner.RunConfig("O1R"), asr, DURATION, "c", energy=(times, db))
    assert res.manifest["repairs"] == []
    assert collapsed([], (times, db)) == ([], 0.0)


def test_repairs_are_cached_with_the_windows(tmp_path):
    asr = WindowASR(broken={(15.0, 45.0, False), (20.0, 50.0, False)})
    first = runner.run(runner.RunConfig("O1R"), asr, DURATION, "c", cache_dir=tmp_path, energy=speech_everywhere())
    n = len(asr.calls)
    again = runner.run(runner.RunConfig("O1R"), asr, DURATION, "c", cache_dir=tmp_path, energy=speech_everywhere())
    assert len(asr.calls) == n and again.manifest["output_hash"] == first.manifest["output_hash"]
    assert any(".p" in p.name for p in next(tmp_path.iterdir()).iterdir())  # 힌트 준 결과는 이름에 해시

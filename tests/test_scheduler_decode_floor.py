#!/usr/bin/env python3
"""Apply overlay/patch_scheduler_decode_floor.py and test skip/cap/fair policy."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_scheduler_decode_floor.py",
        HERE.parent / "overlay" / "patch_scheduler_decode_floor.py",
    )
    if p.is_file()
)
SRC = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY_SRC",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Req:
    def __init__(self, rid: str, prompt: int, computed: int, tokens: int | None = None):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt if tokens is None else tokens
        self.num_output_placeholders = 0


class Sched:
    def __init__(self, running=None, waiting=None, skipped=None, step: int = 1):
        self.running = list(running or [])
        self.waiting = list(waiting or [])
        self.skipped_waiting = list(skipped or [])
        self.current_step = step
        self._glm53_align_prefill_limit = None
        self.requests = {
            r.request_id: r for r in self.running + self.waiting + self.skipped_waiting
        }


class Out:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = scheduled


def _load_patch_mod():
    spec = importlib.util.spec_from_file_location("glm53_decode_floor", PATCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_policy(patched_text: str):
    start = patched_text.index("class _Glm53MixedPrefill:")
    end = patched_text.index("_GLM53_MIXED = _Glm53MixedPrefill()")
    ns: dict = {"os": os, "time": __import__("time")}
    exec(patched_text[start:end], ns)
    return ns["_Glm53MixedPrefill"]


def _apply_v1(mod, text: str) -> str:
    helper = '''
def _glm53_mixed_prefill_policy(running, current):
    """v1 fixture."""
    return 0

'''
    needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    text = text.replace(needle, helper + needle, 1)
    text = text.replace(mod.RUNNING_OLD, mod.V1_RUNNING_NEW, 1)
    text = text.replace(mod.WAITING_OLD, mod.V1_WAITING_NEW, 1)
    return text


def _make_policy(cls, env: dict, clock: Clock | None = None):
    old = dict(os.environ)
    os.environ.update(env)
    try:
        return cls(now=clock)
    finally:
        os.environ.clear()
        os.environ.update(old)


def policy_tests(cls) -> None:
    clock = Clock()

    # skip: decoding peer withholds every prefill, including 2k vs 30k.
    p = _make_policy(cls, {"GLM53_MIXED_PREFILL_CHUNK": "skip"}, clock)
    dec = Req("A", 100, 100)
    b2k = Req("B2k", 2000, 0)
    b30k = Req("B30k", 30000, 0)
    sched = Sched(running=[dec], waiting=[b2k, b30k], step=1)
    for _ in range(10_000):
        sched.current_step += 1
        assert p.cap_for(sched, b2k) == 0
        assert p.cap_for(sched, b30k) == 0
    solo = Sched(running=[b30k], step=1)
    assert p.cap_for(solo, b30k) is None

    # off: no extra policy
    p = _make_policy(cls, {"GLM53_MIXED_PREFILL_CHUNK": "0"}, clock)
    assert p.cap_for(Sched(running=[dec], waiting=[b30k], step=2), b30k) is None

    # positive cap only when a peer is decoding
    p = _make_policy(cls, {"GLM53_MIXED_PREFILL_CHUNK": "128"}, clock)
    mixed = Sched(running=[dec], waiting=[b30k], step=3)
    assert p.cap_for(mixed, b30k) == 128
    assert mixed._glm53_align_prefill_limit == 128
    assert p.cap_for(Sched(running=[b30k], step=4), b30k) is None

    # fair: solo prefill uncapped; mixed first turn admits one aged prefill
    p = _make_policy(
        cls,
        {
            "GLM53_MIXED_PREFILL_CHUNK": "fair",
            "GLM53_FAIR_PREFILL_CHUNK": "256",
            "GLM53_FAIR_PREFILL_SHARE": "0.20",
            "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS": "2000",
            "GLM53_FAIR_PREFILL_MAX_CHUNKS": "1",
        },
        clock,
    )
    assert p.cap_for(Sched(running=[b30k], step=5), b30k) is None
    wait_b = Req("B", 30000, 0)
    wait_c = Req("C", 2000, 0)
    sched = Sched(running=[dec], waiting=[wait_b, wait_c], step=6)
    cap_b = p.cap_for(sched, wait_b)
    cap_c = p.cap_for(sched, wait_c)
    assert sorted([cap_b, cap_c]) == [0, 256], (cap_b, cap_c)
    winner = wait_b if cap_b == 256 else wait_c
    loser = wait_c if winner is wait_b else wait_b
    assert p.step_mode == "prefill_turn"
    p.note_scheduled(winner, 256)
    p.note_scheduled(loser, 0)
    # async in-flight: next schedule must not start another mixed turn
    clock.advance(0.05)
    sched.current_step = 7
    p.begin_step(sched)
    assert p.step_mode == "decode_only"
    assert p.cap_for(sched, loser) == 0
    # completed mixed step credits actual tokens only
    clock.advance(1.0)
    p.observe_output(sched, Out({winner.request_id: 256, "A": 8}))
    assert winner.request_id in p.last_service
    assert loser.request_id not in p.last_service
    # after completion, interval/share can admit the unserved request
    clock.advance(2.0)
    sched.current_step = 8
    p.begin_step(sched)
    assert p.cap_for(sched, loser) == 256
    assert p.cap_for(sched, winner) == 0

    # resume replay (computed >= prompt, remaining output) is prefill work
    replay = Req("R", 100, 100, tokens=5000)
    assert p.needs_prefill_compute(replay)
    assert not p.needs_prefill_compute(dec)

    # alignment: mixed 128 / threshold 3584 / block 3584 used to round to 0
    align = cls.aligned_new_tokens
    assert align(0, 128, 30000, 3584, 7168, None) == 0
    assert align(0, 128, 30000, 3584, 3584, None) == 0
    assert align(0, 128, 30000, 3584, 1024, None) == 128
    assert align(0, 1024, 30000, 3584, 7168, None) == 0
    assert align(0, 1024, 30000, 3584, 1024, None) == 1024
    assert align(0, 128, 30000, 3584, 3584, 128) == 128
    assert align(0, 128, 30000, 3584, 7168, 128) == 128
    assert align(0, 128, 30000, 1792, 7168, None) == 0
    assert align(0, 128, 30000, 1792, 1024, None) == 128

    # abort/finish drops policy state
    p.last_service["gone"] = 1.0
    p.arrival["gone"] = 1.0
    p._prune({"A", winner.request_id, loser.request_id})
    assert "gone" not in p.last_service


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", SRC))
    if not src.is_file():
        raise SystemExit(f"missing scheduler.py at {src}")
    mod = _load_patch_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        shutil.copyfile(src, dst)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(dst)
        env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        compile(text, "scheduler.py", "exec")
        mark_v2 = "# [glm53-decode-floor:v2]"
        assert mark_v2 in text
        assert text.count(mark_v2) >= 6
        assert "class _Glm53MixedPrefill:" in text
        assert "_GLM53_MIXED.begin_step(self)" in text
        assert "_GLM53_MIXED.observe_output(self, scheduler_output)" in text
        assert "_glm53_align_prefill_limit" in text
        assert "_glm53_mixed_prefill_policy(self, request)" in text
        assert "_glm53_mixed_prefill_policy(self.running, request)" not in text
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text() == text

        # v1 -> v2 migration on a fresh copy
        v1 = Path(tmp) / "scheduler_v1.py"
        shutil.copyfile(src, v1)
        v1.write_text(_apply_v1(mod, v1.read_text()))
        assert "def _glm53_mixed_prefill_policy(running, current):" in v1.read_text()
        env["GLM53_SCHEDULER_PY"] = str(v1)
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        v2 = v1.read_text()
        compile(v2, "scheduler_v1.py", "exec")
        assert mark_v2 in v2
        assert "def _glm53_mixed_prefill_policy(running, current):" not in v2
        assert "_glm53_mixed_prefill_policy(self.running, request)" not in v2
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert v1.read_text() == v2

        policy_tests(_extract_policy(text))
    print("scheduler decode-floor patch OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

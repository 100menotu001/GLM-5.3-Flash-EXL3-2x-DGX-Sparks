#!/usr/bin/env python3
"""Regression test for the InstantTensor KV-fit preflight note (#204).

The note is diagnostic only: it must fire for exactly the combination measured not to boot
(loader on, MAX_MODEL_LEN >= 850000, GPU_MEM_UTIL <= 0.85, no --kv-cache-memory-bytes), stay
silent otherwise, and never return non-zero, so it can never abort a boot under set -e.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def note_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 InstantTensor KV-fit note (begin)")
    end_marker = "# GLM53 InstantTensor KV-fit note (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def _run(load_format: str, max_model_len: str, util: str, extra_args: str) -> tuple[int, str]:
    script = (
        "set -euo pipefail\n"
        "warn() { printf 'NOTE:%s\\n' \"$*\" >&2; }\n"
        + note_source()
        + '\npreflight_instanttensor_kv_note "$1" "$2" "$3" "$4"\n'
        + 'printf "REACHED\\n"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "_", load_format, max_model_len, util, extra_args],
        capture_output=True, text=True, timeout=30,
    )
    assert "REACHED" in result.stdout, f"function aborted the script: {result.stderr}"
    return result.returncode, result.stderr


def test_note_fires_for_the_measured_failing_combination() -> None:
    rc, err = _run("instanttensor", "850000", "0.85", "")
    assert rc == 0
    assert "NOTE:" in err and "#204" in err
    assert "--kv-cache-memory-bytes 15032385536" in err, "must point at the .env.example line"
    # larger context and lower share are the same failure or worse
    assert "NOTE:" in _run("instanttensor", "900000", "0.85", "")[1]
    assert "NOTE:" in _run("instanttensor", "850000", "0.80", "")[1]
    # an unrelated EXTRA_ARGS entry does not count as a KV size
    assert "NOTE:" in _run("instanttensor", "850000", "0.85", "--no-async-scheduling")[1]


def test_note_is_silent_when_the_combination_is_not_the_failing_one() -> None:
    for case in (
        ("", "850000", "0.85", ""),                       # loader off
        ("instanttensor", "700000", "0.85", ""),          # below the 850k floor
        ("instanttensor", "850000", "0.88", ""),          # higher share (marginal, but not the stock case)
        ("instanttensor", "850000", "0.85", "--kv-cache-memory-bytes 15032385536"),
        ("instanttensor", "850000", "0.85", "--kv-cache-memory-bytes=15032385536"),
        ("instanttensor", "850000", "0.85", "--foo --kv-cache-memory-bytes 1 --bar"),
        ("instanttensor", "not-a-number", "0.85", ""),   # malformed input must not fire or fail
        ("instanttensor", "850000", "", ""),             # unset share (a sliced harness) must not fire or fail
        ("instanttensor", "850000", "abc", ""),          # malformed share likewise
    ):
        rc, err = _run(*case)
        assert rc == 0, case
        assert "NOTE:" not in err, case


def test_note_lives_inside_the_memory_guard_block() -> None:
    """test_nccl_multi_hca.py loads preflight's helpers by the memory-guard markers and runs
    preflight to completion; the note must be inside that block or preflight dies there."""
    source = START.read_text()
    gb = source.index("# GLM53 preflight memory guard (begin)")
    ge = source.index("# GLM53 preflight memory guard (end)")
    nb = source.index("# GLM53 InstantTensor KV-fit note (begin)")
    ne = source.index("# GLM53 InstantTensor KV-fit note (end)")
    assert gb < nb < ne < ge


def test_note_is_wired_into_preflight() -> None:
    """The call must sit in preflight() after the two memory checks, and nowhere else."""
    source = START.read_text()
    begin = source.index("preflight() {")
    end = source.index("\n}\n", begin)
    body = source[begin:end]
    assert body.count("preflight_instanttensor_kv_note ") == 1
    assert body.index('preflight_memory worker') < body.index("preflight_instanttensor_kv_note ")
    assert source.count("preflight_instanttensor_kv_note ") == 1, "called exactly once"


if __name__ == "__main__":
    test_note_fires_for_the_measured_failing_combination()
    test_note_is_silent_when_the_combination_is_not_the_failing_one()
    test_note_lives_inside_the_memory_guard_block()
    test_note_is_wired_into_preflight()
    print("instanttensor kv-fit note OK")

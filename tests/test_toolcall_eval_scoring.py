#!/usr/bin/env python3
"""CPU-only fixture tests for scripts/quality/toolcall_eval.py scoring.

Completions are built by hand and scored through the pure scorer: no endpoint, no model call and no
network. They pin the two multi-turn Edit cases to the requested file_path and the required
new_string (any Edit call whose old_string appeared in the file used to be enough), and pin
non-object argument payloads (null/array/number) to recorded invalid-argument failures instead of a
predicate crash.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUALITY = ROOT / "scripts" / "quality"
TOOLCALL = QUALITY / "toolcall_eval.py"

SPEC = importlib.util.spec_from_file_location("toolcall_eval", TOOLCALL)
te = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(te)

CASES = {c["id"]: c for c in te.cases()}
SRC = (ROOT / "overlay" / "exl3.py").read_text().splitlines()
# the line the edit_deep_18k prompt points at, read from the case itself so the fixture cannot drift
DEEP_LN = int(re.search(r"line (\d+)", CASES["edit_deep_18k"]["messages"][-1]["content"]).group(1))
DEEP_LINE = SRC[DEEP_LN - 1]


def completion(tool=None, arguments=None, content="", finish_reason="tool_calls"):
    """A /v1/chat/completions body as the endpoint returns it; arguments is the raw payload string."""
    tcs = [] if tool is None else [{"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}]
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tcs}, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": 12, "prompt_tokens": 4096}}


def scored(case_id, **kw):
    return te.score_completion(CASES[case_id], completion(**kw))


def check_edit_fixtures():
    """Right file and the required replacement, for both multi-turn Edit cases."""
    fix = {"file_path": "utils.py", "old_string": "    return items[-n+1:]", "new_string": "    return items[-n:]"}
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(fix))
    assert r["ok"] and not r["bad_json"], r
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, file_path="other.py")))
    assert not r["ok"], r
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(dict(fix, new_string="    return items[-n+2:]")))
    assert not r["ok"], r
    # an unrelated no-op edit of the same file used to score ok: old_string only had to be a substring
    noop = {"file_path": "utils.py", "old_string": '    """Return the last n items of the list."""',
            "new_string": '    """Return the last n items of the list."""'}
    r = scored("fix_after_read", tool="Edit", arguments=json.dumps(noop))
    assert not r["ok"], r

    deep = {"file_path": "overlay/exl3.py", "old_string": DEEP_LINE, "new_string": DEEP_LINE + "  # reviewed"}
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(deep))
    assert r["ok"] and not r["bad_json"], r
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(dict(deep, file_path="overlay/other.py")))
    assert not r["ok"], r
    r = scored("edit_deep_18k", tool="Edit", arguments=json.dumps(dict(deep, new_string=DEEP_LINE)))
    assert not r["ok"], r
    # appending the comment to the next line instead of the requested one used to score ok
    nxt = SRC[DEEP_LN]
    r = scored("edit_deep_18k", tool="Edit",
               arguments=json.dumps({"file_path": "overlay/exl3.py", "old_string": f"{DEEP_LINE}\n{nxt}",
                                     "new_string": f"{DEEP_LINE}\n{nxt}  # reviewed"}))
    assert not r["ok"], r
    print("edit fixtures OK (requested file_path and required new_string; wrong file/replacement fail)")


def check_invalid_arguments():
    """Null/array/number payloads are recorded failures; no checker runs and none of them raises."""
    r = scored("read_main", tool="Read", arguments="null")
    assert r["ok"] is False and r["bad_json"] is True and r["args"] == "null", r
    for raw in ("[]", "[1, 2]", "17", '"src/main.py"', "{not json"):
        r = scored("read_main", tool="Read", arguments=raw)
        assert r["ok"] is False and r["bad_json"] is True, (raw, r)
    # a checker that ignores the arguments (any Bash call) must not score a non-object payload ok either
    for raw in ("null", "[]"):
        r = scored("bash_after_long", tool="Bash", arguments=raw)
        assert r["ok"] is False and r["bad_json"] is True, (raw, r)
    print("invalid arguments OK (null/array/number recorded as failed samples, predicates untouched)")


def check_rescore_record():
    """--rescore rewrites from the recorded payload: a stored non-object payload stays a failed sample."""
    with tempfile.TemporaryDirectory() as d:
        stored = Path(d) / "stored.json"
        out = Path(d) / "out.json"
        stored.write_text(json.dumps({"results": [
            {"id": "read_main", "sample": 0, "tags": [], "ok": True, "why": "want Read", "tool": "Read",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": "null", "content": ""},
            {"id": "read_main", "sample": 1, "tags": [], "ok": True, "why": "want Read", "tool": "Read",
             "n_tool_calls": 1, "bad_json": False, "leak": False, "args": json.dumps({"file_path": "src/main.py"}),
             "content": ""},
        ]}))
        p = subprocess.run([sys.executable, str(TOOLCALL), str(out), "--rescore", str(stored)],
                           capture_output=True, text=True)
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        written = json.loads(out.read_text())
        assert [r["ok"] for r in written["results"]] == [False, True], written["results"]
        assert written["summary"]["total"]["bad_json"] == 1, written["summary"]["total"]
    print("rescore OK (stored null payload re-scores as a failed sample)")


def main() -> int:
    check_edit_fixtures()
    check_invalid_arguments()
    check_rescore_record()
    print("toolcall_eval scoring fixtures OK (file_path + required new_string; non-object arguments failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Wiring anchors and behaviour checks for the bring-up robustness patches.

start.sh is a generated-heredoc-heavy launcher, so the anchor tests below pin
the markers that keep the robustness behaviours wired (worker death detection,
revision-keyed sync marker, HF CLI fallback, worker cache writability
preflight). The lifecycle-lock and health-window checks run the launcher's real
function bodies against stubbed docker/ssh commands and real local flock
holders, so they cover the behaviour and not the source spelling.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Only ever advisory PID-file text: not this process, not the lock holder.
ADVISORY_PID = "424242"


def _source() -> str:
    return (ROOT / "start.sh").read_text()


def _function(name: str) -> str:
    """Slice a top-level ``name() { ... }`` definition out of start.sh."""
    src = _source()
    match = re.search(rf"^{re.escape(name)}\(\)\s*\{{", src, re.M)
    assert match, f"{name}() is missing from start.sh"
    line_end = src.index("\n", match.end())
    if src[match.start():line_end].rstrip().endswith("}"):
        return src[match.start():line_end] + "\n"  # one-liner (log/warn/die)
    end = src.index("\n}\n", match.end())
    return src[match.start():end + 3]


def _run_script(script: Path, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "USER": "glm53", **env}
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=cwd)


# ------------------------------- lifecycle lock -----------------------------

_LIFECYCLE_HEADER = """\
set -euo pipefail
RECORD="{tmp}/record"
: >"$RECORD"
LOGDIR="{tmp}/logs"
mkdir -p "$LOGDIR"
CLUSTER_LOCK="$LOGDIR/cluster.lock"
CLUSTER_LOCK_PID="$LOGDIR/cluster.lock.pid"
# The bound under test. Fast cases pass 1s so a contended stop does not cost
# 30s; the default is the launcher's shipped CLUSTER_LOCK_WAIT.
CLUSTER_LOCK_WAIT={wait}
CONTAINER_HEAD=glm53-exl3-head
CONTAINER_WORKER=glm53-exl3-worker
NFS_SHARE=0
docker() {{ printf 'docker %s\\n' "$*" >>"$RECORD"; }}
worker_ssh() {{ printf 'ssh %s\\n' "$*" >>"$RECORD"; }}
start_unlocked() {{
    # Independent flock attempt: proves the lock is held across the start arm.
    if flock -n "$CLUSTER_LOCK" true 2>/dev/null; then
        printf 'start_unlocked lock_free\\n' >>"$RECORD"
    else
        printf 'start_unlocked lock_held\\n' >>"$RECORD"
    fi
}}
banner() {{ :; }}
validate_numeric_config() {{ :; }}
configure_capture_sizes() {{ :; }}
validate_overlay_artifacts() {{ :; }}
"""


def _shipped_lock_wait() -> int:
    """The launcher's CLUSTER_LOCK_WAIT — the default bound these tests run."""
    match = re.search(r"^CLUSTER_LOCK_WAIT=(\d+)$", _source(), re.M)
    assert match, "CLUSTER_LOCK_WAIT is missing from start.sh"
    return int(match.group(1))


def _lifecycle_script(tmp: Path, command: str, wait_seconds: int) -> str:
    """start.sh's lifecycle dispatch with docker/ssh replaced by recorders."""
    functions = (
        "log", "warn", "die",
        "with_cluster_lock", "with_cluster_lock_for_stop",
        "stop_containers", "stop", "start", "main",
    )
    header = _LIFECYCLE_HEADER.format(tmp=tmp, wait=wait_seconds)
    return header + "".join(_function(name) for name in functions) + f"\nmain {command}\n"


def _run_lifecycle(tmp: Path, command: str, wait_seconds: int | None = None) -> subprocess.CompletedProcess:
    """Run one lifecycle command; default is the shipped lock-wait bound."""
    script = tmp / "lifecycle.sh"
    script.write_text(_lifecycle_script(tmp, command, wait_seconds or _shipped_lock_wait()))
    script.chmod(0o755)
    return _run_script(script, {}, tmp)


def _hold_lock(lock: Path, seconds: int = 300) -> subprocess.Popen:
    """Take the real kernel lock from a separate process."""
    holder = subprocess.Popen(["flock", "-x", str(lock), "sleep", str(seconds)])
    for _ in range(100):
        probe = subprocess.run(["flock", "-n", str(lock), "true"], capture_output=True)
        if probe.returncode != 0:
            return holder
        time.sleep(0.05)
    holder.kill()
    raise AssertionError("background flock holder never acquired the lock")


def _locked_logs(tmp: Path) -> tuple[Path, Path]:
    logs = tmp / "logs"
    logs.mkdir()
    lock = logs / "cluster.lock"
    lock.touch()
    pid_file = logs / "cluster.lock.pid"
    pid_file.write_text(f"{ADVISORY_PID}\n")
    return lock, pid_file


def test_lifecycle_commands_refuse_while_the_lock_is_held() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        lock, pid_file = _locked_logs(tmp)
        inode = lock.stat().st_ino
        holder = _hold_lock(lock)
        try:
            results = {cmd: _run_lifecycle(tmp, cmd, wait_seconds=1) for cmd in ("start", "restart", "stop")}
            assert holder.poll() is None, "the lock holder was signalled"
        finally:
            holder.kill()
            holder.wait()
        assert lock.stat().st_ino == inode, "a refused command replaced the lock file"
        for command, result in results.items():
            assert result.returncode == 1, (command, result.stdout, result.stderr)
            assert f"pid {ADVISORY_PID}" in result.stderr, (command, result.stderr)
            assert (tmp / "record").read_text() == "", f"{command} touched containers without the lock"
            assert pid_file.read_text() == f"{ADVISORY_PID}\n", f"{command} overwrote the holder's pid"
            assert "stopped." not in result.stdout, f"{command} reported a stop"
        assert "already running" in results["start"].stderr, results["start"].stderr
        assert "already running" in results["restart"].stderr, results["restart"].stderr
        assert "still held after 1s" in results["stop"].stderr, results["stop"].stderr
        assert "nothing was stopped" in results["stop"].stderr, results["stop"].stderr


def test_stop_takes_a_free_lock_and_removes_both_containers() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        _locked_logs(tmp)  # lock free, pid file stale/advisory only
        result = _run_lifecycle(tmp, "stop", wait_seconds=1)
        assert result.returncode == 0, result.stderr
        record = (tmp / "record").read_text().splitlines()
        assert "docker rm -f glm53-exl3-head" in record, record
        assert "ssh docker rm -f 'glm53-exl3-worker'" in record, record
        assert "stopped." in result.stdout, result.stdout


def test_restart_holds_one_lock_across_stop_and_start() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        _locked_logs(tmp)
        result = _run_lifecycle(tmp, "restart", wait_seconds=1)
        assert result.returncode == 0, result.stderr
        assert (tmp / "record").read_text().splitlines() == [
            "docker rm -f glm53-exl3-head",
            "ssh docker rm -f 'glm53-exl3-worker'",
            "start_unlocked lock_held",
        ]


# ------------------------------- health wait --------------------------------

# docker/ssh seam: docker inspect answers from a scripted sequence, so the test
# drives the real health loop without containers, ssh or a GPU.
_DOCKER_SEQUENCE_STUB = """\
docker() {
    case "${1:-}" in
        inspect)
            if [ "${2:-}" = "-f" ]; then
                INSPECTS=$(( ${INSPECTS:-0} + 1 ))
                printf '%s' "$INSPECTS" >"$TMP_DIR/inspects"
                if [ "$(sed -n "${INSPECTS}p" "$TMP_DIR/sequence")" = "true" ]; then
                    printf 'true\\n'
                else
                    printf 'false\\n'
                fi
                return 0
            fi
            printf 'running\\n'
            return 0
        ;;
        *) return 0 ;;
    esac
}
"""

# A running container whose inspect exits nonzero (grep -q closing the pipe
# early, docker hiccup): the old `docker inspect ... | grep -q true` pipeline
# turned that into "head container exited".
_DOCKER_NONZERO_EXIT_STUB = """\
docker() {
    case "${1:-}" in
        inspect)
            if [ "${2:-}" = "-f" ]; then
                printf 'true\\n'
                return 141
            fi
            return 0
        ;;
        *) return 0 ;;
    esac
}
"""


def _health_script(tmp: Path, docker_stub: str, ready_timeout: int) -> str:
    preamble = f"""\
set -euo pipefail
TMP_DIR="{tmp}"
READY_TIMEOUT={ready_timeout}
PORT=8888
CONTAINER_HEAD=glm53-exl3-head
CONTAINER_WORKER=glm53-exl3-worker
WORKER_SSH=glm53@10.0.0.2
curl() {{ return 1; }}
sleep() {{ :; }}
worker_ssh() {{ printf 'true\\n'; }}
{docker_stub}
"""
    body = "".join(_function(name) for name in ("log", "warn", "wait_for_health"))
    return preamble + body + '\nif wait_for_health; then printf "RESULT=healthy\\n"; else printf "RESULT=unhealthy\\n"; fi\n'


def _run_health(tmp: Path, docker_stub: str, ready_timeout: int) -> subprocess.CompletedProcess:
    script = tmp / "health.sh"
    script.write_text(_health_script(tmp, docker_stub, ready_timeout))
    script.chmod(0o755)
    return _run_script(script, {}, tmp)


def test_health_wait_needs_three_consecutive_head_misses() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        (tmp / "sequence").write_text("false\ntrue\nfalse\nfalse\nfalse\n")
        result = _run_health(tmp, _DOCKER_SEQUENCE_STUB, ready_timeout=120)
        assert "RESULT=unhealthy" in result.stdout, result.stdout
        # The true inspect resets the window, so the failure needs three more
        # misses: five inspects, not three.
        assert (tmp / "inspects").read_text() == "5", result.stdout
        assert "head container not running during startup" in result.stdout, result.stdout
        assert "3 consecutive checks" in result.stdout, result.stdout


def test_health_wait_does_not_report_a_running_head_as_dead() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        result = _run_health(tmp, _DOCKER_NONZERO_EXIT_STUB, ready_timeout=30)
        assert "RESULT=unhealthy" in result.stdout, result.stdout  # curl never succeeds
        assert "timed out" in result.stdout, result.stdout
        assert "exited/stopped" not in result.stdout, result.stdout


# ------------------------------- wiring anchors -----------------------------

def test_worker_death_detection_wired() -> None:
    src = _source()
    assert "worker_fail=0" in src, "worker death detection missing from wait_for_health"
    assert '[ "$worker_fail" -ge 3 ]' in src, "3-strike tolerance missing"
    assert "not running on ${WORKER_SSH}" in src, "worker death message missing"
    assert 'dead_side="worker"' in src and 'dead_side="head"' in src


def test_sync_revision_marker_wired() -> None:
    src = _source()
    assert ".glm53-exl3-synced" in src, "revision marker file missing"
    assert "FORCE_SYNC" in src, "FORCE_SYNC escape hatch missing"
    assert 'refs/main' in src, "marker must key on the snapshot commit (refs/main)"
    # both weights and DFlash2 go through the marker-checked helper
    assert src.count("sync_repo_to_worker ") >= 2


def test_hf_cli_fallback_wired() -> None:
    src = _source()
    assert "resolve_hf_bin()" in src, "resolve_hf_bin helper missing"
    assert src.count("resolve_hf_bin || die") == 3, "expected 3 call sites (weights, dflash, download-only)"
    assert "huggingface_hub.commands.huggingface_cli" in src, "python fallback missing"
    assert '"${HF_BIN_CMD[@]}" download' in src, "hf_download_repo must use the resolved array"


def test_worker_cache_writability_preflight_wired() -> None:
    src = _source()
    assert "worker cannot write $WORKER_CACHE_DIR/hub" in src
    assert "test -w '$WORKER_CACHE_DIR/hub'" in src


def test_check_port_free_detects_representative_listeners() -> None:
    """Inside double quotes, \\$ must expand to an end anchor, not a literal $."""
    src = _source()
    assert 'grep -qE "[:.]${port}\\$"' in src
    assert 'grep -qE "[:.]${port}\\\\$"' not in src

    script = r"""
set -euo pipefail
port=8000
printf '%s\n' '0.0.0.0:8000' '[::]:8000' | awk '{print $1}' | grep -qE "[:.]${port}\$"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    test_worker_death_detection_wired()
    test_sync_revision_marker_wired()
    test_hf_cli_fallback_wired()
    test_worker_cache_writability_preflight_wired()
    test_lifecycle_commands_refuse_while_the_lock_is_held()
    test_stop_takes_a_free_lock_and_removes_both_containers()
    test_restart_holds_one_lock_across_stop_and_start()
    test_health_wait_needs_three_consecutive_head_misses()
    test_health_wait_does_not_report_a_running_head_as_dead()
    test_check_port_free_detects_representative_listeners()
    print("start.sh bring-up robustness anchors OK")

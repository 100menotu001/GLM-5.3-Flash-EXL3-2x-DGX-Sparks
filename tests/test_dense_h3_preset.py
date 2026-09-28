"""CPU-only tests for GLM53_MODEL_PRESET=dense-h3: staging, pre-stop refusal, build guards."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_tool = load("stage_dense_h3", "tools/stage_dense_h3.py")
overlay_tool = load("dense_overlay", "tools/dense_overlay.py")
profile = load("pack_profile", "tools/pack_profile.py")
fixtures = load("test_pack_profile", "tests/test_pack_profile.py")


def hf_repo(tmp_path):
    """Reuse the pack-profile fixture, reshaped as an HF repo + an unstaged overlay build."""
    hub, target, draft = fixtures.stage(tmp_path)
    repo = target.parent.parent
    base = repo / "snapshots" / ("b" * 40)
    base.mkdir()
    (repo / "blobs").mkdir()
    weights = target / "model.safetensors"
    blob = repo / "blobs" / hashlib.sha256(weights.read_bytes()).hexdigest()
    shutil.move(weights, blob)
    (base / "model.safetensors").symlink_to(Path("../../blobs") / blob.name)
    (repo / "refs/main").write_text(base.name)
    (base / "provenance").mkdir()  # real TR3 snapshots carry sidecar directories
    (base / "provenance/receipt.json").write_text("{}")
    overlay = repo / "snapshots" / ".build"
    overlay_tool.link_pack(str(base), str(overlay), "dense.safetensors")
    cfg = json.loads((target / "config.json").read_text())
    del cfg["glm53_profile"]
    (overlay / "config.json").write_text(json.dumps(cfg))
    shutil.copyfile(target / "model.safetensors.index.json", overlay / "model.safetensors.index.json")
    shutil.rmtree(target)
    return hub, repo, overlay, draft


def test_staged_pair_passes_the_profile_validator(tmp_path):
    hub, repo, overlay, draft = hf_repo(tmp_path)
    link = os.readlink(overlay / "model.safetensors")
    assert link.startswith("../../blobs/"), link  # relative: resolves inside the container mount
    draft_rev = stage_tool.stage_draft(draft, hub, "local/draft")
    assert re.fullmatch(r"[0-9a-f]{40}", draft_rev)
    assert (hub / "models--local--draft/refs/main").read_text() == draft_rev
    assert stage_tool.stage_draft(draft, hub, "local/draft") == draft_rev  # same bytes, same id

    rev = stage_tool.stage_target(overlay, hub, "local/draft", draft_rev, "glm53-dense-h3", {"k": "v"})
    assert re.fullmatch(r"[0-9a-f]{40}", rev) and not overlay.exists()
    assert (repo / "refs/glm53-dense-h3").read_text() == rev
    assert (repo / "refs/main").read_text() == "b" * 40
    result = profile.resolve(repo / "snapshots" / rev, hub, {"GLM53_DENSE_FP8": "all"}, set())
    assert result["DFLASH_MODEL"] == "local/draft"
    assert result["DFLASH_REVISION"] == draft_rev
    assert result["GLM53_DENSE_EXL3"] == "1"


def test_target_revision_tracks_the_draft(tmp_path):
    hub, _repo, overlay, draft = hf_repo(tmp_path)
    draft_rev = stage_tool.stage_draft(draft, hub, "local/draft")
    first = stage_tool.revision(overlay)
    stage_tool.stage_target(overlay, hub, "local/draft", draft_rev, "r", {})
    assert first != stage_tool.revision(_repo / "snapshots" / (_repo / "refs/r").read_text())


def launcher_functions(*names):
    source = (ROOT / "start.sh").read_text()
    return "\n".join(re.search(rf"(?ms)^{n}\(\) \{{\n.*?^\}}", source).group() for n in names)


def test_restart_refuses_an_unbuilt_pair_before_stopping(tmp_path):
    hub, repo, overlay, draft = hf_repo(tmp_path)
    script = launcher_functions("select_dense_h3", "main") + '''
set -eu
die() { printf 'DIE %s\\n' "$*"; exit 1; }
log() { :; }; banner() { :; }; resolve_pack_profile() { :; }; validate_numeric_config() { :; }
configure_capture_sizes() { :; }; validate_overlay_artifacts() { :; }; with_cluster_lock() { :; }
stop_containers() { printf 'STOP\\n'; }
start() { printf 'START %s\\n' "${MODEL_SNAPSHOT:-}"; }
start_unlocked() { printf 'START %s\\n' "${MODEL_SNAPSHOT:-}"; }
main "$CMD"
'''
    env = {"PATH": os.environ["PATH"], "GLM53_MODEL_PRESET": "dense-h3", "DENSE_H3_REF": "glm53-dense-h3",
           "MODEL_PATH": str(tmp_path / "missing"), "MODEL_CACHE_NAME": "models--missing",
           "FALLBACK_MODEL_PATH": str(repo), "MODEL_FALLBACK_CACHE_NAME": repo.name, "MODEL_SNAPSHOT": ""}

    def run(cmd):
        return subprocess.run(["bash", "-c", script], env={**env, "CMD": cmd}, capture_output=True, text=True)

    result = run("restart")
    assert result.returncode == 1 and "STOP" not in result.stdout and "not built" in result.stdout
    assert run("start").stdout.splitlines() == ["START "]  # start builds after download
    draft_rev = stage_tool.stage_draft(draft, hub, "local/draft")
    rev = stage_tool.stage_target(overlay, hub, "local/draft", draft_rev, "glm53-dense-h3", {})
    assert run("restart").stdout.splitlines() == ["STOP", f"START {rev}"]  # fallback repo adopted
    env["GLM53_MODEL_PRESET"] = ""
    assert run("restart").stdout.splitlines() == ["STOP", "START "]


def build_script(tmp_path, repo, docker_running):
    return launcher_functions("select_dense_h3", "build_dense_h3") + f'''
set -eu
die() {{ printf 'DIE %s\\n' "$*"; exit 1; }}
log() {{ :; }}
resolve_hf_bin() {{ HF_BIN_CMD=(true); }}
docker() {{ {"echo abc123" if docker_running else ":"}; }}
python3() {{ printf 'PY %s\\n' "$2"; }}
bash() {{ printf 'QUANT\\n'; }}
GLM53_MODEL_PRESET=dense-h3 TP=2 HF_CACHE_DIR={tmp_path} MODEL_PATH={repo} MODEL_CACHE_NAME={repo.name}
FALLBACK_MODEL_PATH={repo} MODEL_FALLBACK_CACHE_NAME={repo.name} MODEL_SNAPSHOT= CONTAINER_HEAD=glm53-exl3-head
IMAGE=img SCRIPT_DIR={ROOT}
''' + "\n".join(line for line in (ROOT / "start.sh").read_text().splitlines()
                if line.startswith("DENSE_H3_")) + "\nbuild_dense_h3\n"


def test_build_refuses_a_foreign_base_checkpoint(tmp_path):
    _hub, repo, _overlay, _draft = hf_repo(tmp_path)
    result = subprocess.run(["bash", "-c", build_script(tmp_path, repo, False)], capture_output=True, text=True)
    assert "DIE dense-h3 builds on the pinned TR3 4-bpw pack" in result.stdout


def repinned_build(tmp_path, docker_running):
    """The real TR3 sidecars are not fixtures; re-pin the base check to a stand-in."""
    _hub, repo, _overlay, _draft = hf_repo(tmp_path)
    base = repo / "snapshots" / ("b" * 40)
    for name in ("config.json", "model.safetensors.index.json"):
        (base / name).write_text("{}")
    empty = hashlib.sha256(b"{}").hexdigest()
    script = re.sub(r"^(DENSE_H3_TR3_(CONFIG|INDEX)_SHA256)=\w+$", rf"\1={empty}",
                    build_script(tmp_path, repo, docker_running), flags=re.M)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout


def test_build_refuses_the_gpu_step_under_a_live_serve(tmp_path):
    out = repinned_build(tmp_path, True)
    assert "DIE dense-h3: quantizing the draft needs the head GPU" in out
    assert "QUANT" not in out


def test_build_refuses_a_draft_that_misses_the_pin(tmp_path):
    out = repinned_build(tmp_path, False)
    assert "QUANT" in out
    assert "does not match the pinned recipe" in out
    assert "dense_overlay" not in out  # refused before the target is fetched


def test_newest_snapshot_fallback_skips_the_built_target(tmp_path):
    hub, repo, overlay, draft = hf_repo(tmp_path)
    draft_rev = stage_tool.stage_draft(draft, hub, "local/draft")
    rev = stage_tool.stage_target(overlay, hub, "local/draft", draft_rev, "glm53-dense-h3", {})
    os.utime(repo / "snapshots" / ("b" * 40), (1, 1))  # the built target is the newest entry
    script = launcher_functions("newest_snapshot") + f"""
DENSE_H3_REF=glm53-dense-h3
newest_snapshot {repo}
rm {repo}/refs/glm53-dense-h3
newest_snapshot {repo}
"""
    lines = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout.splitlines()
    assert lines == ["b" * 40, rev]


def test_quant_script_mounts_the_whole_hf_repo(tmp_path):
    snap = tmp_path / "hub/models--x--draft/snapshots/rev1"
    snap.mkdir(parents=True)
    (tmp_path / "hub/models--x--draft/blobs").mkdir()
    (tmp_path / "hub/models--x--draft/blobs/c").write_text("{}")
    (snap / "config.json").symlink_to("../../blobs/c")
    (tmp_path / "build/src/exllamav3").mkdir(parents=True)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$LOG"\n')
    (fake / "docker").chmod(0o755)
    log = tmp_path / "docker.log"
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "LOG": str(log),
           "DRAFT_SNAP": str(snap), "BUILD": str(tmp_path / "build"), "OUT": str(tmp_path / "out"), "IMG": "img"}
    subprocess.run(["bash", str(ROOT / "tools/dflash2_exl3_quant.sh")], env=env, check=True, capture_output=True)
    args = log.read_text().splitlines()
    assert f"{tmp_path}/hub/models--x--draft:/draft:ro" in args
    assert "DRAFT_IN=/draft/snapshots/rev1" in args
    assert any("--src /draft/snapshots/rev1" in a for a in args)

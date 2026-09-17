"""Offline provenance and artifact hashes; no Torch/CUDA import needed."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root):
    data = json.loads((root / "PROVENANCE.json").read_text())
    for relative, expected in data["upstreams"][1]["header_sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"vendored pinned header changed: {relative}")


def create(root):
    contract = {"abi": 2, "hidden": 4096, "intermediate_local": 2048,
                "topk": 8, "experts_max": 96, "row_capacities": [32, 64],
                "counter_lengths": {"32": 5635, "64": 11267}, "params_size": 344, "activation_limit": 10.0}
    paths = [root / name for name in ("runtime.py", "dispatch_policy.json", "cooperative_moe.so", "PROVENANCE.json", "toolchain.txt")]
    paths += sorted(q for q in (root / "source").rglob("*") if q.is_file())
    paths += sorted(q for q in (root / "headers").rglob("*") if q.is_file())
    data = {"schema": 1, "contract": contract,
            "files": {str(q.relative_to(root)): digest(q) for q in paths},
            "gpu_validation": "pending; build success is not correctness or performance proof"}
    (root / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    action, directory = sys.argv[1:]
    {"verify-sources": verify_sources, "create": create}[action](Path(directory))

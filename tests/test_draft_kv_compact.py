"""CPU-only draft-page tests; no torch import, model, or serving process.

Set GLM53_VLLM_SRC to pristine vLLM 487ecf187d3dfe74d2cf6119a92881dba403c219
sources to include real grouping, tensor allocation, backend-selection, and
prefix-cache lookup tests. Only the five files in PINS under vllm/v1 are
needed. The fixtures are hash-checked and copied before patching.
"""
from __future__ import annotations

import abc
import ast
import collections
import collections.abc
import copy
import dataclasses
import enum
import hashlib
import importlib.util
import itertools
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import typing

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "draft_patch", ROOT / "overlay/patch_glm5_drafter_group.py"
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)
ENV = "GLM53_DRAFT_KV_COMPACT"
PINS = {
    "core/kv_cache_utils.py": "f1c553daa9f214e03126fe30de9c89e764b0a5168464e66fd9f739e168ffffc7",
    "worker/utils.py": "3dcd6ad34ee1d1db2875f7f7dd51d90ee0e64041ab282180687770a38b26acb1",
    "kv_cache_interface.py": "54a761dd60907945c8f3bfc450b1264033013b83b204320581961f887b03e5b0",
    "core/kv_cache_coordinator.py": "f640b5c42f6bc718926329a8f8b48a0fd0c8c8f24368af4e4b9ec8d13f68432f",
    "core/single_type_kv_cache_manager.py": "41043976d1d5e38e0465c8004fc04e01f35f66b39a40099c62d79b3756337d00",
}
HYBRID_PATCH = ROOT / "overlay/patch_hybrid_prefix_hit.py"
MLA_BLOCK = 3584
WINDOW = 2048


def test_compact_block_preserves_alignment_and_fits_page():
    ns = {"os": os, "SlidingWindowSpec": type("SlidingWindowSpec", (), {})}
    exec(patch.COMPACT_BLOCK_HELPER, ns)
    choose = ns["_glm53_draft_block_size"]
    # Sweep non-power-of-two MLA pages and different TP-local draft widths.
    for mla_block in range(64, 8193, 64):
        for bytes_per_token in (512, 1024, 2048, 3072, 4096):
            page = mla_block * 656
            if page < 64 * bytes_per_token:
                with pytest.raises(ValueError):
                    choose(mla_block, page, bytes_per_token, True)
                continue
            block = choose(mla_block, page, bytes_per_token, True)
            assert block % 64 == 0
            assert math.lcm(mla_block, block) == mla_block
            assert block * bytes_per_token <= page
            legal = [b for b in range(64, mla_block + 1, 64)
                     if mla_block % b == 0 and b * bytes_per_token <= page]
            assert block == max(legal)
    assert choose(3584, 3584 * 656, 2048, True) == 896
    assert choose(4608, 4608 * 656, 2048, True) == 1152
    # Off keeps the 64-token page whatever the geometry.
    assert choose(3584, 3584 * 656, 2048, False) == 64
    for geometry in ((65, 65536, 512), (0, 65536, 512), (64, 65536, 0)):
        with pytest.raises(ValueError):
            choose(*geometry, True)
        assert choose(*geometry, False) == 64


def speculative(method, draft_layers=5):
    simple = types.SimpleNamespace
    return None if method is None else simple(
        use_dflash=lambda: method == "dflash",
        draft_model_config=simple(hf_config=simple(num_hidden_layers=draft_layers)),
    )


def test_compact_preflight_env_and_identity(monkeypatch):
    swa = type("SlidingWindowSpec", (), {})
    ns = {"os": os, "SlidingWindowSpec": swa}
    exec(patch.COMPACT_BLOCK_HELPER, ns)
    preflight = ns["_glm53_draft_kv_compact"]
    dflash = types.SimpleNamespace(speculative_config=speculative("dflash"))
    specs = {f"draft.{i}": swa() for i in range(5)}
    monkeypatch.delenv(ENV, raising=False)
    assert preflight(dflash, specs) is False
    for bad in ("", "auto", "true", "01"):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError, match="0 or 1"):
            preflight(dflash, specs)
    monkeypatch.setenv(ENV, "1")
    assert preflight(dflash, specs) is True
    # Nothing to gate without sliding-window layers (e.g. a stage without
    # the drafter); a KpoolTailSpec subclass is not an exact match.
    assert preflight(types.SimpleNamespace(speculative_config=None), {}) is True
    tail = type("KpoolTailSpec", (swa,), {})
    assert preflight(types.SimpleNamespace(speculative_config=None), {"t": tail()}) is True
    for config in (
        types.SimpleNamespace(speculative_config=speculative("mtp")),
        types.SimpleNamespace(speculative_config=None),
        types.SimpleNamespace(speculative_config=speculative("dflash", 6)),
    ):
        with pytest.raises(ValueError, match="DFlash drafter"):
            preflight(config, specs)
    monkeypatch.setenv(ENV, "0")
    for config in (
        types.SimpleNamespace(speculative_config=speculative("mtp")),
        types.SimpleNamespace(speculative_config=None),
    ):
        assert preflight(config, specs) is False


@pytest.fixture
def sources(tmp_path):
    root = os.environ.get("GLM53_VLLM_SRC")
    if not root:
        pytest.skip("set GLM53_VLLM_SRC for pinned-source CPU integration")
    for rel, sha in PINS.items():
        data = (Path(root) / "vllm/v1" / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, f"source drift: {rel}"
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return tmp_path


def definitions(path, ns, names=None):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and (names is None or n.name in names)]
    tree = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)


@pytest.fixture
def allocator(sources, monkeypatch):
    patch.patch_file(str(sources / "core/kv_cache_utils.py"))
    module = types.ModuleType("draft_cache_cpu_fixture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    ns.update({
        "dataclass": dataclasses.dataclass, "fields": dataclasses.fields,
        "replace": dataclasses.replace, "Enum": enum.Enum, "IntEnum": enum.IntEnum,
        "Counter": collections.Counter, "defaultdict": collections.defaultdict,
        "prod": math.prod, "math": math, "copy": copy, "os": os,
        "cast": typing.cast, "get_dtype_size": lambda dtype: dtype,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "round_up": lambda a, b: (a + b - 1) // b * b,
        "MambaAttentionBackendEnum": types.SimpleNamespace(MAMBA2="mamba2"),
        "logger": logging.getLogger(__name__),
        "MultipleOf": dataclasses.make_dataclass("MultipleOf", [("base", int)]),
    })
    definitions(sources / "kv_cache_interface.py", ns)
    ns["KVCacheSpecRegistry"] = types.SimpleNamespace(
        get_uniform_type_base_spec=lambda spec: type(spec)
    )
    definitions(sources / "core/kv_cache_utils.py", ns, {
        "_glm53_draft_kv_compact", "_glm53_draft_block_size",
        "get_kv_cache_groups", "is_kv_cache_type_attention_free",
        "is_kv_cache_spec_uniform", "_get_kv_cache_groups_uniform_spec",
        "group_and_unify_kv_cache_specs", "_get_kv_cache_groups_glm5_next",
        "_pp_balanced_mamba_group_count", "create_kv_cache_group_specs",
        "_glm5_next_tensor_layout", "_pool_bytes_per_block",
        "get_kv_cache_config_from_groups", "may_override_num_blocks",
    })
    definitions(sources / "worker/utils.py", ns, {
        "select_common_block_size", "prepare_kernel_block_sizes",
    })
    return ns


def make_specs(ns, block=3584, draft_heads=4, target=True):
    specs = {}
    for i in range(11 if target else 0):
        specs[f"mla.{i}"] = ns["MLAAttentionSpec"](
            block_size=block, num_kv_heads=1, head_size=576,
            dtype=1, cache_dtype_str="fp8_ds_mla",
        )
        specs[f"indexer.{i}"] = ns["MLAAttentionSpec"](
            block_size=block, num_kv_heads=1, head_size=128,
            dtype=2, compress_ratio=4,
        )
        specs[f"tail.{i}"] = ns["KpoolTailSpec"](
            block_size=4, num_kv_heads=1, head_size=128,
            dtype=2, sliding_window=4,
        )
    for i in range(34 if target else 0):
        specs[f"mamba.{i}"] = ns["MambaSpec"](
            block_size=block, shapes=((1024,),), dtypes=(2,),
        )
    for i in range(5):
        specs[f"draft.{i}"] = ns["SlidingWindowSpec"](
            block_size=64, num_kv_heads=draft_heads, head_size=128,
            dtype=2, sliding_window=2048,
        )
    return specs


def make_groups(ns, block=3584, draft_heads=4, method="dflash", draft_layers=5):
    """The allocator entry point (every grouping path) on the GLM-5-Next layout."""
    simple = types.SimpleNamespace
    config = simple(
        parallel_config=simple(pipeline_parallel_size=1),
        cache_config=simple(num_gpu_blocks_override=None),
        scheduler_config=simple(disable_hybrid_kv_cache_manager=False),
        speculative_config=speculative(method, draft_layers),
    )
    groups = ns["get_kv_cache_groups"](config, make_specs(ns, block, draft_heads))
    cache = ns["get_kv_cache_config_from_groups"](config, groups, 16 * 1024**3)
    draft = next(iter(groups[-1].kv_cache_spec.kv_cache_specs.values()))
    return groups, cache, draft


def test_reservation_reduction_without_extra_allocation(allocator, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    old_groups, old_cache, old_draft = make_groups(allocator)
    monkeypatch.setenv(ENV, "1")
    groups, cache, draft = make_groups(allocator)
    assert old_draft.max_admission_blocks_per_request(2048, 262144) == 65
    assert draft.max_admission_blocks_per_request(2048, 262144) == 6
    assert cache.num_blocks == old_cache.num_blocks
    assert cache.kv_cache_tensors == old_cache.kv_cache_tensors
    assert groups[:-1] == old_groups[:-1]
    assert math.lcm(3584, draft.block_size) == 3584
    # Reject a page that fits in bytes but increases prefix-cache alignment.
    malformed = dataclasses.replace(draft, block_size=1024)
    groups[-1] = allocator["KVCacheGroupSpec"](
        ["draft.0"], allocator["UniformTypeKVCacheSpecs"].from_specs({"draft.0": malformed})
    )
    assert allocator["_glm5_next_tensor_layout"](groups) is None


# Exact fit: MLA page 4096 * 656 = 2,686,976 bytes; 41 heads * 128 * 2 * 2 =
# 20,992 draft bytes/token -> 128-token page, no padding (SIX finding 1).
EXACT_FIT = dict(block=4096, draft_heads=41)
NOT_DFLASH = (("mtp", 5), (None, 5), ("dflash", 6))


def test_compact_pages_require_positively_identified_dflash(allocator, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    # Padded (deployed) and exact-fit geometries fail closed alike: the
    # preflight runs before either branch is chosen.
    for geometry in ({}, EXACT_FIT):
        for method, draft_layers in NOT_DFLASH:
            with pytest.raises(ValueError, match="DFlash drafter"):
                make_groups(allocator, method=method, draft_layers=draft_layers, **geometry)
    assert make_groups(allocator)[2].block_size == 896
    exact = make_groups(allocator, **EXACT_FIT)[2]
    assert exact.block_size == 128 and exact.page_size_padded is None
    # Off never gates: padded geometry keeps the 64-token page and an exact
    # fit keeps its pre-existing contiguous page (the coordinator's boundary
    # lookup stays off for both).
    monkeypatch.setenv(ENV, "0")
    for method, draft_layers in NOT_DFLASH + (("dflash", 5),):
        assert make_groups(allocator, method=method, draft_layers=draft_layers)[2].block_size == 64
        exact = make_groups(allocator, method=method, draft_layers=draft_layers, **EXACT_FIT)[2]
        assert exact.block_size == 128 and exact.page_size_padded is None


def test_compact_preflight_covers_non_glm_grouping_paths(allocator, monkeypatch):
    """A drafter-only spec takes the generic uniform path, not the GLM-5-Next
    fast path; the flag must still fail closed there."""
    simple = types.SimpleNamespace
    specs = make_specs(allocator, target=False)
    for method, draft_layers in (("dflash", 5),) + NOT_DFLASH:
        config = simple(
            scheduler_config=simple(disable_hybrid_kv_cache_manager=False),
            speculative_config=speculative(method, draft_layers),
        )
        monkeypatch.setenv(ENV, "0")
        assert len(allocator["get_kv_cache_groups"](config, dict(specs))) == 1
        monkeypatch.setenv(ENV, "1")
        if method == "dflash" and draft_layers == 5:
            assert len(allocator["get_kv_cache_groups"](config, dict(specs))) == 1
        else:
            with pytest.raises(ValueError, match="DFlash drafter"):
                allocator["get_kv_cache_groups"](config, dict(specs))


def test_backend_split_guard_and_unpadded_exception(allocator, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    groups, cache, draft = make_groups(allocator)
    cache.kv_cache_groups = [groups[-1]]

    def select(supported):
        backend = type("Backend", (), {
            "get_supported_kernel_block_sizes": staticmethod(lambda: supported)
        })
        return allocator["prepare_kernel_block_sizes"](
            cache, [[types.SimpleNamespace(backend=backend)]]
        )

    assert select([allocator["MultipleOf"](16)]) == [896]
    with pytest.raises(ValueError, match="cannot be split"):
        select([64])
    # Exact-fit pages are contiguous and can still be virtually split.
    unpadded = dataclasses.replace(draft, page_size_padded=None)
    cache.kv_cache_groups = [allocator["KVCacheGroupSpec"](["draft.0"], unpadded)]
    assert select([64]) == [64]
    # Even zero extra padding selects the strided reshape when the field is set.
    explicit_stride = dataclasses.replace(
        draft, page_size_padded=draft.real_page_size_bytes
    )
    cache.kv_cache_groups = [
        allocator["KVCacheGroupSpec"](["draft.0"], explicit_stride)
    ]
    with pytest.raises(ValueError, match="cannot be split"):
        select([64])


def test_patch_is_idempotent_and_preflights_both_files(sources):
    kv = sources / "core/kv_cache_utils.py"
    worker = sources / "worker/utils.py"
    original = (kv.read_bytes(), worker.read_bytes())
    patch.patch_file(str(kv), dry_run=True)
    assert (kv.read_bytes(), worker.read_bytes()) == original
    worker.write_text(worker.read_text().replace(
        "selected_kernel_size = select_common_block_size(",
        "selected_kernel_size = changed_selector(",
    ))
    drifted = worker.read_bytes()
    with pytest.raises(AssertionError):
        patch.patch_file(str(kv))
    assert kv.read_bytes() == original[0]
    assert worker.read_bytes() == drifted
    worker.write_bytes(original[1])
    patch.patch_file(str(kv))
    applied = (kv.read_bytes(), worker.read_bytes())
    patch.patch_file(str(kv))
    assert (kv.read_bytes(), worker.read_bytes()) == applied


# ---------------------------------------------------------------------------
# Prefix-cache lookup: the pinned HybridKVCacheCoordinator (with
# overlay/patch_hybrid_prefix_hit.py applied) and the pinned single-type
# managers, driven by the allocator's real group layout. Hashing, caching,
# and lookup use the real code; only the block pool store is a dict.

class FakeBlockPool:
    def __init__(self, num_gpu_blocks=0, enable_caching=True, hash_block_size=64, **_):
        self.hash_block_size = hash_block_size
        self.null_block = types.SimpleNamespace(block_id=-1, is_null=True, ref_cnt=0)
        self.cached = {}
        self.view = None  # BlockHashListWithBlockSize, bound by the fixture
        self._next_id = 0

    def new_block(self):
        self._next_id += 1
        return types.SimpleNamespace(
            block_id=self._next_id, is_null=False, ref_cnt=0, block_hash=None
        )

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        blocks = [self.cached.get((block_hash, gid)) for gid in kv_cache_group_ids]
        return None if None in blocks else blocks

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks,
                          block_size, kv_cache_group_id, block_mask):
        hashes = request.block_hashes
        if block_size != self.hash_block_size:
            hashes = self.view(hashes, self.hash_block_size, block_size)
        for i in range(num_cached_blocks, num_full_blocks):
            if block_mask is None or block_mask[i - num_cached_blocks]:
                blocks[i].block_hash = (hashes[i], kv_cache_group_id)
                self.cached[blocks[i].block_hash] = blocks[i]

    def evict(self, block):
        self.cached = {k: v for k, v in self.cached.items() if v is not block}


@pytest.fixture
def prefix_hits(allocator, sources):
    coordinator = sources / "core/kv_cache_coordinator.py"
    subprocess.check_call(
        [sys.executable, str(HYBRID_PATCH)],
        env={**os.environ, "GLM53_KV_COORDINATOR_PY": str(coordinator)},
    )
    ns = allocator
    ns.update({
        "ABC": abc.ABC, "abstractmethod": abc.abstractmethod,
        "ClassVar": typing.ClassVar, "Sequence": collections.abc.Sequence,
        "NamedTuple": typing.NamedTuple, "itertools": itertools,
        "overload": typing.overload,
        "BlockPool": FakeBlockPool,
        "envs": types.SimpleNamespace(VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None),
    })
    definitions(sources / "core/kv_cache_utils.py", ns, {
        "BlockHashListWithBlockSize", "resolve_block_hashes",
        "generate_scheduler_kv_cache_config",
    })
    definitions(sources / "core/single_type_kv_cache_manager.py", ns, {
        "SingleTypeKVCacheManager", "FullAttentionManager", "KpoolTailManager",
        "SlidingWindowManager", "MambaManager",
    })

    def manager_for(kv_cache_spec, max_in_flight_tokens, max_model_len, **kwargs):
        cls = {
            "SlidingWindowSpec": ns["SlidingWindowManager"],
            "KpoolTailSpec": ns["KpoolTailManager"],
            "MambaSpec": ns["MambaManager"],
        }.get(type(kv_cache_spec).__name__, ns["FullAttentionManager"])
        return cls(kv_cache_spec, **kwargs)

    ns["get_manager_for_kv_cache_spec"] = manager_for
    definitions(coordinator, ns, {
        "_glm53_inner_kv_spec", "_glm53_is_draft_swa_spec",
        "_glm53_dflash_swa_replay_tokens", "_glm53_dflash_replay_safe_hit",
        "_glm53_dflash_boundary_lookup_enabled",
        "_validate_prefix_cache_retention_interval", "SpecGroup",
        "KVCacheCoordinator", "HybridKVCacheCoordinator",
    })
    return ns


class Cache:
    """One scheduler-side coordinator over the allocator's real groups."""

    def __init__(self, ns, cache_config, swa_retention=None):
        """``swa_retention`` mirrors VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA
        (overlay/patch_apc_per_group_retention.py): None = dense, 0 = only the
        replay-boundary window. Target groups stay dense here."""
        scheduler = ns["generate_scheduler_kv_cache_config"]([cache_config])
        sizes = [g.kv_cache_spec.block_size for g in scheduler.kv_cache_groups
                 if g.kv_cache_spec.participates_in_prefix_caching]
        self.hash_block = math.gcd(*sizes)
        self.coordinator = ns["HybridKVCacheCoordinator"](
            kv_cache_config=scheduler, max_model_len=262144,
            max_in_flight_tokens=2048, use_eagle=True, enable_caching=True,
            enable_kv_cache_events=False, dcp_world_size=1, pcp_world_size=1,
            scheduler_block_size=math.lcm(*sizes), hash_block_size=self.hash_block,
        )
        self.pool = self.coordinator.block_pool
        self.pool.view = ns["BlockHashListWithBlockSize"]
        self.draft_gid = len(scheduler.kv_cache_groups) - 1
        draft = scheduler.kv_cache_groups[self.draft_gid].kv_cache_spec
        assert type(draft).__name__ == "SlidingWindowSpec"
        self.draft_block = draft.block_size
        swa = self.coordinator.single_type_managers[self.draft_gid]
        cache_blocks = swa.cache_blocks
        swa.cache_blocks = lambda request, num_tokens, retention_interval=None: (
            cache_blocks(request, num_tokens, retention_interval=swa_retention)
        )
        self.requests = 0

    def request(self, tokens):
        self.requests += 1
        hashes, prev = [], b""
        for i in range(len(tokens) // self.hash_block):
            prev = hashlib.blake2b(
                prev + tokens[i * self.hash_block:(i + 1) * self.hash_block],
                digest_size=16,
            ).digest()
            hashes.append(prev)
        return types.SimpleNamespace(
            request_id=f"req{self.requests}", num_prompt_tokens=len(tokens),
            num_tokens=len(tokens), block_hashes=hashes, shared_prefix_boundary=None,
        )

    def prefill(self, tokens, num_computed=None):
        """Run a request to ``num_computed`` tokens and cache like the scheduler."""
        request = self.request(tokens)
        num_computed = len(tokens) if num_computed is None else num_computed
        for manager in self.coordinator.single_type_managers:
            blocks = manager.req_to_blocks[request.request_id]
            while len(blocks) < num_computed // manager.block_size:
                blocks.append(self.pool.new_block())
        self.coordinator.cache_blocks(request, num_computed)
        return request

    def lookup(self, tokens):
        """``KVCacheManager.get_computed_blocks``: the last token is recomputed."""
        request = self.request(tokens)
        blocks, hit, self.uncached = self.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        assert len(blocks[0]) * MLA_BLOCK == hit
        return blocks, hit

    def draft_blocks(self, request):
        return self.coordinator.single_type_managers[self.draft_gid].req_to_blocks[
            request.request_id
        ]


def tokens(n, seed=0):
    return random.Random(seed).randbytes(n)


def layout(prefix_hits, monkeypatch, compact, swa_retention=None):
    monkeypatch.setenv(ENV, "1" if compact else "0")
    _, cache_config, _ = make_groups(prefix_hits)
    return Cache(prefix_hits, cache_config, swa_retention)


LIVE_PROMPT = 100701       # the A/B/A 100k reuse prompt
LIVE_BOUNDARY = 100352     # 28 x 3584, the MLA/mamba hit in every phase


def test_live_reuse_regression_is_the_missing_lookahead_block(prefix_hits, monkeypatch):
    prompt = tokens(LIVE_PROMPT)
    # A1/A2: 64-token draft blocks. The EAGLE lookahead block
    # [100352, 100416) is complete, so the drafter verifies the boundary.
    baseline = layout(prefix_hits, monkeypatch, compact=False, swa_retention=0)
    baseline.prefill(prompt)
    _, hit = baseline.lookup(prompt)
    assert hit == LIVE_BOUNDARY
    # B without the boundary lookup: only 349 tokens follow the boundary, so
    # no complete 896-token lookahead block exists; the replay clamp backs the
    # shared hit up one 3584-token page. Exactly the measured 96,768.
    compact = layout(prefix_hits, monkeypatch, compact=True, swa_retention=0)
    assert compact.draft_block == 896 and compact.coordinator.dflash_boundary_group_ids
    compact.prefill(prompt)
    compact.coordinator.dflash_boundary_group_ids = frozenset()
    blocks, hit = compact.lookup(prompt)
    assert hit == 96768 == LIVE_BOUNDARY - MLA_BLOCK
    assert blocks[compact.draft_gid] == []
    assert compact.uncached == MLA_BLOCK  # the clamped-away page is reported
    # B with the boundary lookup: the cached window ends at the boundary.
    compact.coordinator.dflash_boundary_group_ids = frozenset({compact.draft_gid})
    blocks, hit = compact.lookup(prompt)
    assert hit == LIVE_BOUNDARY and compact.uncached == 0
    draft = blocks[compact.draft_gid]
    assert len(draft) == LIVE_BOUNDARY // 896
    window = math.ceil((WINDOW - 1) / 896)
    assert all(not b.is_null for b in draft[-window:])
    assert all(b.is_null for b in draft[:-window])


@pytest.mark.parametrize("swa_retention", [None, 0])
def test_boundary_lookup_hits_at_every_alignment_offset(prefix_hits, monkeypatch, swa_retention):
    compact = layout(prefix_hits, monkeypatch, True, swa_retention)
    base = 8 * MLA_BLOCK
    # Every tail length past the aligned boundary across one full page, plus
    # tails longer than the draft window (no replay clamp at all).
    for offset in [*range(1, MLA_BLOCK + 1), 2049, 3000]:
        prompt = tokens(base + offset, seed=offset)
        aligned = (base + offset - 1) // MLA_BLOCK * MLA_BLOCK
        compact.prefill(prompt)
        blocks, hit = compact.lookup(prompt)
        assert hit == aligned and compact.uncached == 0, offset
        assert len(blocks[compact.draft_gid]) == aligned // 896, offset
        assert not blocks[compact.draft_gid][-1].is_null, offset


@pytest.mark.parametrize("swa_retention", [None, 0])
def test_eagle_lookup_needs_a_complete_lookahead_block(prefix_hits, monkeypatch, swa_retention):
    """The default 64-token layout: the EAGLE path only verifies when a
    complete draft block follows the boundary; otherwise the replay clamp
    costs one MLA page. Exhaustive around that threshold."""
    baseline = layout(prefix_hits, monkeypatch, False, swa_retention)
    base = 8 * MLA_BLOCK
    for offset in [*range(1, 129), 896, 2048, 2049, MLA_BLOCK]:
        prompt = tokens(base + offset, seed=offset)
        aligned = (base + offset - 1) // MLA_BLOCK * MLA_BLOCK
        baseline.prefill(prompt)
        _, hit = baseline.lookup(prompt)
        lookahead = offset - 1 >= 64
        assert hit == (aligned if lookahead or offset > WINDOW else aligned - MLA_BLOCK), offset


def test_boundary_lookup_equals_eagle_lookup_when_lookahead_exists(prefix_hits, monkeypatch):
    compact = layout(prefix_hits, monkeypatch, True, 0)
    prompt = tokens(5 * MLA_BLOCK + 1000)
    compact.prefill(prompt)
    with_boundary = compact.lookup(prompt)
    compact.coordinator.dflash_boundary_group_ids = frozenset()
    with_eagle = compact.lookup(prompt)
    assert with_boundary[1] == with_eagle[1] == 5 * MLA_BLOCK
    assert with_boundary[0] == with_eagle[0]


def test_changed_suffix_and_shared_prefix_reuse_the_window(prefix_hits, monkeypatch):
    compact = layout(prefix_hits, monkeypatch, True, 0)
    shared = tokens(LIVE_PROMPT)
    first = compact.prefill(shared)
    cached = compact.draft_blocks(first)
    for divergence in (LIVE_BOUNDARY, LIVE_BOUNDARY + 1, LIVE_PROMPT - 1):
        other = shared[:divergence] + tokens(LIVE_PROMPT - divergence + 500, seed=divergence)
        blocks, hit = compact.lookup(other)
        assert hit == LIVE_BOUNDARY
        draft = blocks[compact.draft_gid]
        # Shared read: the window is the first request's cached blocks; the
        # returned prefix stops at the 896-aligned boundary, so every write of
        # the new request (context KV from its own fresh tokens) lands past it.
        assert draft[-1] is cached[LIVE_BOUNDARY // 896 - 1]
        assert len(draft) * 896 == LIVE_BOUNDARY and LIVE_BOUNDARY % 896 == 0
        assert blocks[0][-1] is compact.coordinator.single_type_managers[0].req_to_blocks[first.request_id][LIVE_BOUNDARY // MLA_BLOCK - 1]
    # A prefix diverging inside the last MLA page shares one page less.
    blocks, hit = compact.lookup(shared[:LIVE_BOUNDARY - 1] + tokens(2000, seed=9))
    assert hit == LIVE_BOUNDARY - MLA_BLOCK


@pytest.mark.parametrize("swa_retention", [None, 0])
def test_missing_window_block_falls_back_to_the_replay_clamp(prefix_hits, monkeypatch, swa_retention):
    compact = layout(prefix_hits, monkeypatch, True, swa_retention)
    prompt = tokens(LIVE_PROMPT)
    first = compact.prefill(prompt)
    cached = compact.draft_blocks(first)
    last = LIVE_BOUNDARY // 896 - 1
    window = math.ceil((WINDOW - 1) / 896)
    # Blocks before the window are irrelevant to the boundary verification.
    compact.pool.evict(cached[last - window])
    blocks, hit = compact.lookup(prompt)
    assert hit == LIVE_BOUNDARY and not blocks[compact.draft_gid][-1].is_null
    # Any evicted window block leaves the drafter unverified: the shared hit
    # backs up one page. With boundary-only retention the drafter gets a fresh
    # window; with dense retention it verifies again at the earlier page.
    compact.pool.evict(cached[last - 1])
    blocks, hit = compact.lookup(prompt)
    assert hit == LIVE_BOUNDARY - MLA_BLOCK
    draft = blocks[compact.draft_gid]
    if swa_retention == 0:
        assert draft == []
    else:
        assert len(draft) == hit // 896 and not draft[-1].is_null


def test_boundary_lookup_is_off_by_default_and_dflash_only(prefix_hits, monkeypatch):
    compact = layout(prefix_hits, monkeypatch, True, 0)
    assert compact.coordinator.dflash_boundary_group_ids == frozenset({compact.draft_gid})
    assert compact.coordinator.eagle_group_ids == {compact.draft_gid}
    assert compact.coordinator.single_type_managers[compact.draft_gid].use_eagle
    baseline = layout(prefix_hits, monkeypatch, False, 0)
    assert baseline.coordinator.dflash_boundary_group_ids == frozenset()
    # The coordinator validates the flag itself; the allocator runs earlier.
    _, cache_config, _ = make_groups(prefix_hits)
    monkeypatch.setenv(ENV, "yes")
    with pytest.raises(ValueError, match="GLM53_DRAFT_KV_COMPACT"):
        Cache(prefix_hits, cache_config)

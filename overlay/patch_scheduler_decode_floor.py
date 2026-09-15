#!/usr/bin/env python3
"""Mixed-prefill policy: skip / cap / off / opt-in fair (issue #6).

A decode lane on this backend needs ~8 tokens (1 + DFlash2 k=7). The leftover
MNBT budget otherwise goes to a peer FLASHINFER_MLA_SPARSE_SM120 prefill
chunk. Mixed execution leaves the uniform decode FULL-graph path, and a
128-token cap is still ~10 tok/s at 80k KV. Token caps are not a millisecond
budget.

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode (TP=2 default). Starves every
               waiting/running prefill while any peer is decoding; there is
               no age limit. Solo prefill is unchanged.
  N>0        — cap mixed prefill chunks to N tokens while a peer decodes.
               The cap is fed into hybrid Mamba alignment so N < block_size
               still makes sub-block progress. 128 still stalls ~10 tok/s.
  0 / off    — disable the extra isolation policy.
  fair       — opt-in service-time mixing (not the default). Decode-only
               steps between prefill turns; at most
               GLM53_FAIR_PREFILL_MAX_CHUNKS chunks per turn (default 1),
               each GLM53_FAIR_PREFILL_CHUNK tokens (default 256). Turns are
               gated by a prefill GPU-time share and a maximum interval so a
               mixed step that takes seconds cannot be admitted every N
               millisecond decode steps. In-flight async work blocks the next
               mixed turn. Prefills are selected by time since last *actual*
               prefill service (admission does not reset entitlement) plus
               round-robin. Solo prefill is unchanged.

Fair knobs (read at runtime; identical on every rank):
  GLM53_FAIR_PREFILL_CHUNK            default 256
  GLM53_FAIR_PREFILL_SHARE            default 0.20 (recent GPU-time fraction)
  GLM53_FAIR_PREFILL_MAX_INTERVAL_MS  default 2000
  GLM53_FAIR_PREFILL_MAX_CHUNKS       default 1

Versioned installer: `# [glm53-decode-floor:v2]`. A v1 image (marker without
`:v2`) is unpatched then re-patched. Fail closed if anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"
MARK_V2 = "# [glm53-decode-floor:v2]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

# v1 helper + insertions (recipe f906ee9 / this overlay before v2).
V1_HELPER_START = "def _glm53_mixed_prefill_policy(running, current):"
V1_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""
V1_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

HELPER = '''
class _Glm53MixedPrefill:  # [glm53-decode-floor:v2]
    """Skip / cap / off / fair mixed-prefill policy. CPU-only; no GPU sync."""

    def __init__(self, now=None) -> None:
        self._now = now or time.monotonic
        self.mode = "skip"
        self.legacy_cap = 0
        self.chunk = 256
        self.share = 0.20
        self.interval_s = 2.0
        self.max_chunks = 1
        self.hist_every = 50
        self.logged_boot = False
        self._parse()
        self.last_service = {}
        self.arrival = {}
        self.rr_seq = {}
        self.rr_n = 0
        self.step_prefill_ids = set()
        self.inflight = []
        self.inflight_mixed = 0
        self.history = []
        self.last_schedule_mono = None
        self.last_prefill_turn_mono = 0.0
        self.step_tag = None
        self._sched_id = None
        self.step_mode = "solo"
        self.selected = set()
        self.defer_reason = "none"
        self.granted_this_step = 0
        self.steps = 0

    def _e(self, name, default):
        v = os.environ.get(name)
        return default if v is None or not str(v).strip() else str(v).strip()

    def _parse(self) -> None:
        raw = self._e("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
        self.legacy_cap = 0
        if raw in ("0", "off", "no"):
            self.mode = "off"
        elif raw in ("skip", "-1"):
            self.mode = "skip"
        elif raw == "fair":
            self.mode = "fair"
        else:
            try:
                cap = int(raw)
            except ValueError:
                cap = 0
            if cap <= 0:
                self.mode = "off"
            else:
                self.mode = "cap"
                self.legacy_cap = cap
        try:
            self.chunk = int(self._e("GLM53_FAIR_PREFILL_CHUNK", "256"))
        except ValueError:
            self.chunk = 256
        if self.chunk <= 0:
            self.chunk = 256
        try:
            self.share = float(self._e("GLM53_FAIR_PREFILL_SHARE", "0.20"))
        except ValueError:
            self.share = 0.20
        self.share = min(1.0, max(0.0, self.share))
        try:
            self.interval_s = int(self._e("GLM53_FAIR_PREFILL_MAX_INTERVAL_MS", "2000")) / 1000.0
        except ValueError:
            self.interval_s = 2.0
        if self.interval_s <= 0:
            self.interval_s = 2.0
        try:
            self.max_chunks = int(self._e("GLM53_FAIR_PREFILL_MAX_CHUNKS", "1"))
        except ValueError:
            self.max_chunks = 1
        self.max_chunks = max(1, min(self.max_chunks, 16))
        if self.mode == "fair" and not self.logged_boot:
            print(
                f"[glm53-decode-floor] fair chunk={self.chunk} share={self.share} "
                f"interval_s={self.interval_s} max_chunks={self.max_chunks}",
                flush=True,
            )
            self.logged_boot = True

    @staticmethod
    def prefill_remaining(request) -> int:
        """Tokens still requiring prefill/replay compute (not the decode row)."""
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        num_tokens = int(getattr(request, "num_tokens", prompt) or prompt)
        if computed < prompt:
            return prompt - computed
        prefill_end = max(prompt, num_tokens - 1 if num_tokens > 0 else prompt)
        return max(0, prefill_end - computed)

    def needs_prefill_compute(self, request) -> bool:
        return self.prefill_remaining(request) > 0

    def _iter_waiting(self, sched):
        for name in ("waiting", "skipped_waiting"):
            q = getattr(sched, name, None)
            if not q:
                continue
            try:
                for r in q:
                    yield r
            except TypeError:
                continue

    def _live_ids(self, sched):
        ids = set()
        for r in list(getattr(sched, "running", None) or []):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        for r in self._iter_waiting(sched):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        reqs = getattr(sched, "requests", None)
        if isinstance(reqs, dict):
            ids.update(reqs.keys())
        return ids

    def _prune(self, live):
        for store in (self.last_service, self.arrival, self.rr_seq):
            dead = [k for k in store if k not in live]
            for k in dead:
                store.pop(k, None)

    def _recent_share(self, now):
        window = 8.0
        while self.history and now - self.history[0][0] > window:
            self.history.pop(0)
        prefill_s = sum(dt for _, dt, mix in self.history if mix)
        decode_s = sum(dt for _, dt, mix in self.history if not mix)
        total = prefill_s + decode_s
        if total <= 0:
            return 0.0, 0.0, 0.0
        return prefill_s / total, prefill_s, decode_s

    def _oldest_wait(self, now, prefills):
        oldest = 0.0
        for r in prefills:
            rid = getattr(r, "request_id", None)
            if rid is None:
                continue
            served = self.last_service.get(rid)
            if served is None:
                oldest = max(oldest, now - self.arrival.get(rid, now))
            else:
                oldest = max(oldest, now - served)
        return oldest

    def _select(self, prefills):
        ranked = []
        for r in prefills:
            rid = getattr(r, "request_id", None)
            if rid is None:
                continue
            ranked.append((self.last_service.get(rid, 0.0), self.rr_seq.get(rid, 0), rid, r))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        out = []
        for _, _, rid, r in ranked[: self.max_chunks]:
            out.append(r)
        return out

    def _flush_prior_schedule(self) -> None:
        if self.last_schedule_mono is None:
            return
        ids = set(self.step_prefill_ids)
        self.inflight.append(ids)
        if ids:
            self.inflight_mixed += 1
        self.step_prefill_ids = set()
        self.last_schedule_mono = None

    def begin_step(self, sched) -> None:
        now = self._now()
        sid = id(sched)
        if self._sched_id is not None and self._sched_id != sid:
            self.inflight.clear()
            self.inflight_mixed = 0
            self.step_prefill_ids = set()
            self.last_schedule_mono = None
        self._sched_id = sid
        tag = (sid, int(getattr(sched, "current_step", 0) or 0))
        if tag == self.step_tag:
            return
        self._flush_prior_schedule()
        self.step_tag = tag
        self.steps += 1
        self.granted_this_step = 0
        self.selected = set()
        self.step_prefill_ids = set()
        self.last_schedule_mono = now
        sched._glm53_align_prefill_limit = None
        live = self._live_ids(sched)
        self._prune(live)
        running = list(getattr(sched, "running", None) or [])
        waiting = list(self._iter_waiting(sched))
        for r in running + waiting:
            rid = getattr(r, "request_id", None)
            if rid is not None and rid not in self.arrival:
                self.arrival[rid] = now
        prefills = [r for r in running + waiting if self.needs_prefill_compute(r)]
        decodes = [r for r in running if not self.needs_prefill_compute(r)]
        if self.mode != "fair":
            self.step_mode = "legacy"
            self.defer_reason = "none"
            return
        if not prefills:
            self.step_mode = "solo"
            self.defer_reason = "none"
            return
        if not decodes:
            self.step_mode = "solo"
            self.defer_reason = "none"
            return
        if self.inflight_mixed > 0:
            self.step_mode = "decode_only"
            self.defer_reason = "async_inflight"
            self._maybe_log(now, len(prefills), len(decodes))
            return
        share, _, _ = self._recent_share(now)
        since = now - self.last_prefill_turn_mono if self.last_prefill_turn_mono else self.interval_s
        starved = self._oldest_wait(now, prefills) >= self.interval_s
        want = (
            self.last_prefill_turn_mono == 0.0
            or share < self.share
            or since >= self.interval_s
            or starved
        )
        if not want:
            self.step_mode = "decode_only"
            self.defer_reason = "share"
            self._maybe_log(now, len(prefills), len(decodes))
            return
        chosen = self._select(prefills)
        self.selected = {getattr(r, "request_id", None) for r in chosen}
        self.selected.discard(None)
        self.step_mode = "prefill_turn"
        self.defer_reason = "none"
        self._maybe_log(now, len(prefills), len(decodes))

    def _maybe_log(self, now, n_prefill, n_decode) -> None:
        if self.hist_every <= 0 or self.steps % self.hist_every != 1:
            return
        share, pre_s, dec_s = self._recent_share(now)
        print(
            f"[glm53-decode-floor] step={self.steps} mode={self.step_mode} "
            f"defer={self.defer_reason} inflight={self.inflight_mixed} "
            f"prefill={n_prefill} decode={n_decode} selected={len(self.selected)} "
            f"share={share:.3f} pre_s={pre_s:.3f} dec_s={dec_s:.3f}",
            flush=True,
        )

    def cap_for(self, sched, request):
        self.begin_step(sched)
        if not self.needs_prefill_compute(request):
            sched._glm53_align_prefill_limit = None
            return None
        if self.mode == "off":
            sched._glm53_align_prefill_limit = None
            return None
        running = list(getattr(sched, "running", None) or [])
        rid = getattr(request, "request_id", None)
        peer_decode = False
        for r in running:
            if r is request or getattr(r, "request_id", None) == rid:
                continue
            if not self.needs_prefill_compute(r):
                peer_decode = True
                break
        if self.mode == "skip":
            cap = 0 if peer_decode else None
        elif self.mode == "cap":
            cap = self.legacy_cap if peer_decode else None
        else:
            # fair
            if self.step_mode == "solo":
                cap = None
            elif self.step_mode != "prefill_turn":
                cap = 0
            elif rid not in self.selected:
                cap = 0
            elif self.granted_this_step >= self.max_chunks:
                cap = 0
            else:
                cap = self.chunk
        if cap is not None and cap > 0:
            sched._glm53_align_prefill_limit = cap
            self.step_prefill_ids.add(rid)
            self.granted_this_step += 1
        else:
            sched._glm53_align_prefill_limit = None
        return cap

    def note_scheduled(self, request, num_new_tokens: int) -> None:
        """Keep step_prefill_ids as actual post-alignment prefill tokens only."""
        rid = getattr(request, "request_id", None)
        if rid is None:
            return
        if num_new_tokens <= 0 or not self.needs_prefill_compute(request):
            self.step_prefill_ids.discard(rid)
            return
        self.step_prefill_ids.add(rid)

    def observe_output(self, sched, scheduler_output) -> None:
        now = self._now()
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        if self.inflight:
            ids = self.inflight.pop(0)
            if ids:
                self.inflight_mixed = max(0, self.inflight_mixed - 1)
        elif self.last_schedule_mono is not None:
            ids = set(self.step_prefill_ids)
            self.step_prefill_ids = set()
            self.last_schedule_mono = None
        else:
            ids = set()
        served = set()
        for rid in ids:
            n = int(num_scheduled.get(rid, 0) or 0)
            if n > 0:
                self.last_service[rid] = now
                self.rr_n += 1
                self.rr_seq[rid] = self.rr_n
                served.add(rid)
        mixed = bool(served)
        start = getattr(self, "_observe_prev", None)
        dt = 0.0 if start is None else max(0.0, now - start)
        self._observe_prev = now
        if dt > 0:
            self.history.append((now, dt, mixed))
        if mixed:
            self.last_prefill_turn_mono = now
        live = self._live_ids(sched)
        self._prune(live)

    @staticmethod
    def aligned_new_tokens(
        start, num_new, prefill_end, block_size, max_prefill_tokens, policy_cap=None
    ) -> int:
        """Hybrid align clip. policy_cap is an intentional mixed cap, not leftover budget."""
        if policy_cap is not None and policy_cap > 0:
            max_prefill_tokens = min(max_prefill_tokens, policy_cap)
        end = start + num_new
        if end < prefill_end:
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
        return max(0, end - start)


_GLM53_MIXED = _Glm53MixedPrefill()  # [glm53-decode-floor:v2]


def _glm53_mixed_prefill_policy(sched, request):  # [glm53-decode-floor:v2]
    return _GLM53_MIXED.cap_for(sched, request)


'''

BEGIN_OLD = """        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
"""
BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v2]
        # NOTE(woosuk) on the scheduling algorithm:
"""

OBS_OLD = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
"""
OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v2]
        pooler_outputs = model_runner_output.pooler_output
"""

RUNNING_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
"""
RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
"""
WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

ALIGN_OLD = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v2]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""

RUNNING_MAMBA_OLD = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
"""
RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
"""

WAITING_MAMBA_OLD = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if num_new_tokens == 0:
                            break
"""
WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
                        if num_new_tokens == 0:
                            break
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def _strip_v1_helper(text: str) -> str:
    start = text.find(V1_HELPER_START)
    if start < 0:
        raise SystemExit(f"{P}: v1 helper start not found")
    # Include the leading newline so we do not leave a blank pile-up.
    if start > 0 and text[start - 1] == "\n":
        start -= 1
    end_ak = text.find("class _Glm53AdaptiveK:", start)
    end_cg = text.find("from vllm.compilation.cuda_graph import CUDAGraphStat\n", start)
    candidates = [i for i in (end_ak, end_cg) if i > start]
    if not candidates:
        raise SystemExit(f"{P}: v1 helper end not found")
    end = min(candidates)
    return text[:start] + "\n" + text[end:]


def unpatch_v1(text: str) -> str:
    if V1_RUNNING_NEW in text:
        text = replace_once(text, V1_RUNNING_NEW, RUNNING_OLD, "v1-running")
    elif "_glm53_mixed_prefill_policy(self.running, request)" in text:
        raise SystemExit(f"{P}: v1 running insertion drifted")
    if V1_WAITING_NEW in text:
        text = replace_once(text, V1_WAITING_NEW, WAITING_OLD, "v1-waiting")
    elif "_glm53_mixed_prefill_policy(self.running, request)" in text:
        raise SystemExit(f"{P}: v1 waiting insertion drifted")
    if V1_HELPER_START in text:
        text = _strip_v1_helper(text)
    leftover = [
        "_glm53_mixed_prefill_policy(self.running, request)",
        V1_HELPER_START,
    ]
    for s in leftover:
        if s in text:
            raise SystemExit(f"{P}: v1 leftover after unpatch: {s}")
    return text


def apply_v2(text: str) -> str:
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    if "class _Glm53MixedPrefill:" not in text:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")
        text = text.replace(needle, HELPER + needle, 1)
    text = replace_once(text, BEGIN_OLD, BEGIN_NEW, "begin-step")
    text = replace_once(text, OBS_OLD, OBS_NEW, "observe-output")
    text = replace_once(text, RUNNING_OLD, RUNNING_NEW, "running-prefill")
    text = replace_once(text, WAITING_OLD, WAITING_NEW, "waiting-prefill")
    text = replace_once(text, ALIGN_OLD, ALIGN_NEW, "mamba-align")
    text = replace_once(text, RUNNING_MAMBA_OLD, RUNNING_MAMBA_NEW, "running-mamba-note")
    text = replace_once(text, WAITING_MAMBA_OLD, WAITING_MAMBA_NEW, "waiting-mamba-note")
    return text


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK_V2 in text:
        print(f"{P.name}: {MARK_V2} already present — skipping")
        return 0
    if MARK in text or V1_HELPER_START in text:
        print(f"{P.name}: migrating {MARK} -> {MARK_V2}", flush=True)
        text = unpatch_v1(text)
    text = apply_v2(text)
    if MARK_V2 not in text:
        raise SystemExit(f"{P}: v2 marker missing after patch")
    P.write_text(text)
    cap = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")
    print(f"patched {P.name} ({MARK_V2} mixed prefill policy={cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

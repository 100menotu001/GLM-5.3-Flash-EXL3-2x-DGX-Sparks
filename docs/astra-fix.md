The reported concurrency failure has two causes: `skip` deliberately withholds every newcomer’s prefill while a peer decodes; disabling it admits the newcomer into expensive shared execution steps that slow the incumbent. `LONG_PREFILL_TOKEN_THRESHOLD=1024` reduces the second effect, but the supplied measurements still show about a 10× decode slowdown. Neither changing `skip` to `0` alone nor the newer default threshold of `3584` establishes acceptable interactive concurrency on this kit.

**Status, 2026-09-15.** The opt-in `fair` policy proposed below is implemented and was enabled on this kit. Incumbent decode collapse is fixed at the default fair knobs (~1.1× vs the reporter’s 10–36×). A large cold newcomer still does not get a first token during a short decode window. `fair` is **not** the TP2 published default (`start.sh` still defaults to `skip`). This is not a closed reporter-recipe validation.

The 2026-09-14 text is the diagnosis and original design; its performance table is the reporter’s. The 2026-09-15 table is from this machine after `./start.sh restart`.

**2026-09-15 implementation.** `overlay/patch_scheduler_decode_floor.py` is versioned `# [glm53-decode-floor:v2]`. A v1 image is unpatched then re-patched; fail-closed if anchors drift. `GLM53_MIXED_PREFILL_CHUNK` now accepts `skip` / `-1`, `N>0`, `0`/`off`, and `fair`. Fair knobs (identical on every rank): `GLM53_FAIR_PREFILL_CHUNK=256`, `GLM53_FAIR_PREFILL_SHARE=0.20`, `GLM53_FAIR_PREFILL_MAX_INTERVAL_MS=2000`, `GLM53_FAIR_PREFILL_MAX_CHUNKS=1`. One decision per `schedule()`; completed-step feedback via `observe_output` (no GPU sync); decode reserved first; prefills selected by last actual prefill service plus round-robin; in-flight async mixed steps block the next mixed turn; solo prefill uncapped; `_glm53_align_prefill_limit` so `N < block_size` still progresses. Launchers (`start.sh`, `start-tp3.sh`, `start-tp4.sh`), `.env.example`, README, and host tests were wired; TP2/TP4 default remains `skip`, TP3 remains `0`. Overlay is bind-mounted at start — no image rebuild was required. CPU tests: `tests/test_scheduler_decode_floor.py`, `tests/test_numeric_config.py`.

**2026-09-15 live enable.** `.env` was set to `CHUNK=fair` with the default fair knobs and `LONG_PREFILL_TOKEN_THRESHOLD=3584`, then `./start.sh restart`. Image `glm53-flash-sm121:e3-20260907`. Head container `glm53-exl3-head` migrated `v1 → v2` and logged `mixed prefill policy=fair chunk=256 share=0.2 interval_s=2.0 max_chunks=1`. `/health` passed. Serve `http://127.0.0.1:8888`, model `GLM-5.3-Flash-EXL3`.

**2026-09-15 overlap results (single runs).** Thinking off, temp 0. Overlap tok/s for the longer arms is estimated from stream events scaled by `completion_tokens / events` (the stock harness’s `overlap_tok_s` is event rate, not token rate). The stock `Count from 1 to 80` task stops around 160 tokens, so `--max-tokens 400` does not keep A in decode.

| Arm | A prompt | A decode | C1 tok/s | A overlap tok/s | C1/overlap | B TTFT | B first token during A? |
|---|---|---:|---:|---:|---:|---:|---|
| Stock harness (`the` × 8k) | ~8.0k tok | ~160 tok, then `stop` | 74.4 | 7.6 *events/s* (do not use) | 1.22× whole-window | 8.6 s | No (A finished first) |
| Longer A, repeated `the` × 8k | ~8.1k tok | 800 tok / 13.1 s | 66.8 | **60.9** | **1.10×** | **4.2 s** | **Yes**; B also finished during A |
| Cold unique words (~8k words) | **29.1–30.6k tok** | 800 tok / 14.2 s | 64.4 | **56.4** | **1.14×** | **36.3 s** | **No** |

JSON: `/tmp/mixed-prefill-fair.json`, `/tmp/mixed-prefill-fair-long.json`, `/tmp/mixed-prefill-fair-cold.json`.

**What that means.** Decode isolation works: A stayed in the 56–61 tok/s band instead of the reporter’s 0.8–2.8 tok/s under `CHUNK=0`. The repeated-`the` arm is cache-cheap (C1 TTFT fell from 6.8 s on the first post-restart request to ~1.0 s); it shows B can be admitted during decode, not that a 30k cold prompt is fast. The unique-token arm is the honest mixed step: B’s 36 s TTFT is about “wait for A’s 14 s decode, then a full ~22 s prefill.” At 256 tokens every 2 s, 14 s of overlap moves only a few thousand of 30k tokens, so B cannot finish during a short decode. Head logs sampled `mode=decode_only defer=share` (`pre_s`/`dec_s` in the rolling window, share often ≥ 0.20). A’s own just-finished chunked prefill inflates that share, so B is deferred until the window ages — extra skip-like delay on top of the small chunk.

**Still not done.** The reporter sequence (thinking essay, `max_tokens=4000`, B ~30k cold, 10 s into steady decode) was not rerun. p50/p95 delivery gaps, three-repeat alternating prefixes, 2k vs 30k B, and multi-newcomer arms were not collected. Do not promote `fair` to the TP2 default on this evidence. If the 30k newcomer must finish during a short decode, raise `GLM53_FAIR_PREFILL_CHUNK` / `SHARE` (more cost to A) or keep independent capacity; scheduling at 256/2 s cannot meet that TTFT.

The remainder of this note is the 2026-09-14 diagnosis. Checkout caveat in the next paragraph was already stale for later HEADs; as of 2026-09-15, TP2 `start.sh` / `.env.example` default mixed prefill to `skip` and leave long-prefill empty unless `.env` sets it. Live `.env` on this kit used `fair` and threshold `3584`. Omitted reasoning effort in the chat template is **Max**, not High.

**The version matters.** I inspected recipe `f906ee990596486e10ddbe381efa6f0e496f77e3` with `git show`, rather than treating this checkout as that recipe. Its launcher, example environment, and injected scheduler helper all default mixed prefill to `skip`; its long-prefill threshold is empty, so the launcher omits the flag. This checkout is already at `c589286` with mixed prefill defaulting to `0` and the threshold to `3584`. Those later changes are partial mitigations, and must not be confused with the reported image’s defaults. [Recipe launcher](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f906ee990596486e10ddbe381efa6f0e496f77e3/start.sh), [recipe environment](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f906ee990596486e10ddbe381efa6f0e496f77e3/.env.example).

For the underlying scheduler, I extracted source from the locally available, exact base-image digest pinned by that recipe: `vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`. Its version file identifies `0.1.dev20051+g487ecf187`. Extraction used a temporary read-only container without GPUs or networking. The reporter’s final `local/mia-glm53-exl3:recipe-f906ee9` image is not installed here, so its actual runtime environment and any unreported customizations remain unverified. The recipe patch applies successfully to the pinned base source in a temporary directory.

**What the measurements establish.** The setup is 2× GB10, EXL3 TR3-4bpw, MNBT 7168, four sequence slots, maximum context 850000, DFlash2 k=7 with adaptive-k EMA, ABLIT transplant, and default reasoning effort reported as `high` (omitted template effort is actually **Max**). A streams a thinking-enabled essay with `max_tokens=4000`; ten seconds into steady decode, B submits an approximately 30k-token cold random-word prompt with thinking disabled and `max_tokens=8`.

| Mixed policy / global threshold | B admission observed | Reported B prefill duration | A before → during → after, approximate tok/s | Interpretation |
|---|---:|---:|---:|---|
| `skip` / unset | After A finished, 221 s | Not supplied | Unaffected | Prefill admission starvation |
| `0` / unset | 1.0 s | 41 s | 28.8 → 0.8 → 23.7 | Admission succeeds; incumbent decode is about 36× slower during overlap |
| `0` / `1024` | 1.0 s | 52 s | 27.7 → 2.8 → 26.5 | About 10× decode slowdown remains; B takes about 27% longer than in the uncapped run |

These are single runs with characters/4 as a token estimate, and admission sampled once per second. They demonstrate a large contention effect, not precise throughput guarantees or an optimal threshold. The head’s 0.4–0.6 generation tok/s is an aggregate logging-window measurement; it need not equal the client’s 0.8 estimate. B’s exact prefill interval should be verified from the harness, since submission-to-first-token also contains queueing, tokenization, and first-token execution.

The original 20-minute coding example is the same starvation mechanism lasting longer. `max-model-len=850000` is a limit, not evidence that A actually had 850k tokens in KV.

**The admission bug is explicit in the overlay.** This subsection describes the recipe / v1 helper. v2 still implements the same `skip` predicate when `CHUNK=skip`; `fair` replaces it. In `overlay/patch_scheduler_decode_floor.py`, `_glm53_mixed_prefill_policy()` scans `self.running`. If any other request has `num_computed_tokens >= num_prompt_tokens`, it returns zero for `skip`. The waiting-request insertion then executes:

```python
if mixed_cap <= 0:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

The request is returned to the skipped waiting queue and encounters the same decision next time. There is no age limit, service credit, or maximum deferral count. A 2k prompt and a 30k prompt meet the same predicate. For a partially prefilling request already in `running`, the other insertion caps `num_new_tokens` to zero; the scheduler skips that request for the step. This also makes a running-count metric insufficient to prove progress. [Exact recipe patch, helper and both insertion points](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f906ee990596486e10ddbe381efa6f0e496f77e3/overlay/patch_scheduler_decode_floor.py#L39).

`MAX_NUM_SEQS=4` only permits up to four running requests; it does not override this policy or reserve execution time for each one. With a continuously present decoding peer, a prefill can wait indefinitely. The supplied configuration change admitting B immediately is strong evidence for this cause, rather than a four-slot configuration failure or an HTTP streaming-parser hang. KV exhaustion can separately prevent admission, but is not needed to explain this controlled comparison.

Reasoning effort affects how long A occupies its lane. The default `high` is not a ceiling on a request selecting `max`; the recipe explicitly allows per-request template kwargs to override the default. The scheduler predicate does not inspect thinking, effort, or ABLIT. Reducing output budgets can shorten the incident, but does not repair admission fairness. [Recipe template](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f906ee990596486e10ddbe381efa6f0e496f77e3/files/chat_template.jinja#L1).

**Why `0` transfers the delay to A.** `0` makes the helper return `None`: the extra isolation policy is disabled. The scheduler allocates tokens to running requests first and then waiting requests. In this experiment A is already running when B arrives, so A can receive its small decode allocation and B can consume much of the remaining step budget. Once admitted, B continues chunked prefill in the running loop.

The two GB10s are tensor-parallel ranks of one engine. They execute that batch together; they are not independent workers for A and B. A’s next output depends on completing the shared model step. Reserving A a few token positions does not let it execute many additional decode iterations while the same step processes thousands of B’s tokens. DFlash k=7 adds verification work of up to eight positions per decoding request; adaptive-k changes that work, and accepted output tokens are not the same as scheduled positions. The scheduler also reserves drafting input slots, so the prefill allowance is not exactly `7168 - 8`.

The global threshold is applied to `num_new_tokens` in both running and waiting paths. It caps a request’s scheduled chunk, including solo prefill; it is neither a millisecond budget nor an aggregate prefill budget across all requests. At C4, several prefills can each receive their cap until the engine’s other budgets run out. Leaving token capacity unused is not the same as inserting additional decode iterations.

The backend makes this tradeoff more severe than a token-count model suggests. The recipe routes SM120 sparse-MLA prefill through its supported sparse path. The pinned `FlashInferMLASparseMetadataBuilder` declares `UNIFORM_BATCH` graph support; adding a long prefill does not preserve the uniform decode FULL-graph path. Mixed execution can still use applicable piecewise graphs. The overlay’s original rationale also records expensive mixed execution at a 128-token cap and 80k history. Sparse-indexer work, graph-path changes, MoE work, and TP synchronization are plausible contributors; **the supplied data does not identify their individual costs**. Do not label this an ABLIT, acceptance-rate, or NCCL defect without a trace. [Recipe backend adaptation](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/f906ee990596486e10ddbe381efa6f0e496f77e3/Dockerfile#L1), [adaptive-k implementation](../overlay/patch_adaptive_k.py).

Upstream also documents that smaller prefill batches generally improve inter-token latency at a cost to prefill performance. That supports tuning the tradeoff; it does not promise latency isolation on this backend. [vLLM chunked-prefill tuning](https://docs.vllm.ai/en/latest/configuration/optimization/#performance-tuning-with-chunked-prefill).

**A second code hazard affects proposed small-cap fixes.** The pinned scheduler’s `_mamba_block_aligned_split()` runs after the mixed cap. In hybrid `mamba_cache_mode="align"`, its decision to allow sub-block progress uses:

```python
max_prefill_tokens = self.max_num_scheduled_tokens
if long_prefill_threshold > 0:
    max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
aligned_end = end // block_size * block_size
if aligned_end > start or block_size <= max_prefill_tokens:
    end = aligned_end
```

It does not know the overlay’s smaller mixed cap. At a block boundary, if the prompt needs more chunks and `mixed_cap < block_size <= max_prefill_tokens`, the capped chunk rounds down to zero. A waiting request then hits the alignment branch’s `break`; a running prefill takes the zero-token `continue`. Repeating this decision can reproduce starvation even with a positive mixed cap. This is an additional hazard when choosing a fix, not the explanation for the reporter’s `CHUNK=0` runs.

I executed the actual extracted alignment method with CPU request/configuration stubs. With MNBT 7168 and an illustrative 3584-token block:

| Global threshold | Proposed chunk | Aligned result at position zero of a 30k prompt |
|---:|---:|---:|
| Unset / `0` | 128 | **0** |
| 3584 | 128 | **0** |
| 1024 | 128 | 128 |
| Unset / `0` | 1024 | **0** |
| 1024 | 1024 | 1024 |

The same probes passed with block size 1792. This verifies the conditional code behavior, not the reporter’s actual block geometry. The repository has documented 3584-token hybrid geometry, but the live `cache_config.block_size` and cache mode should be recorded before relying on it. [Recorded hybrid geometry](DESIGN-apc-per-group-retention.md). In particular, do not recommend “mixed cap 128, global threshold 3584” as universally safe.

**The immediate mitigation is already measured, but incomplete.** For an operator who needs newcomers to make progress before a scheduler change, the evidence-backed starting configuration is:

```dotenv
GLM53_MIXED_PREFILL_CHUNK=0
LONG_PREFILL_TOKEN_THRESHOLD=1024
MAX_NUM_BATCHED_TOKENS=7168
MAX_NUM_SEQS=4
```

That is a degraded-service workaround: on the reported run A still fell to 2.8 tok/s during B’s prefill. The later `3584` default has useful evidence from other workloads, but no result here establishes that it protects A; even `1024` fails to do so adequately. Changing these two existing settings requires recreating/restarting the serving processes with the intended environment and arguments, not rebuilding the image. Verify the effective head environment and both rank commands, since exported launcher variables can override `.env`. The 2026-09-14 investigation did not restart. The 2026-09-15 enable used `CHUNK=fair` (not this `0`/`1024` workaround) and did restart.

A bounded tuning experiment could hold the global threshold at 1024 and sweep the positive mixed cap through 512, 256, and 128, **after confirming that alignment allows progress at that threshold and at every boundary**. Positive mixed caps apply only when a decoding peer exists, whereas the global threshold still caps solo prefill. Measure both A’s delivery gaps and B’s continuing progress. This sweep may find a tolerable compromise, but the existing 128-token long-history observation means success cannot be assumed. Changing MNBT globally is another throughput/latency experiment, not an independent fix for starvation.

**The durable fix is scheduling by service time as well as tokens.** This was implemented on 2026-09-15 as opt-in `fair` in `overlay/patch_scheduler_decode_floor.py` and the scheduler it patches. Do not promote it to the multi-client default until the reporter-shaped overlap gates pass. Keep the legacy `skip` option available with its starvation behavior clearly documented. The implementation does the following:

1. **Make one shared policy decision per engine step.** Apply it to both waiting prefills and partially completed running prefills. Track time since each request last received actual prefill service, so admission alone does not reset its entitlement. Reserve eligible decodes first, including speculative input/slot accounting, regardless of the order in which prefills entered `self.running`.

2. **Reserve periodic prefill service and regular decode-only steps.** Between prefill turns, run the existing uniform decode path. On a prefill turn, admit a bounded aggregate amount of prefill work, initially at most one prefill chunk, alongside eligible decodes if profiling favors mixed execution. Select prefills with aging and round-robin service so an admitted long prompt cannot monopolize those turns. A pure-prefill turn is an alternative to benchmark, but it pauses A for that turn too.

3. **Control elapsed GPU service, not just step counts.** Start with a conservative chunk cap and update a cost estimate from completed engine steps, accounting for prompt position/history and batch shape. Combine a configurable prefill time share with a maximum interval between prefill opportunities. A rule such as “one mixed step every N decodes” alone is inadequate when a mixed step takes seconds and a decode step takes milliseconds. Account for async work already in flight so stale timing cannot admit a burst of expensive chunks. Avoid introducing a GPU synchronization on every scheduler decision solely for measurement.

4. **Make the small-chunk policy compatible with hybrid alignment.** Pass the effective policy chunk limit into the alignment decision, or otherwise explicitly allow the base implementation’s private sub-block state progression under that limit. Preserve mandatory cache boundaries, partial-tail handling, and resumed-request replay. Do not remove alignment or publish an incomplete recurrent state as a reusable prefix entry. Distinguish an intentional policy cap from temporary residual capacity after other requests consume a step budget.

5. **Handle transitions and overload.** Classify work using the actual tokens still requiring computation, including resumed/recomputed output, cached prefixes, and async placeholders. Remove policy state on completion/abort. Yield a partial prefill without discarding its KV. Log separate deferral reasons for policy, alignment, token budget, sequence slots, and KV capacity. Fair service promises apply when a request can obtain a slot and memory; they cannot override resource exhaustion or make an overloaded queue have bounded latency.

This design bounds starvation by offering prefill turns while preserving useful stretches of fast decode. It necessarily spends some GPU time on B. The share devoted to prefill, B’s TTFT, and A’s throughput must be chosen together. A chunk already executing cannot be interrupted by a scheduler-side timer, so if the smallest safe chunk exceeds the delivery-gap target, scheduling alone cannot meet that target. That outcome calls for reducing the measured mixed-step costs or adding independently provisioned serving capacity. Splitting this two-rank model into a “prefill Spark” and a “decode Spark” is not a configuration-only solution; disaggregation would also need validated transfer of this model’s hybrid recurrent, sparse, and draft state.

**Concrete implementation scope (done 2026-09-15, except GPU overlap gates):**

| Location | Status |
|---|---|
| `overlay/patch_scheduler_decode_floor.py` | Done: shared step policy, progress aging, aggregate prefill allowance, alignment-aware cap, fail-closed v2 anchors |
| Base `vllm/v1/core/sched/scheduler.py` / async completion integration | Done at patch time: both scheduling paths plus `observe_output` without a GPU sync |
| `start.sh` and supported alternate launchers, `.env.example` | Done: policy controls, numeric validation, identical rank env, effective-value logging |
| `README.md` and overlay docstring | Done: admission vs decode, `fair` still opt-in |
| `tests/test_scheduler_decode_floor.py` | Done: v2 apply, v1→v2 migrate, skip/cap/off/fair, 10k skip starvation, alignment table, abort prune |
| `tests/test_mixed_prefill_decode.py` or the reporter’s harness | Partial: CHARLIE vs ALPHA prefix; overlap tok/s and `--b-max-tokens` exist; still no B-during-A assertion, 5× whole-window gate, event-rate overlap metric, count-to-80 stops early |

The v1 installer returned immediately on `# [glm53-decode-floor]`. v2 uses `# [glm53-decode-floor:v2]` and migrates; the live head log on 2026-09-15 showed that migrate then `policy=fair`.

**Validation must gate both clients.** The scheduler unit tests now cover fair behavior on CPU. The mixed-load harness still starts B after A’s first token but reports A’s whole decode-window throughput, has no assertion that B prefills before A finishes, and permits a 5× ratio despite a comment mentioning 3×. It can therefore miss `skip` starvation and dilute a severe overlap-only slowdown. `overlap_tok_s` counts SSE events, not tokens. The 2026-09-15 GPU runs used a separate unique-token script because repeated `the` is cache-cheap and the stock task stops around 160 tokens. [Scheduler patch test](../tests/test_scheduler_decode_floor.py), [mixed-load test](../tests/test_mixed_prefill_decode.py).

Use the reporter’s sequence as the regression workload, with these additions:

- Repeat each arm at least three times in alternating order with fresh prefixes, keeping A alive throughout the intended overlap. Compare `skip`, `0`/unset, `0`/1024, any cap candidate, and the new policy. Hold ABLIT, effort, adaptive-k, image, and graph configuration constant.
- Cover B at 2k and 30k cold tokens; cover A with short and long actual histories. Exercise one decoder plus three newcomers and multiple decoders plus a newcomer. Add sustained arrivals below measured capacity, cached-prefix follow-ups, cancellation, and a partially prefilling request that later gains a decoding peer.
- Timestamp every content and reasoning SSE event. Report p50/p95/p99 and maximum delivery gaps during the actual overlap, plus A’s overlap throughput and post-prefill recovery. Include both reasoning and visible content. Use final usage/tokenizer counts for throughput validation; do not count one speculative SSE event as one token. Label client event gaps separately from server inter-token latency.
- Separate B’s arrival, first scheduled prefill, subsequent prefill progress, first generated token, and completion. `Running=2` is only an admission indicator. Collect scheduled prefill/decode tokens per step, elapsed step time, actual cap after alignment, graph mode, speculative acceptance, KV usage, and preemptions.
- Define a joint target before selecting a default. An example product gate is A retaining at least 70% of its solo rate during overlap, p99 client delivery gaps below 1 s, and B receiving a positive prefill chunk at least every 2 s while resources permit. These are proposed acceptance targets, not observed or guaranteed results. Also set a TTFT target for the 2k newcomer and report the 30k tradeoff; neither admitting B once nor protecting A alone is sufficient.
- Add CPU cases for waiting and running prefills, growing queue age, several prefills sharing one aggregate budget, alignment zero-progress cases and boundary crossings, async/speculative transitions, and abort cleanup. Add GPU correctness coverage for hybrid prefix reuse and chunk boundaries before shipping an alignment change.

For the 2026-09-14 diagnosis, the completed checks were source inspection at the exact recipe and pinned base, successful application/parsing/idempotence of the recipe patch on a temporary copy, execution of its helper for `skip`, `0`, and positive caps, and execution of the alignment method for the table above. The helper continued returning zero for 10,000 calls with a decoding peer.

For the 2026-09-15 follow-up, v2 is installed on the live TP2 serve, host scheduler/numeric tests passed, and three overlap arms were run (stock harness, longer A with repeated `the`, cold unique ~30k). Decode floor holds (~1.1×). Large-cold B TTFT during a 14 s decode does not. The remaining harness/JSON items in the validation list (reporter thinking essay, repeats, p99 gaps, 2k B, multi-newcomer) are still required before changing the published default.

#!/usr/bin/env python3
"""Calibrated numerical panel for quantization/kernel A/B (GLM-5.3-Flash class).

Teacher-forced prompt_logprobs on fixed texts; judge a candidate against the
model's OWN stock-vs-stock variation instead of fixed absolute thresholds.

    GLM53_BENCH_BASE=http://127.0.0.1:8888 python3 tests/bench_numerical_calibrated.py capture --out A.json
    python3 tests/bench_numerical_calibrated.py calibrate --out C.json A1.json A2.json A3.json A4.json [A5.json ...]
    python3 tests/bench_numerical_calibrated.py compare --calibration C.json A.json B.json
    python3 tests/bench_numerical_calibrated.py selftest

Why: a fixed "argmax agreement >= 0.99 / mean KL <= 0.01" gate fails on
unchanged stock repeats of this model family (near-tie tokens flip between
identical runs). Calibration measures that wobble first (>= 4 stock captures),
derives a tie margin m* and a flip-rate bound, and the comparison only flags
changes beyond stock's own variation. Screening panel, not full qualification.

Exit codes: 0 within calibrated variation, 1 FLAG, 2 unqualified inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("GLM53_BENCH_BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("GLM53_BENCH_MODEL", "GLM-5.3-Flash-EXL3")
MIN_STOCK_CAPTURES = 4
MARGIN_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
MARGIN_FLOOR, MARGIN_CAP = 0.25, 1.5

TEXTS = {
    "prose": (
        "Explain how a hash map handles collisions, covering separate chaining "
        "with linked lists, open addressing with linear probing, load factor "
        "thresholds, and amortized resizing cost. Be thorough and precise."
    ),
    "code": (
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
        "Explain the average-case time complexity of this implementation."
    ),
    "arithmetic": (
        "A train travels 120 km in 2 hours, then 180 km in the next 3 hours. "
        "What is its average speed over the whole journey? Show each step."
    ),
    "structured": "Count from 1 to 40. Output only the numbers, separated by spaces.",
    "dense_tokens": (
        " ".join(
            f"Entry {i}: node NODE{i % 7} reported checksum CK-{i:06d} after "
            f"the maintenance window; temperature {40 + (i * 7) % 23} C, "
            f"fan duty {30 + (i * 13) % 60} percent, no faults."
            for i in range(40)
        )
    ),
}


def _post(path: str, body: dict, timeout: float = 600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def parse_position(pos) -> tuple[dict, str, float]:
    """-> (probs, argmax token, margin in nats). margin=inf when runner-up unknown."""
    if not isinstance(pos, dict) or not pos:
        raise ValueError("missing position distribution")
    probs, logs = {}, {}
    for token, value in pos.items():
        logprob = value.get("logprob") if isinstance(value, dict) else value
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) \
                or not math.isfinite(logprob) or logprob > 0:
            raise ValueError("invalid log probability")
        logs[token] = logprob
        probs[token] = math.exp(logprob)
    argmax = max(logs, key=logs.get)
    ordered = sorted(logs.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else math.inf
    return probs, logs, argmax, margin


def kl_shared(ax: dict, by: dict) -> float | None:
    keys = ax.keys() & by.keys()
    if not keys or any(ax[k] <= 0 or by[k] <= 0 for k in keys):
        return None
    za, zb = sum(ax[k] for k in keys), sum(by[k] for k in keys)
    return sum((ax[k] / za) * math.log((ax[k] / za) / (by[k] / zb)) for k in keys)


def poisson_critical(lam: float, alpha: float = 0.05) -> int:
    """Smallest c with P(X >= c | lam) <= alpha.

    Exact for lam <= 500 (incremental Poisson pmf). Above that the same
    one-sided normal tail is used at the same alpha, so a weak calibration
    cannot silently raise the FLAG bar.
    """
    if lam <= 0:
        return 1
    if lam > 500:
        low, high = 0.0, 40.0  # bisect the one-sided normal quantile at alpha
        for _ in range(80):
            mid = (low + high) / 2
            if 0.5 * math.erfc(mid / math.sqrt(2)) > alpha:
                low = mid
            else:
                high = mid
        # One-count margin over the normal-tail quantile: this branch only runs
        # for weak calibrations and must never lower the FLAG bar there.
        return max(1, math.ceil(lam + ((low + high) / 2) * math.sqrt(lam)) + 1)
    term = math.exp(-lam)  # P(X = 0)
    tail = 1.0  # P(X >= 0)
    c = 0
    while tail > alpha and c <= 10 * lam + 50:
        tail -= term  # now P(X >= c + 1)
        c += 1
        term = term * lam / c
    return c


def load_receipt(path: str) -> dict:
    receipt = json.loads(Path(path).read_text())
    texts = receipt.get("texts")
    if not isinstance(texts, dict) or not texts or not receipt.get("model"):
        raise ValueError(f"{path}: not a capture receipt")
    for name, rec in texts.items():
        positions = rec.get("positions")
        if rec.get("mode") != "prompt_logprobs" or not isinstance(positions, list) or len(positions) < 2:
            raise ValueError(f"{path}/{name}: unqualified capture")
    return receipt


def parsed_texts(receipt: dict) -> dict:
    out = {}
    for name, rec in receipt["texts"].items():
        rows = []
        for pos in rec["positions"][1:]:
            probs, logs, argmax, margin = parse_position(pos)
            rows.append((probs, logs, argmax, margin))
        out[name] = rows
    return out


# ---------------------------------------------------------------- capture

def capture(out: str) -> int:
    res = {"base": BASE, "model": MODEL, "texts": {}}
    for name, text in TEXTS.items():
        rec = {"prompt_chars": len(text), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
        try:
            status, d = _post("/v1/completions", {
                "model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
                "echo": True, "logprobs": 1, "prompt_logprobs": 20,
            })
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            print(f"{name}: capture request failed ({exc}); capture is unqualified",
                  file=sys.stderr)
            return 2
        pl = (d["choices"][0].get("prompt_logprobs") or []) if status == 200 else []
        if not isinstance(pl, list) or len(pl) < 2 or not any(pl):
            print(f"{name}: prompt_logprobs unavailable; capture is unqualified", file=sys.stderr)
            return 2
        rec.update(mode="prompt_logprobs", positions=pl)
        print(f"{name}: positions={len(pl)}", flush=True)
        res["texts"][name] = rec
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res))
    print("wrote", out)
    return 0


# ------------------------------------------------------------- calibrate

def calibrate(out: str, inputs: list[str]) -> int:
    if len(inputs) < MIN_STOCK_CAPTURES:
        print(f"calibrate needs >= {MIN_STOCK_CAPTURES} stock receipts", file=sys.stderr)
        return 2
    receipts = [load_receipt(p) for p in inputs]
    model = receipts[0]["model"]
    if any(r["model"] != model for r in receipts):
        print("UNQUALIFIED: mixed models in calibration set")
        return 2
    names = receipts[0]["texts"].keys()
    if any(r["texts"].keys() != receipts[0]["texts"].keys() for r in receipts):
        print("UNQUALIFIED: calibration texts differ")
        return 2
    cal = {"schema": "calibrated-numerical-panel/v1", "model": model,
           "stock_captures": len(receipts), "inputs_sha256":
           [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs], "texts": {}}
    for name in names:
        if any(r["texts"][name]["prompt_sha256"] != receipts[0]["texts"][name]["prompt_sha256"]
               for r in receipts):
            print(f"UNQUALIFIED: {name} prompt changed between calibration captures")
            return 2
        caps = [parsed_texts(r)[name] for r in receipts]
        n = len(caps[0])
        if any(len(c) != n for c in caps):
            print(f"UNQUALIFIED: {name} position count differs")
            return 2
        # cross-capture disagreement vs position min-margin -> tie cut m*
        min_margin = [min(c[i][3] for c in caps) for i in range(n)]
        disagree = [len({c[i][2] for c in caps}) > 1 for i in range(n)]
        m_star = MARGIN_CAP
        for m in MARGIN_GRID:
            if not any(d and mm >= m for d, mm in zip(disagree, min_margin)):
                m_star = max(m, MARGIN_FLOOR)
                break
        conf = [i for i in range(n) if min_margin[i] >= m_star]
        ties = [i for i in range(n) if min_margin[i] < m_star]
        f = sum(disagree[i] for i in conf)
        n_conf = len(conf)
        if n_conf == 0:
            print(f"UNQUALIFIED: {name} has no confident positions at m*={m_star}")
            return 2
        r_up = 3.0 / n_conf if f == 0 else min(1.0, (f + 2 * math.sqrt(f) + 2) / n_conf)
        # consensus token per position (majority; tie -> higher mean logprob)
        def consensus(captures):
            cons = []
            for i in range(n):
                votes = {}
                for c in captures:
                    tok = c[i][2]
                    votes[tok] = votes.get(tok, 0) + 1
                cons.append(max(votes, key=votes.get))
            return cons
        cons = consensus(caps)
        # stock self-spread of the consensus token's logprob (confident positions)
        def token_logprob(capture_row, token):
            return capture_row[1].get(token)
        spreads = []
        for i in conf:
            vals = [token_logprob(c[i], cons[i]) for c in caps]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                spreads.append(max(vals) - min(vals))
        tau = max(spreads) if spreads else 0.0
        # pairwise null KL (confident positions only: gate must compare like
        # with like — tie-position renormalization noise would inflate it)
        kls, tie_dis = [], []
        for a in range(len(caps)):
            for b in range(len(caps)):
                if a == b:
                    continue
                for i in conf:
                    kl = kl_shared(caps[a][i][0], caps[b][i][0])
                    if kl is not None:
                        kls.append(kl)
                tie_dis += [caps[a][i][2] != caps[b][i][2] for i in ties]
        cal["texts"][name] = {
            "positions": n, "m_star": m_star, "n_confident": n_conf, "n_tie": len(ties),
            "prompt_sha256": receipts[0]["texts"][name]["prompt_sha256"],
            "stock_disagreements_confident": f, "r_up": r_up,
            "null_kl_mean": sum(kls) / len(kls) if kls else None,
            "null_kl_max": max(kls) if kls else None,
            "tau_stock_spread": tau,
            "tie_disagreement_rate": (sum(tie_dis) / len(tie_dis)) if tie_dis else 0.0,
        }
        print(f"{name:14s} pos={n} m*={m_star:.2f} conf={n_conf} flips={f} "
              f"r_up={r_up:.2e} nullKLmax={(cal['texts'][name]['null_kl_max'] or 0):.2e}")
    Path(out).write_text(json.dumps(cal, indent=1))
    print("wrote", out)
    return 0


# --------------------------------------------------------------- compare

def compare(cal_path: str, a_paths: list[str], b_paths: list[str]) -> int:
    """Consensus compare: any number of stock (A) and candidate (B) receipts.

    A position counts as a flip only when the CANDIDATE CONSENSUS argmax
    (majority across B captures) differs from the STOCK CONSENSUS argmax
    (majority across A captures). Systematic damage flips whole consensus
    groups; random near-tie wobble does not. Single receipts per side also
    work (degenerates to a plain pair comparison with less power)."""
    cal = json.loads(Path(cal_path).read_text())
    if cal.get("schema") != "calibrated-numerical-panel/v1":
        print("UNQUALIFIED: not a calibration receipt")
        return 2
    ra = [load_receipt(p) for p in a_paths]
    rb = [load_receipt(p) for p in b_paths]
    if any(r["model"] != cal["model"] for r in ra + rb):
        print("UNQUALIFIED: model differs from calibration")
        return 2
    pa = [parsed_texts(r) for r in ra]
    pb = [parsed_texts(r) for r in rb]
    flagged, invalid = 0, False
    print(f"{'text':14s} {'pos':>4s} {'consFlips':>9s} {'crit':>4s} {'sep':>11s} "
          f"{'tieDis':>6s} {'stockTie':>8s} {'KL':>9s} {'KLcrit':>9s}")
    for name in cal["texts"]:
        if any(name not in x for x in pa + pb):
            print(f"{name}: UNQUALIFIED missing from a capture")
            invalid = True
            continue
        tc = cal["texts"][name]
        n = tc["positions"]
        rows_a = [x[name] for x in pa]
        rows_b = [x[name] for x in pb]
        if any(len(r) != n for r in rows_a + rows_b):
            print(f"{name}: UNQUALIFIED position count differs from calibration")
            invalid = True
            continue
        fingerprint = tc.get("prompt_sha256")
        if fingerprint is not None and any(
                r["texts"][name].get("prompt_sha256") != fingerprint for r in ra + rb):
            print(f"{name}: UNQUALIFIED prompt differs from calibration")
            invalid = True
            continue
        flips = tie_dis = sep_exceed = 0
        kls = []
        # stock consensus token per position (majority across A captures)
        cons = []
        for i in range(n):
            votes = {}
            for r in rows_a:
                votes[r[i][2]] = votes.get(r[i][2], 0) + 1
            cons.append(max(votes, key=votes.get))
        tau_gate = max(0.25, tc.get("tau_stock_spread") or 0.0)
        for i in range(n):
            margins = [r[i][3] for r in rows_a + rows_b]
            ta = {r[i][2] for r in rows_a}
            tb = {r[i][2] for r in rows_b}
            kl_pairs = [(x[name][i][0], y[name][i][0]) for x in pa for y in pb]
            amax = max(tb, key=[r[i][2] for r in rows_b].count)
            if min(margins) >= tc["m_star"]:
                flips += amax != cons[i]
                kls += [k for k in (kl_shared(x, y) for x, y in kl_pairs) if k is not None]
            else:
                tie_dis += amax != cons[i]
            # separation: every candidate capture's logprob for the stock
            # consensus token strictly outside every stock capture's, by > tau.
            # The stock side must be COMPLETE: a token stock itself lost in some
            # capture is not a reference, and partial ranges would widen the
            # stock span and hide real shifts.
            svals = [r[i][1].get(cons[i]) for r in rows_a]
            if any(v is None for v in svals):
                continue
            cvals = [r[i][1].get(cons[i]) for r in rows_b]
            cvals = [v for v in cvals if v is not None]
            if not cvals:
                sep_exceed += 1  # token vanished from EVERY candidate top-k
                continue
            if (min(svals) - max(cvals) > tau_gate
                    or min(cvals) - max(svals) > tau_gate):
                sep_exceed += 1
        lam = tc["n_confident"] * 2 * tc["r_up"]
        crit = poisson_critical(lam)
        n_tie = max(1, tc["n_tie"])
        tie_rate = tie_dis / n_tie
        mean_kl = sum(kls) / len(kls) if kls else 0.0
        kl_crit = 2 * (tc["null_kl_max"] or 0.0)
        text_flag = flips >= crit or mean_kl > kl_crit or sep_exceed >= 2
        flagged += text_flag
        annotation = ""
        if tie_rate > 2 * tc["tie_disagreement_rate"] + 0.05:
            annotation += " tieRate!"
        if text_flag:
            annotation += " FLAG"
        print(f"{name:14s} {n:4d} {flips:9d} {crit:4d} sep>{tau_gate:.2f}:{sep_exceed:3d} "
              f"{tie_rate:6.3f} {tc['tie_disagreement_rate']:8.3f} {mean_kl:9.2e} {kl_crit:9.2e}{annotation}")
    if invalid:
        print("UNQUALIFIED")
        return 2
    print("FLAG (exceeds calibrated stock variation)" if flagged
          else "WITHIN CALIBRATED STOCK VARIATION (shared top-k screening)")
    return 1 if flagged else 0


# --------------------------------------------------------------- selftest

def _base_rows() -> list[tuple[int, int, float]]:
    """Fixed row skeleton shared by every synthetic capture (deterministic)."""
    rng = random.Random(7)
    rows = []
    for i in range(200):
        top = rng.randrange(100000, 999999)
        gap = 0.02 if i % 10 == 0 else 4.0  # near-tie rows wobble; confident rows don't
        rows.append((top, top + 1, gap))
    return rows


_BASE = _base_rows()


def _synthetic(seed: int, flips: dict[str, int], kl_boost: float) -> dict:
    """Deterministic synthetic capture: near-tie rows wobble with `seed`;
    `flips` counts extra CONFIDENT-row flips; `kl_boost` shifts the runner-up
    probability mass on confident rows (a distribution shift)."""
    rng = random.Random(seed)
    texts = {}
    for name in TEXTS:
        n_flips = flips.get(name, 0)
        rows, done = [], 0
        for i, (top, second, gap) in enumerate(_BASE):
            eff_gap, eff_top, eff_second = gap, top, second
            if gap < 0.25:  # near-tie: wobble the winner per capture
                if rng.random() < 0.5:
                    eff_top, eff_second = second, top
            elif done < n_flips:  # damage: flip a confident row
                eff_top, eff_second = second, top
                done += 1
            if gap >= 0.25 and kl_boost:
                eff_gap = max(0.3, gap - kl_boost)
            rows.append({str(eff_top): {"logprob": 0.0},
                         str(eff_second): {"logprob": -eff_gap},
                         str(top + 13): {"logprob": -eff_gap - 3.0}})
        texts[name] = {"prompt_sha256": hashlib.sha256(TEXTS[name].encode()).hexdigest(),
                       "mode": "prompt_logprobs", "positions": [None] + rows}
    return {"model": MODEL, "texts": texts}


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for k in range(5):  # stock wobbles only at near-ties (seed noise)
            p = Path(tmp) / f"s{k}.json"
            p.write_text(json.dumps(_synthetic(1000 + k, {}, 0.0)))
            paths.append(str(p))
        calp = Path(tmp) / "c.json"
        rc = calibrate(str(calp), paths)
        assert rc == 0, "calibrate failed"
        clean = Path(tmp) / "clean.json"
        clean.write_text(json.dumps(_synthetic(999, {}, 0.0)))
        damaged = Path(tmp) / "damaged.json"
        damaged.write_text(json.dumps(_synthetic(999, {"prose": 15, "dense_tokens": 12}, 0.0)))
        klshift = Path(tmp) / "kl.json"
        klshift.write_text(json.dumps(_synthetic(999, {}, 2.0)))
        rc0 = compare(str(calp), [str(paths[0])], [str(clean)])
        rc1 = compare(str(calp), [str(paths[0])], [str(damaged)])
        rc2 = compare(str(calp), [str(paths[0])], [str(klshift)])
        assert rc0 == 0, "clean candidate should be WITHIN"
        assert rc1 == 1, "systematic confident flips should FLAG"
        assert rc2 == 1, "KL shift should FLAG"

        def tail_at(start: int, lam: float) -> float:
            bound = int(lam + 12 * math.sqrt(lam)) + 40
            return sum(math.exp(-lam) * lam ** k / math.factorial(k)
                       for k in range(max(0, start), bound + 1))

        # The critical count is the documented tail definition: the smallest c
        # with P(X >= c) <= alpha, not alpha one step early.
        for lam in (0.5, 5.0, 30.0):
            crit = poisson_critical(lam)
            assert tail_at(crit, lam) <= 0.05 < tail_at(crit - 1, lam), (lam, crit)

        # A capture whose prompt does not match the calibration is unqualified
        # instead of being compared position-by-position.
        tampered = json.loads(json.dumps(_synthetic(999, {}, 0.0)))
        tampered["texts"]["prose"]["prompt_sha256"] = "0" * 64
        tampered_path = Path(tmp) / "tampered.json"
        tampered_path.write_text(json.dumps(tampered))
        assert compare(str(calp), [str(paths[0])], [str(tampered_path)]) == 2, \
            "prompt mismatch should be UNQUALIFIED"

        print("selftest: clean WITHIN, flips FLAG, kl-shift FLAG, "
              "poisson tail + prompt pin OK -> OK")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--out", required=True)
    k = sub.add_parser("calibrate"); k.add_argument("--out", required=True); k.add_argument("inputs", nargs="+")
    p = sub.add_parser("compare")
    p.add_argument("--calibration", required=True)
    p.add_argument("a", nargs="+", help="stock receipt(s)")
    p.add_argument("b", nargs="+", help="candidate receipt(s)")
    sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "capture":
        return capture(args.out)
    if args.cmd == "calibrate":
        return calibrate(args.out, args.inputs)
    if args.cmd == "compare":
        return compare(args.calibration, list(args.a), list(args.b))
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())

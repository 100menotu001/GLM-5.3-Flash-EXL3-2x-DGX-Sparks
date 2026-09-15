#!/usr/bin/env bash
# scripts/spec-accept-gate.sh: exit-code contract against synthetic /metrics.
#
# A stub `curl` first on PATH serves an inert fixture file, so no server, GPU,
# container or network is involved. Each case pins a behaviour the gate's
# callers depend on being distinguishable:
#   1. >100 drafts and no per-position series  -> rc 2, named diagnostic
#      (before this, the run died rc 1 printing only drafts_total)
#   2. position="0" absent while others exist  -> rc 2, named diagnostic
#   3. two label sets for one series           -> rc 2, no ratio emitted
#   4. pinned pos0 -> rc 1 FAIL, healthy decay -> rc 0 PASS, <100 drafts ->
#      rc 0 SKIP: the three verdicts stay distinct from the rc 2 diagnostics
#   5. default endpoint is loopback; an explicit argument still wins
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GATE="$HERE/../scripts/spec-accept-gate.sh"
[ -f "$GATE" ] || { echo "spec-accept-gate.sh not found" >&2; exit 1; }
fail=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/bin"
export GATE_FIXTURE="$WORK/metrics.txt" GATE_ARGV="$WORK/curl-argv.txt"
cat > "$WORK/bin/curl" <<'SHIM'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GATE_ARGV"
[ -n "${GATE_CURL_FAIL:-}" ] && exit 7
cat "$GATE_FIXTURE"
SHIM
chmod +x "$WORK/bin/curl"
export PATH="$WORK/bin:$PATH"

# fixture <drafts_total> [<position>:<accepted_tokens> ...] -> $GATE_FIXTURE
fixture() {
    { echo "# TYPE vllm:spec_decode_num_drafts_total counter"
      echo "vllm:spec_decode_num_drafts_total{model_name=\"glm\"} ${1}.0"
      echo "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter"
      shift
      for pair in "$@"; do
          echo "vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name=\"glm\",position=\"${pair%%:*}\"} ${pair##*:}.0"
      done
    } > "$GATE_FIXTURE"
}

out=""
rc=0
run() { : > "$GATE_ARGV"; out="$("$GATE" "$@" 2>&1)"; rc=$?; }
check() { # $1 description, $2 expected rc, $3 required output substring
    if [ "$rc" != "$2" ]; then
        echo "FAIL $1: rc $rc (want $2): $out"; fail=1; return
    fi
    case "$out" in
        *"$3"*) echo "ok   $1 (rc $rc)" ;;
        *) echo "FAIL $1: rc $rc but output lacks [$3]: $out"; fail=1 ;;
    esac
}

# --- missing and mismatched series ----------------------------------------
fixture 150
run
check "150 drafts, no per-position series" 2 "no accepted_tokens_per_pos_total series"

fixture 150 1:90 2:60
run
check "position 0 missing" 2 'position="0" is missing'

fixture 150 0:150 1:20
printf '%s\n' 'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="other",position="0"} 3.0' >> "$GATE_FIXTURE"
run
check "two label sets for position 0" 2 "ambiguous label sets"
case "$out" in
    *"ratio="*) echo "FAIL two label sets: emitted a ratio anyway: $out"; fail=1 ;;
    *) echo "ok   two label sets: no ratio emitted" ;;
esac

fixture 150 0:120 1:90
printf '%s\n' 'vllm:spec_decode_num_drafts_total{model_name="other"} 7.0' >> "$GATE_FIXTURE"
run
check "two label sets for the denominator" 2 "ambiguous label sets"

# --- the PASS / FAIL / SKIP verdicts stay distinct ------------------------
fixture 150 0:150 1:150
run
check "pinned pos0" 1 "FAIL: pos0 acceptance"

fixture 100 0:80 1:60 2:40
run
check "healthy decay at exactly 100 drafts" 0 "PASS: pos0 acceptance"

fixture 99 0:70
run
check "99 drafts" 0 "SKIP: only 99 drafts"

# --- endpoint -------------------------------------------------------------
fixture 100 0:80 1:60
run
check "loopback default verdict" 0 "PASS"
case "$(cat "$GATE_ARGV")" in
    *"http://127.0.0.1:8888/metrics"*) echo "ok   default endpoint is loopback" ;;
    *) echo "FAIL default endpoint: $(cat "$GATE_ARGV")"; fail=1 ;;
esac
case "$(cat "$GATE_ARGV")" in
    *"192.168."*) echo "FAIL default endpoint is still site-specific: $(cat "$GATE_ARGV")"; fail=1 ;;
    *) echo "ok   no site-specific host requested" ;;
esac

run "http://127.0.0.1:9999"
case "$(cat "$GATE_ARGV")" in
    *"http://127.0.0.1:9999/metrics"*) echo "ok   explicit argument overrides the default" ;;
    *) echo "FAIL explicit endpoint: $(cat "$GATE_ARGV")"; fail=1 ;;
esac

# --- unreachable endpoint -------------------------------------------------
export GATE_CURL_FAIL=1
run
unset GATE_CURL_FAIL
check "unreachable endpoint" 2 "cannot read http://127.0.0.1:8888/metrics"

[ "$fail" = 0 ] && echo "spec-accept-gate tests: PASS"
exit $fail

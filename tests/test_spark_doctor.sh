#!/usr/bin/env bash
# spark_doctor.sh: CPU-only behavioral coverage. Nothing here contacts a node, a
# Docker daemon, a GPU or the network: every probe the doctor makes (ping, ssh,
# ip, ibv_devinfo, nvidia-smi, docker, curl) is shadowed by a stub placed first on
# PATH, and the doctor runs from a throwaway copy of scripts/ so the repository's
# own .env is never sourced.
#
# Recipe:  bash tests/test_spark_doctor.sh
#
# Contracts pinned here:
#   * an unjudgeable GPU_MEM_UTIL FAILs (exit 1) instead of falling through to PASS
#   * 0.97 and 0.65 warn, 0.85 passes, and the printed band is the 0.70 - 0.95 checked
#   * the default tracks start.sh/.env.example (0.85, not the retired 0.87)
#   * an HTTP 200 is reported as endpoint reachability, never as engine health
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
DOCTOR="$HERE/../scripts/spark_doctor.sh"
[ -f "$DOCTOR" ] || { echo "scripts/spark_doctor.sh not found" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/scripts" "$tmp/bin"
cp "$DOCTOR" "$tmp/scripts/spark_doctor.sh"

stub() { # $1 = command name; stdin = the stub body
    cat > "$tmp/bin/$1"
    chmod +x "$tmp/bin/$1"
}

stub ping <<'EOF'
#!/usr/bin/env bash
exit "${STUB_PING_RC:-0}"
EOF
stub ssh <<'EOF'
#!/usr/bin/env bash
exit "${STUB_SSH_RC:-0}"
EOF
stub ip <<'EOF'
#!/usr/bin/env bash
printf '%s %s %s\n' dev "${STUB_LINK_STATE:-UP}" aa:bb:cc:dd:ee:ff
EOF
stub ibv_devinfo <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${STUB_IB_STATE:-PORT_ACTIVE}"
EOF
stub nvidia-smi <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *memory.total*) echo "131072 MiB" ;;
    *) echo "NVIDIA GB10" ;;
esac
EOF
stub docker <<'EOF'
#!/usr/bin/env bash
exit "${STUB_DOCKER_RC:-0}"
EOF
stub curl <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    case "$arg" in
        */v1/models) printf '{"data":[{"id":"GLM-5.3-Flash-EXL3"}]}\n'; exit 0 ;;
    esac
done
printf '%s\n' "${STUB_HEALTH_CODE:-200}"
EOF

out=""
rc=0
run_doctor() { # $1 = GPU_MEM_UTIL ("" = leave unset), $2 = stub health code
    local -a envv=(PATH="$tmp/bin:$PATH" STUB_HEALTH_CODE="$2")
    [ -n "$1" ] && envv+=("GPU_MEM_UTIL=$1")
    env -u GPU_MEM_UTIL "${envv[@]}" bash "$tmp/scripts/spark_doctor.sh" 2>&1 \
        | sed $'s/\x1b\\[[0-9;]*m//g' > "$tmp/doctor.out"
    rc="${PIPESTATUS[0]}"
    out="$(cat "$tmp/doctor.out")"
}

fail=0
expect_rc() { # $1 = label, $2 = expected exit status
    if [ "$rc" -eq "$2" ]; then echo "ok   $1"
    else echo "FAIL $1 (exit $rc, want $2)"; fail=1; fi
}
expect() { # $1 = label, $2 = text the doctor must report
    case "$out" in
        *"$2"*) echo "ok   $1" ;;
        *) echo "FAIL $1 (not reported: $2)"; fail=1 ;;
    esac
}
reject() { # $1 = label, $2 = text the doctor must no longer report
    case "$out" in
        *"$2"*) echo "FAIL $1 (still reported: $2)"; fail=1 ;;
        *) echo "ok   $1" ;;
    esac
}

[ -x "$DOCTOR" ] && echo "ok   ./scripts/spark_doctor.sh is executable" \
    || { echo "FAIL ./scripts/spark_doctor.sh is not executable"; fail=1; }

# ---- default util: start.sh / .env.example's 0.85, inside the accepted band ----
run_doctor "" 200
expect_rc "default GPU_MEM_UTIL: clean host exits 0" 0
expect "default GPU_MEM_UTIL is 0.85" "[PASS] GPU_MEM_UTIL=0.85"
expect "band text states the checked interval" "0.70 - 0.95"
reject "band text no longer describes 0.80 - 0.90" "0.80 - 0.90"

# ---- unjudgeable util: FAIL, never a silent PASS ----
for bad in "not-a-util" "0" "1.5"; do
    run_doctor "$bad" 200
    expect_rc "GPU_MEM_UTIL='$bad': exits 1" 1
    expect "GPU_MEM_UTIL='$bad': reported as FAIL" "[FAIL] GPU_MEM_UTIL must be greater than 0 and at most 1"
    reject "GPU_MEM_UTIL='$bad': never reported as PASS" "[PASS] GPU_MEM_UTIL"
done

# ---- band warnings on either side of 0.70 - 0.95 ----
run_doctor "0.97" 200
expect_rc "GPU_MEM_UTIL=0.97: warning only, exits 0" 0
expect "GPU_MEM_UTIL=0.97: warned above the ceiling" "[WARN] GPU_MEM_UTIL=0.97 is above the 0.95 ceiling"

run_doctor "0.65" 200
expect_rc "GPU_MEM_UTIL=0.65: warning only, exits 0" 0
expect "GPU_MEM_UTIL=0.65: warned below the floor" "[WARN] GPU_MEM_UTIL=0.65 is below the 0.70 floor"

# ---- endpoint reachability is not engine health ----
run_doctor "0.85" 200
expect_rc "health 200: exits 0" 0
expect "health 200: reported as an HTTP 200" "HTTP 200"
reject "health 200: not called 'Active & Healthy'" "Active & Healthy"
reject "summary: no serving-readiness claim" "ready for GLM-5.3-Flash EXL3 serving"

run_doctor "0.85" "000"
expect_rc "nothing listening: exits 0" 0
reject "nothing listening: no HTTP 200 claim" "HTTP 200"

[ "$fail" = 0 ] && echo "spark_doctor.sh tests: PASS"
exit "$fail"

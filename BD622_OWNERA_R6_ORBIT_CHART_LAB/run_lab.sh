#!/usr/bin/env bash
# Single no-argument entry point for the sealed C-R6 orbit-chart lab.
# Exit codes: 0 all-pass, 10 scientific FAIL, 20 mechanical ERROR,
# 30 isolation/contract violation. See EXPERIMENT_CONTRACT.json.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -ne 0 ]; then
    echo "ARGS_FORBIDDEN"
    exit 30
fi

AUTHORED="EXPERIMENT_CONTRACT.json run_lab.sh mechanism.py fixtures.py mutants.py test_mechanism.py"

if [ "${R6LAB_INNER:-}" != "1" ]; then
    if ! command -v unshare >/dev/null 2>&1; then
        echo "NO_UNSHARE"
        exit 30
    fi
    if ! sha256sum --quiet -c MANIFEST_AUTHORED.sha256; then
        echo "SEAL_MISMATCH"
        exit 30
    fi
    rm -rf raw RESULTS.json MUTANTS.json MANIFEST.sha256
    mkdir -p raw
    {
        echo "{"
        echo " \"contract_sha256\": \"$(sha256sum EXPERIMENT_CONTRACT.json | cut -d' ' -f1)\","
        echo " \"git_head\": \"$(git rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)\","
        echo " \"utc_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo " \"uid\": $(id -u),"
        echo " \"python\": \"$(python3 -c 'import sys; print(sys.version.split()[0])')\","
        echo " \"numpy\": \"$(python3 -c 'import numpy; print(numpy.__version__)')\","
        echo " \"scipy\": \"$(python3 -c 'import scipy; print(scipy.__version__)')\""
        echo "}"
    } > raw/SEAL_CHRONOLOGY.json
    exec unshare -rn env R6LAB_INNER=1 PYTHONHASHSEED=0 \
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash "$0"
fi

# ---- inner phase: user+network namespace, network off ----
ulimit -v 4194304

if ! python3 - > raw/ISOLATION.json <<'PY'
import json, os, socket, sys

def probe(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((host, port))
        reached = True
    except OSError:
        reached = False
    finally:
        s.close()
    return reached

report = {
    "probe_127_0_0_1_9": probe("127.0.0.1", 9),
    "probe_8_8_8_8_53": probe("8.8.8.8", 53),
    "uid_in_namespace": os.getuid(),
}
print(json.dumps(report, sort_keys=True))
sys.exit(0 if not (report["probe_127_0_0_1_9"] or report["probe_8_8_8_8_53"]) else 1)
PY
then
    echo "ISOLATION_FAIL"
    exit 30
fi

if ! python3 - <<'PY'
import sys, numpy, scipy
assert sys.version.split()[0] == "3.12.3", sys.version
assert numpy.__version__ == "2.4.2", numpy.__version__
assert scipy.__version__ == "1.17.0", scipy.__version__
PY
then
    echo "VERSION_MISMATCH"
    exit 30
fi

sha256sum $AUTHORED > raw/pre_hashes.txt

overall=0
for tag in run_primary replay_1 replay_2; do
    mkdir -p "raw/$tag"
    set +e
    timeout 1800 python3 test_mechanism.py --full --out "raw/$tag/RESULTS.json" \
        > "raw/$tag/stdout.log" 2> "raw/$tag/stderr.log"
    rc=$?
    if [ $rc -eq 0 ]; then
        timeout 1800 python3 mutants.py --out "raw/$tag/MUTANTS.json" \
            >> "raw/$tag/stdout.log" 2>> "raw/$tag/stderr.log"
        rc=$?
    fi
    set -e
    case $rc in
        0|10|20|30) ;;
        *) rc=20 ;;
    esac
    echo "phase $tag exit=$rc" >> raw/PHASES.txt
    if [ $rc -ne 0 ]; then
        overall=$rc
        break
    fi
done

if [ $overall -eq 0 ]; then
    if ! cmp -s raw/run_primary/RESULTS.json raw/replay_1/RESULTS.json ||
       ! cmp -s raw/run_primary/RESULTS.json raw/replay_2/RESULTS.json ||
       ! cmp -s raw/run_primary/MUTANTS.json raw/replay_1/MUTANTS.json ||
       ! cmp -s raw/run_primary/MUTANTS.json raw/replay_2/MUTANTS.json; then
        echo "REPLAY_MISMATCH"
        overall=10
    else
        cp raw/run_primary/RESULTS.json RESULTS.json
        cp raw/run_primary/MUTANTS.json MUTANTS.json
    fi
fi

sha256sum $AUTHORED > raw/post_hashes.txt
if ! cmp -s raw/pre_hashes.txt raw/post_hashes.txt; then
    echo "SOURCE_MUTATED"
    exit 30
fi

{
    echo "authored_files=6"
    echo "authored_loc=$(cat $AUTHORED | wc -l)"
    echo "wall_seconds=$SECONDS"
    echo "exit_code=$overall"
    echo "token_use_exact=UNAVAILABLE - no reliable stage-scoped counter"
    echo "runtime_behavior_changed=false"
    echo "physics_behavior_changed=false"
    echo "scientific_blocker_movement_ratio=0.00"
} > raw/ACCOUNTING.txt
cat raw/ACCOUNTING.txt

find . -type f ! -name MANIFEST.sha256 | sed 's|^\./||' | LC_ALL=C sort \
    | xargs sha256sum > MANIFEST.sha256

if [ $overall -eq 0 ]; then
    echo "REQUEST_OWNER_B_AUTHORIZATION_FOR_TARGET_REPLAY_REVIEW"
else
    echo "DO_NOT_REOPEN"
fi
exit $overall

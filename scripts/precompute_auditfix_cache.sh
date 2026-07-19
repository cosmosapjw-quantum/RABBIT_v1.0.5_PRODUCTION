#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/auditfix_cache"
mkdir -p "$CACHE/typeI" "$CACHE/chi2"
cd "$ROOT"

python3 -m pip install -e .

# fig03
for s in 0.1 0.3 0.5; do
  python3 scripts/dump_typeI_point.py --sigma "$s" --traj --out "$CACHE/typeI/fig03_sigma_${s}.json"
done

# fig16 baseline + sweeps
python3 scripts/dump_typeI_point.py --sigma 0.3 --nq 30 --nmu 12 --out "$CACHE/typeI/fig16_baseline.json"
for nmu in 4 6 8 12; do
  python3 scripts/dump_typeI_point.py --sigma 0.3 --nq 20 --nmu "$nmu" --out "$CACHE/typeI/fig16_nmu_${nmu}.json"
done
for nq in 10 14 20 30; do
  python3 scripts/dump_typeI_point.py --sigma 0.3 --nq "$nq" --nmu 12 --out "$CACHE/typeI/fig16_nq_${nq}.json"
done

# fig19
for cl in 0 1 2; do
  python3 scripts/dump_typeI_point.py --sigma 0.0 --cl "$cl" --out "$CACHE/typeI/fig19_cl${cl}.json"
done

# fig20
for s in 0.0 0.1 0.3 0.5; do
  python3 scripts/dump_typeI_point.py --sigma "$s" --mode linearized --out "$CACHE/typeI/fig20_linearized_sigma_${s}.json"
  python3 scripts/dump_typeI_point.py --sigma "$s" --mode characteristic --out "$CACHE/typeI/fig20_characteristic_sigma_${s}.json"
  python3 scripts/dump_typeI_point.py --sigma "$s" --mode characteristic --teff --out "$CACHE/typeI/fig20_characteristic_teff_sigma_${s}.json"
done

# fig21 coarse grid (5x5 quick)
for eta in 5.96 6.03 6.10 6.17 6.24; do
  for s in 0.0 0.15 0.30 0.45 0.60; do
    python3 scripts/dump_chi2_point.py --eta10 "$eta" --sigma "$s" --out "$CACHE/chi2/eta_${eta}_sigma_${s}.json"
  done
done

echo "Cache built in $CACHE"

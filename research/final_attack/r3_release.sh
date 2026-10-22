#!/usr/bin/env bash
# R3 release after ml.make_submission_v2 (same steps as research/entity_lookup/build_release.sh from "package" on).
set -euo pipefail
REPO=/home/makhzaun/Документы/Gi/hakaton; FA=/home/makhzaun/Документы/Gi/hakaton-research/work/final_attack
L=$FA/release_logs; mkdir -p $L; S=$L/R3_status.log
cd "$REPO"; export PYTHONPATH=. OMP_NUM_THREADS=4; PY=.venv/bin/python
fail() { echo "[$(date +%T)] FAIL $*" >> $S; exit 1; }
DEL=artifacts/delivery/20261022T041000Z-r3negbp; RUN=20261022T041000Z-r3negbp
VERSION=entity-te15lkbp16-mcs100-nb-negbp-$RUN; PKG=$VERSION-pkg2
SUB=$(tail -1 $FA/r3_submission.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["csv"])')
echo "[$(date +%T)] submission $SUB $(sha256sum "$SUB" | cut -d' ' -f1)" > $S
cmp "$SUB" $FA/csv/r3_rule_submission.csv && echo "[$(date +%T)] package CSV == rule CSV (byte-identical)" >> $S || fail csv-mismatch
$PY -m ml.package_delivery --model-dir "$DEL" --wheelhouse artifacts/offline-wheels --output-dir artifacts/handoff/$PKG > $L/R3_package.log 2>&1 || fail package
(cd artifacts/handoff && tar -czf $PKG.tar.gz $PKG && sha256sum $PKG.tar.gz > $PKG.tar.gz.sha256)
ASHA=$(cut -d' ' -f1 artifacts/handoff/$PKG.tar.gz.sha256)
echo "[$(date +%T)] archive $ASHA" >> $S
OMP_NUM_THREADS=1 $PY -m ml.verify_delivery --archive artifacts/handoff/$PKG.tar.gz --sha256 $ASHA > $L/R3_verify_delivery.log 2>&1 || fail offline-verify
grep -q '"status": "PASS"' $L/R3_verify_delivery.log || fail offline-verify-status
EVID=$(tail -1 $L/R3_verify_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["evidence"])')
echo "[$(date +%T)] offline verification PASS $EVID" >> $S
R=artifacts/release/R3-$RUN
mkdir -p $R/checks
cp -a "$SUB" $R/submission_R3.csv; cp -a "$(dirname "$SUB")/submission.json" $R/submission_R3.json
cp -a artifacts/handoff/$PKG.tar.gz artifacts/handoff/$PKG.tar.gz.sha256 $R/
cp -a "$EVID" $R/offline-verification
cp -a $FA/r3_submission.log $R/checks/
echo "[$(date +%T)] release dir $R" >> $S
(cd $R && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum -c --quiet SHA256SUMS) || fail sha256sums
echo "[$(date +%T)] READY $R (exhaustive single_vs_batch follows; ~20 min)" >> $S
OMP_NUM_THREADS=1 $PY -m ml.verify_single_batch --model-dir artifacts/handoff/$PKG --csv $R/submission_R3.csv --output $R/checks/single_vs_batch.json > $L/R3_single_batch.log 2>&1 || fail single-batch
grep -q '"status": "PASS"' $R/checks/single_vs_batch.json || fail single-batch-status
(cd $R && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum -c --quiet SHA256SUMS) || fail sha256sums2
echo "[$(date +%T)] single_vs_batch PASS; SHA256SUMS updated" >> $S

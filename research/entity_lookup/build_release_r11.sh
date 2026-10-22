#!/usr/bin/env bash
# R1.1 = same frozen model as R1, hardened inference code. Full rebuild + checks, then research round 2.
# Survives agent-session loss (run under nohup); every step logs to work/logs/r11_*.log and stops on the first failure.
set -euo pipefail
REPO=/home/makhzaun/Документы/Gi/hakaton
RES=/home/makhzaun/Документы/Gi/hakaton-research/work
L=$RES/logs
cd "$REPO"
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=.venv/bin/python
echo "[$(date +%T)] start R1.1" > $L/r11_status.log
$PY -m unittest discover -s tests > $L/r11_tests.log 2>&1
echo "[$(date +%T)] tests ok" >> $L/r11_status.log
$PY -m ml.entity_delivery --reference $RES/reference_te15lkbp_bag3_notxt/test_scores.npz > $L/r11_delivery.log 2>&1
DEL=$(tail -1 $L/r11_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["package"])')
RUN=$(basename "$DEL")
echo "[$(date +%T)] delivery $DEL" >> $L/r11_status.log
for f in model_0.txt model_1.txt model_2.txt te_tables.npz entity_tables.npz nb_weights.npy control_vectors.json; do
  cmp -s "$DEL/$f" artifacts/delivery/20261021T195245Z-fcf23067/$f || { echo "[$(date +%T)] FAIL payload differs: $f" >> $L/r11_status.log; exit 1; }
done
echo "[$(date +%T)] model payload identical to R1" >> $L/r11_status.log
$PY -m ml.make_submission_v2 --model-dir "$DEL" --reference-csv $RES/reference_te15lkbp_bag3_notxt/submission_te15lkbp_bag3_notxt_reference.csv > $L/r11_submission.log 2>&1
SUB=$(tail -1 $L/r11_submission.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["csv"])')
SHA=$(sha256sum "$SUB" | cut -d' ' -f1)
[ "$SHA" = c59aa747f37bcd126166a9f03b00ec75805d47550937c0359f7f49be00744f7d ] || { echo "[$(date +%T)] FAIL submission sha $SHA" >> $L/r11_status.log; exit 1; }
echo "[$(date +%T)] submission $SUB identical to R1 ($SHA)" >> $L/r11_status.log
PKG=entity-te15lkbp-nb-$RUN-pkg2
$PY -m ml.package_delivery --model-dir "$DEL" --wheelhouse artifacts/offline-wheels --output-dir artifacts/handoff/$PKG > $L/r11_package.log 2>&1
(cd artifacts/handoff && tar -czf $PKG.tar.gz $PKG && sha256sum $PKG.tar.gz > $PKG.tar.gz.sha256)
ASHA=$(cut -d' ' -f1 artifacts/handoff/$PKG.tar.gz.sha256)
OMP_NUM_THREADS=1 $PY -m ml.verify_delivery --archive artifacts/handoff/$PKG.tar.gz --sha256 $ASHA > $L/r11_verify_delivery.log 2>&1
grep -q '"status": "PASS"' $L/r11_verify_delivery.log || { echo "[$(date +%T)] FAIL offline verification" >> $L/r11_status.log; exit 1; }
EVID=$(tail -1 $L/r11_verify_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["evidence"])')
echo "[$(date +%T)] offline verification PASS $EVID archive $ASHA" >> $L/r11_status.log
R=artifacts/release/R1.1-entity-$RUN
mkdir -p $R/checks $R/fallback-rows65
cp -a "$SUB" $R/submission3.csv; cp -a "$(dirname "$SUB")/submission.json" $R/submission3.json
cp -a artifacts/handoff/$PKG.tar.gz artifacts/handoff/$PKG.tar.gz.sha256 $R/
cp -a "$EVID" $R/offline-verification
cp -a artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz.sha256 $R/fallback-rows65/
cp -a artifacts/submissions/20261021T125017Z-33955f0a/submission.csv $R/fallback-rows65/submission1.csv
OMP_NUM_THREADS=1 $PY -m ml.verify_single_batch --model-dir artifacts/handoff/$PKG --csv $R/submission3.csv --output $R/checks/single_vs_batch.json > $L/r11_single_batch.log 2>&1
grep -q '"status": "PASS"' $R/checks/single_vs_batch.json || { echo "[$(date +%T)] FAIL single-vs-batch" >> $L/r11_status.log; exit 1; }
cp -a $L/r11_tests.log $L/r11_delivery.log $L/r11_submission.log $R/checks/
(cd $R && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum -c --quiet SHA256SUMS)
echo "[$(date +%T)] R1.1 READY $R" >> $L/r11_status.log
echo "R11_DONE $R" >> $L/r11_status.log
# ---- research round 2 (pre-registered in hakaton-research/EXPERIMENTS.md)
S2=$RES/scripts/rowcv2.py; S3=$RES/scripts/rowcv3.py
BAG=3 FIXED=214 NOTXT=1 $PY $S2 te15lkbp 20261027 > $L/round2_base27.log 2>&1
$PY $RES/scripts/nb_rowcv.py 20261027 > $L/round2_nb27.log 2>&1
for sd in 20261027 20261024 20261026; do
  BAG=3 FIXED=214 NOTXT=1 BP16=1 MCS=100 $PY $S3 te15lkbp $sd > $L/round2_cand_$sd.log 2>&1
done
echo "[$(date +%T)] ROUND2_DONE" >> $L/r11_status.log

#!/usr/bin/env bash
# Generic release build: tests -> entity_delivery -> submission -> offline package -> clean-venv verify ->
# release dir -> exhaustive single-vs-batch -> SHA256SUMS. Stops at the first failure; logs in work/logs/<LABEL>_*.
# env: LABEL (required), BAND_WIDTHS (default 32), MCS (default 0 = preset), REFERENCE (npz, optional),
#      REF_CSV (optional), EXPECT_SHA (optional expected submission sha256)
set -euo pipefail
: "${LABEL:?LABEL required}"
BAND_WIDTHS=${BAND_WIDTHS:-32}; MCS=${MCS:-0}
REPO=/home/makhzaun/Документы/Gi/hakaton
L=/home/makhzaun/Документы/Gi/hakaton-research/work/logs
S=$L/${LABEL}_status.log
cd "$REPO"
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=.venv/bin/python
fail() { echo "[$(date +%T)] FAIL $*" >> $S; exit 1; }
echo "[$(date +%T)] start $LABEL widths=$BAND_WIDTHS mcs=$MCS" > $S
$PY -m unittest discover -s tests > $L/${LABEL}_tests.log 2>&1 || fail tests
echo "[$(date +%T)] tests ok" >> $S
REFARG=(); [ -n "${REFERENCE:-}" ] && REFARG=(--reference "$REFERENCE")
VARARG=(); [ "$BAND_WIDTHS" != 32 ] && VARARG+=(--band-widths "$BAND_WIDTHS"); [ "$MCS" != 0 ] && VARARG+=(--min-child-samples "$MCS")
$PY -m ml.entity_delivery "${VARARG[@]}" "${REFARG[@]}" > $L/${LABEL}_delivery.log 2>&1 || fail delivery
DEL=$(tail -1 $L/${LABEL}_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["package"])')
RUN=$(basename "$DEL"); VERSION=$(tail -1 $L/${LABEL}_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["model_version"])')
echo "[$(date +%T)] delivery $DEL $(tail -1 $L/${LABEL}_delivery.log | $PY -c 'import sys, json; r=json.loads(sys.stdin.read()); print({k: r[k] for k in r if k.startswith("max_abs")})')" >> $S
REFCSV=(); [ -n "${REF_CSV:-}" ] && REFCSV=(--reference-csv "$REF_CSV")
$PY -m ml.make_submission_v2 --model-dir "$DEL" "${REFCSV[@]}" > $L/${LABEL}_submission.log 2>&1 || fail submission
SUB=$(tail -1 $L/${LABEL}_submission.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["csv"])')
SHA=$(sha256sum "$SUB" | cut -d' ' -f1)
if [ -n "${EXPECT_SHA:-}" ] && [ "$SHA" != "$EXPECT_SHA" ]; then fail "submission sha $SHA != $EXPECT_SHA"; fi
echo "[$(date +%T)] submission $SUB $SHA" >> $S
PKG=$VERSION-pkg2
$PY -m ml.package_delivery --model-dir "$DEL" --wheelhouse artifacts/offline-wheels --output-dir artifacts/handoff/$PKG > $L/${LABEL}_package.log 2>&1 || fail package
(cd artifacts/handoff && tar -czf $PKG.tar.gz $PKG && sha256sum $PKG.tar.gz > $PKG.tar.gz.sha256)
ASHA=$(cut -d' ' -f1 artifacts/handoff/$PKG.tar.gz.sha256)
OMP_NUM_THREADS=1 $PY -m ml.verify_delivery --archive artifacts/handoff/$PKG.tar.gz --sha256 $ASHA > $L/${LABEL}_verify_delivery.log 2>&1 || fail offline-verify
grep -q '"status": "PASS"' $L/${LABEL}_verify_delivery.log || fail offline-verify-status
EVID=$(tail -1 $L/${LABEL}_verify_delivery.log | $PY -c 'import sys, json; print(json.loads(sys.stdin.read())["evidence"])')
echo "[$(date +%T)] offline verification PASS $EVID archive $ASHA" >> $S
R=artifacts/release/$LABEL-$RUN
mkdir -p $R/checks $R/fallback-rows65
cp -a "$SUB" $R/submission3.csv; cp -a "$(dirname "$SUB")/submission.json" $R/submission3.json
cp -a artifacts/handoff/$PKG.tar.gz artifacts/handoff/$PKG.tar.gz.sha256 $R/
cp -a "$EVID" $R/offline-verification
cp -a artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz.sha256 $R/fallback-rows65/
cp -a artifacts/submissions/20261021T125017Z-33955f0a/submission.csv $R/fallback-rows65/submission1.csv
OMP_NUM_THREADS=1 $PY -m ml.verify_single_batch --model-dir artifacts/handoff/$PKG --csv $R/submission3.csv --output $R/checks/single_vs_batch.json > $L/${LABEL}_single_batch.log 2>&1 || fail single-batch
grep -q '"status": "PASS"' $R/checks/single_vs_batch.json || fail single-batch-status
cp -a $L/${LABEL}_tests.log $L/${LABEL}_delivery.log $L/${LABEL}_submission.log $R/checks/
(cd $R && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum -c --quiet SHA256SUMS) || fail sha256sums
echo "[$(date +%T)] READY $R" >> $S
echo "RELEASE_DONE $R" >> $S

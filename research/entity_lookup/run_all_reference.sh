#!/usr/bin/env bash
# Reproduce every research reference artefact from the repo caches (TRAIN npy cache, text CSR cache, TEST.csv).
# Run from anywhere; ~3 min, ~5 GB RAM peak. Does not modify the repository.
set -euo pipefail
REPO=/home/makhzaun/Документы/Gi/hakaton
S=/home/makhzaun/Документы/Gi/hakaton-research/work/scripts
cd "$REPO"
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=.venv/bin/python
$PY "$S/build_test_cache.py"                 # TEST.csv -> work/cache/test_features.npy, ids, row hashes
$PY "$S/te15nb_reference.py"                 # work/reference_te15nb/ (TE15 tables, NB weights, LGB, scores)
$PY "$S/final_reference.py" te15lkbp 214     # work/reference_te15lkbp/ (needs reference_te15nb/nb_weights.npy)
$PY "$S/verify_entity_lookup.py"             # portable LK+BP tables + exact equality with the reference

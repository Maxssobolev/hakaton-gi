"""R3 package = R2 package (same boosters/tables/NB) + negative_partner_rule in metadata + current ml/inference.py.
Usage (repo root): make_r3_package.py R2_PKG_DIR OUT_DIR RUN_ID"""
import hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
src, out, run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
shutil.copytree(src, out)
for p in out.rglob('*'):
    if p.is_file(): p.chmod(0o644)
shutil.copy2('ml/inference.py', out / 'ml' / 'inference.py')
meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
meta['base_model_version'] = meta['model_version']
meta['model_version'] = 'entity-te15lkbp16-mcs100-nb-negbp-' + run_id
meta['run_id'] = run_id; meta['created_utc'] = datetime.now(timezone.utc).isoformat()
meta['negative_partner_rule'] = {'width': 32, 'min_shared_bands': 5, 'factor': 0.001,
    'definition': 'score*factor when bp_n5>0 and bp_pos5==0 (all close TRAIN band partners negative); applied after the NB blend; ranks these rows below all others; scores are not probabilities for them'}
meta['negative_partner_rule_cv'] = {'protocol': 'pre-registered 22.10 07:04 MSK; R2 final OOF (+NB) on identical row-level folds; rule features from fit-part labels only',
    'delta_vs_R2': {'20261027': [0.00411, '5/5'], '20261028': [0.00389, '5/5'], '20261029': [0.00409, '5/5'],
                    '20261024': [0.00314, '4/5'], '20261026': [0.00272, '5/5']},
    'note': 'seeds 24/26 were used to find the rule (diagnostics); 27/28/29 are its clean confirmation'}
(out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
for name in ('metadata.json', 'ml/inference.py'):
    b = (out / name).read_bytes(); man['files'][name] = dict(man['files'][name], sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))
(out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({'out': str(out), 'model_version': meta['model_version']}))

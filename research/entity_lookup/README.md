# Исследование: повторяющиеся сущности и entity-модель (2026-10-21)

Код экспериментов, обосновавших пакет формата 2 (`ml/features_v2.py`, `ml/entity_delivery.py`). Это provenance
для ревью, не production-код: скрипты ожидают локальный research-каталог
`/home/makhzaun/Документы/Gi/hakaton-research/` (OOF, эталоны, журнал `EXPERIMENTS.md` — вне Git, приватные)
и кеши репозитория `artifacts/ml-1`, `artifacts/text/20261021T125246Z-822ce8c9`. Запуск — из корня репозитория
с `PYTHONPATH=.`.

| файл | что делает |
| --- | --- |
| `cv_original/` | скрипты прошлой сессии (TE15, NB-текст на 9 групповых folds), пути исправлены с `/tmp` |
| `eval_te15nb.py` | пересчёт TE15+NB на 9 групповых folds из сохранённых OOF |
| `lookup_diag.py`, `lookup_union.py`, `band_diag.py` | диагностика совпадений по ключам/полосам (построчные folds) |
| `lookup_features.py`, `band_partner.py` | исследовательские версии LK/BP (эталон для `ml/features_v2.py`) |
| `rowcv.py`, `rowcv2.py`, `nb_rowcv.py`, `nbsvm_rowcv.py`, `eval_rowcv.py` | построчная CV вариантов и оценка блендов |
| `te15nb_reference.py`, `final_reference.py`, `entity_lookup.py`, `verify_entity_lookup.py` | full-TRAIN→TEST эталоны и их сверка |
| `holdout_check.py` | разведочная проверка на внешнем holdout после заморозки конфигурации |
| `run_all_reference.sh` | воспроизведение эталонов |

Результаты и решения — `docs/CHECKPOINT_SUBMISSION3.md`, `docs/ML_EXPERIMENTS.md`, `docs/ML_VALIDATION.md`.

Ночь 21→22.10: `rowcv3.py`/`band_partner2.py` (W=16 полосы), `build_release_r11.sh`, `build_release.sh` (сборка релиза
с полным циклом проверок). Отклонённая реализация варианта W=16+MCS100 лежит вне Git:
`/home/makhzaun/Документы/Gi/hakaton-research/work/r2_code/`.

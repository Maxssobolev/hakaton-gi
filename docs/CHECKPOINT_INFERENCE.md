# Checkpoint 3 — переносимый табличный inference

Дата 2026-10-21. Табличный fallback готов заранее, до окончания text experiment. Final selection фиксируется после текста; эта версия уже пригодна для интеграции backend по [ML_HANDOFF](ML_HANDOFF.md).

## Артефакты

- Model version `rows65-tabular-20261021T124857Z-440cc2cd`, 191 дерево, full TRAIN; float32 без импутации/масштабирования.
- Готовый каталог: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1/`.
- Архив: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz`, **56707268 bytes**.
- Archive SHA-256: `fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b`.
- Manifest SHA-256: `ee2954cbea04ebce12acc854b03ee7fea5e070cc3454351562b70052b05e539f`.
- Submission 1 и его исходный model package сохранены; [checkpoint 1](CHECKPOINT_SUBMISSION1.md). Прогнозы финального табличного пакета дополнительно сверяются с CSV.

Состав: `model.txt`, `schema.json`, `metadata.json`, `manifest.json`, `ml/__init__.py`, `ml/inference.py`, `requirements-inference.txt`, `control_vectors.json` (три синтетических вектора), `smoke_test.py`, `ML_HANDOFF.md`; wheelhouse из четырёх wheels (pip, numpy, scipy, lightgbm). Полные SHA-256 всех payload-файлов в manifest; данные людей и экспериментальные артефакты отсутствуют.

## Реальные проверки

- **22 synthetic unit tests PASS**, в том числе повреждение модели, нарушенный порядок schema, лишние/пропущенные features, типы/NaN/overflow, batch/single, ошибки submission, stateless text и validation-label invariance.
- Архив распакован в отдельный `/tmp` каталог; создан новый Python 3.8.20 venv. Установлены только numpy 1.24.4, scipy 1.10.1, lightgbm 4.5.0, через `--no-index` из wheelhouse; pip offline обновлён до 25.0.1 для manylinux_2_28 tag. CatBoost/pandas/sklearn не устанавливались.
- `python -I -B` smoke — **PASS**: load/reload, single=batch, порядок ключей, finite score [0,1], отсутствующий текст, ошибка схемы, empty batch. Все synthetic expected scores совпали с atol=1e-12.
- На smoke действует audit hook, запрещающий open любого файла исходного репозитория или TRAIN/TEST/text и `socket.connect`. Копии данных в /tmp не создавались. Проверка прошла с этими запретами.
- Smoke вместе со стартом процесса: **1.69 s**, peak RSS **66176 KiB (64.625 MiB)**; это измерение одного synthetic smoke, не нагрузочный SLA.
- Локальные evidence: `artifacts/handoff/verification-pkg1/{receipt.json,smoke.json,resources.txt,installed.txt,bootstrap.txt,install.txt}`.

Требования: CPython 3.8 / Linux x86_64, glibc >=2.28, системная libgomp.so.1. Начальный бюджет 1 CPU/512 MiB RAM; другие Python/ОС/архитектуры требуют совместимых wheels и повторного smoke. Backend сообщил об отсутствии ограничений среды. Калибровка и временная доступность анонимных признаков не подтверждены. Внешнее ChatGPT review по опубликованному Git-снимку ещё не выполнено; дополнительное агентное критическое ревью проводится отдельно, без подмены внешнего.

Дополнительное read-only ревью финального архива: blocker нет; все payload hashes/размеры совпали с tar, все четыре wheels сверены с официальными PyPI digests. Независимый повторный validator по **финальному** pkg1 manifest `ee2954cbea04ebce12acc854b03ee7fea5e070cc3454351562b70052b05e539f` подтвердил тот же CSV SHA и max score error 0.0. Исходный submission.json не переписывается и сохраняет ранний manifest; это новая проверка совместимости финального runtime с тем же обученным booster.

Воспроизводимая усиленная проверка с сохранением исходника audit hook:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ml.verify_delivery --archive artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz --sha256 fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b
```

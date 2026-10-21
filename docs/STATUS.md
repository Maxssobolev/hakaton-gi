# Статус проекта

**2026-10-21: ML HANDOFF READY.** Финальный кандидат зафиксирован досрочно: `lightgbm_rows65` без текста. Дедлайн ML — **22 октября 10:00 МСК**, предельный freeze — **09:00 МСК**. Новые исследования и LightGBM tuning не запускаются. Основание и приёмка — [ML-DELIVERY](exec-plans/active/ML-DELIVERY.md). Backend/frontend и перенос самого приложения — область принимающих команд.

## Готовая поставка

- Версия `rows65-tabular-20261021T124857Z-440cc2cd`. Full TRAIN refit 247972 строк, 191 дерево — медиана девяти prior stopping iterations, без нового tuning или holdout-подбора. Python API load/predict/predict_batch реализован, strict 1367 float32 features, без скрытого fit и исходных данных.
- Каталог: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1/`. Архив: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz`, 56707268 bytes. SHA-256 **`fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b`**. Все payload hashes в manifest. Состав, чистая установка и доказательства — [checkpoint inference](CHECKPOINT_INFERENCE.md), backend contract — [ML_HANDOFF](ML_HANDOFF.md).
- Проверенная среда: Python3.8.20/Linux x86_64, glibc>=2.28/libgomp; numpy1.24.4/scipy1.10.1/lightgbm4.5.0. Wheels и offline pip bootstrap включены.
- 22 unit tests PASS; fresh offline venv и synthetic smoke PASS, single=batch/reload. Проверка запрещает доступ к исходному репозиторию, dataset paths и сети; negative selfchecks guard PASS. Первый smoke 1.69 s/64.625 MiB RSS. Последний усиленный receipt: `artifacts/handoff-verifications/20261021T131130Z-741a4aa0/`.
- Submission 1: `artifacts/submissions/20261021T125017Z-33955f0a/submission.csv`. SHA-256 **`28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057`**. 106274 unique index, exact TEST order, finite [0,1], max prediction error **0.0**, в том числе по финальному pkg1. Исходные model package и provenance сохранены, не перезаписывались. [Checkpoint submission](CHECKPOINT_SUBMISSION1.md).

## Текст: исследован и отклонён для релиза

Полный parse всех274842 TSV records валиден: частоты1…11583, 396 пустых, duplicate/orphan/malformed не обнаружено. CSR кеш и все модели остаются локально. На тех же6 folds repeats1/2: A tabular AUC0.639551/AP0.024494; B text SGD AUC0.549281/AP0.015879; C fixed50/50 OOF blend AUC0.594379/AP0.018083. C уступил на6/6; правило включения не выполнено. В поставку не входят text model/CSR/TSV; Submission2 не создаётся. Полные foldwise данные и ограничения — [checkpoint text](CHECKPOINT_TEXT.md). Это отрицательный результат данного подхода, а не доказательство бесполезности любых текстовых моделей.

## Ревью и оставшиеся действия

Дополнительное read-only агентное ревью критичных кода/артефактов/OOF/метрик завершено: PASS, новых blocker/important нет. Воспроизведены метрики, partition development без holdout, exact blend/fallback, SHA всех cache arrays/моделей/OOF и состава архива. Это не внешнее ChatGPT review по опубликованному Git-снимку; его вердикт и формальная приёмка владельца ещё не получены. Исторический ML-4 tuning закрыт как исследование, но внешнее ML-4 review не объявляется проведённым.

All Cups форма доступна, однако upload-инструмент отверг доступ к локальной рабочей папке. Владелец выбрал ручную отправку готового CSV. Автоматическая отправка не состоялась, публичной оценки нет. Требуется ручная отправка/выбор решения на платформе и получение score. Фактическая установка backend во внутреннем контуре ещё не проверена; пакет готов для передачи по выбранному командой каналу.

Ограничения модели: анонимные признаки, неизвестное время их формирования и горизонт события; score не откалиброван. Baseline holdout ранее просмотрен; независимой финальной оценки нет. Код, агрегаты и checkpoints опубликованы отдельно от приватных данных/моделей/CSV.

# Статус проекта

**2026-10-22: решение владельца — R2 утверждён основным кандидатом Submission 3; R1.2 — fallback и возможный
Submission 4 для сравнения; rows65 — fallback.** R2 не является строго доказанным улучшением: исходный
предрегистрированный gate раунда 2 его отклонил, принятие в раунде 3 — адаптивное.
TRAIN/TEST оказались случайным построчным split с повторными снимками сущностей; групповая CV этот сигнал скрывала.
Модель R2 `entity-te15lkbp16-mcs100-nb-20261021T222359Z-22f845d4`: TE15 + совпадения сущностей + band-партнёры
(полосы 32 и 16 столбцов) → LightGBM ×3 (214 деревьев, min_child_samples=100) + NB-текст. Построчная CV R2:
0.6904–0.6984 по пяти seed (R1.2: 0.6868–0.6959; rows65 на seed 20261026: 0.6557). R2 к R1.2: +0.0027 в среднем,
отбор адаптивный — см. [REVIEW_PACKAGE_ENTITY](REVIEW_PACKAGE_ENTITY.md). Разведочный holdout (только R1):
0.6593 против rows65 0.6342. Подробно — [checkpoint 3](CHECKPOINT_SUBMISSION3.md), протокол —
[ML_VALIDATION](ML_VALIDATION.md), контракт — [ML_HANDOFF](ML_HANDOFF.md). Решения приняты Claude по делегированию
владельца. Код и документы закоммичены в ветку `ml/entity-r2-submission3`; CSV в All Cups загружает владелец
(отправка не считается выполненной без подтверждения платформы).

## Рекомендованный релиз R2

- `artifacts/release/R2-20261021T222359Z-22f845d4/`:
  - `submission3.csv`, SHA-256 **`260bbb892ab86ad4bd872b38a531a0a2a1a4db2622c439d629c2fd48042d9668`**;
  - архив `entity-te15lkbp16-mcs100-nb-20261021T222359Z-22f845d4-pkg2.tar.gz` (207418014 bytes), SHA-256
    **`185a525339d8875b2eec3cc233a9c99f86b77c7d15da40178c26721770ad745d`**;
  - fallback rows65, `SHA256SUMS`.
- Пакет = независимый эталон (2.2e-16); чистый venv PASS (6.4 s); single-row = batch на всём TEST: 0.0.
  Один экземпляр: 3.4 s загрузка, 427 MiB RSS, ~11 ms на запись. Codex-ревью PASS. 40 unit tests PASS.

## Альтернатива: релиз R1.2 (формат 2, без полос 16)

- Каталог релиза `artifacts/release/R1.2-20261021T212050Z-7b1d4c1f/` (`SHA256SUMS`, fallback rows65 внутри).
- Офлайн-пакет `entity-te15lkbp-nb-20261021T212050Z-7b1d4c1f-pkg2.tar.gz`, 136068913 bytes, SHA-256
  **`b22e94e007374123095157a4adbf9c9362dc2c34b182b7a3c390a336ce4765f3`**; чистая установка PASS (4.5 s, 453 MiB RSS),
  квитанция `artifacts/handoff-verifications/20261021T212843Z-0c1aa20f/`.
- Submission 3 `submission3.csv`, SHA-256 **`c59aa747f37bcd126166a9f03b00ec75805d47550937c0359f7f49be00744f7d`**
  (бит-в-бит одинаков во всех сборках R1/R1.1/R1.2).
- Single-row = batch на всех 106274 строках TEST, словарный API, батчи 1/7/256: max |Δ| 0.0 (`checks/single_vs_batch.json`).
- Тот же API (`BankruptcyPredictor.load/predict/predict_batch`); `text_counts` используется; без текста score точно
  табличный. `entity_tables.npz`/`te_tables.npz` — приватные производные меток TRAIN. 38 unit tests PASS.
- Ревью Codex CLI (2 прохода): MAJOR исправлены/обоснованы — [REVIEW_PACKAGE_ENTITY](REVIEW_PACKAGE_ENTITY.md).
- Утренняя сводка — [MORNING_HANDOFF_20261022.md](../MORNING_HANDOFF_20261022.md).

## Прежняя поставка (fallback, формат 1)

- Версия `rows65-tabular-20261021T124857Z-440cc2cd`. Full TRAIN refit 247972 строк, 191 дерево — медиана девяти prior stopping iterations, без нового tuning или holdout-подбора. Python API load/predict/predict_batch реализован, strict 1367 float32 features, без скрытого fit и исходных данных.
- Каталог: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1/`. Архив: `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz`, 56707268 bytes. SHA-256 **`fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b`**. Все payload hashes в manifest. Состав, чистая установка и доказательства — [checkpoint inference](CHECKPOINT_INFERENCE.md), backend contract — [ML_HANDOFF](ML_HANDOFF.md).
- Проверенная среда: Python3.8.20/Linux x86_64, glibc>=2.28/libgomp; numpy1.24.4/scipy1.10.1/lightgbm4.5.0. Wheels и offline pip bootstrap включены.
- 22 unit tests PASS; fresh offline venv и synthetic smoke PASS, single=batch/reload. Проверка запрещает доступ к исходному репозиторию, dataset paths и сети; negative selfchecks guard PASS. Первый smoke 1.69 s/64.625 MiB RSS. Последний усиленный receipt: `artifacts/handoff-verifications/20261021T131130Z-741a4aa0/`.
- Submission 1: `artifacts/submissions/20261021T125017Z-33955f0a/submission.csv`. SHA-256 **`28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057`**. 106274 unique index, exact TEST order, finite [0,1], max prediction error **0.0**, в том числе по финальному pkg1. Исходные model package и provenance сохранены, не перезаписывались. [Checkpoint submission](CHECKPOINT_SUBMISSION1.md).

## Текст (SGD, 13:00–14:00): отклонён; NB-текст вошёл в формат 2

Полный parse всех274842 TSV records валиден: частоты1…11583, 396 пустых, duplicate/orphan/malformed не обнаружено. CSR кеш и все модели остаются локально. На тех же6 folds repeats1/2: A tabular AUC0.639551/AP0.024494; B text SGD AUC0.549281/AP0.015879; C fixed50/50 OOF blend AUC0.594379/AP0.018083. C уступил на6/6; правило включения не выполнено. В поставку не входят text model/CSR/TSV; Submission2 не создаётся. Полные foldwise данные и ограничения — [checkpoint text](CHECKPOINT_TEXT.md). Это отрицательный результат данного подхода, а не доказательство бесполезности любых текстовых моделей.

Quality sprint дополнительно проверил только готовые OOF: 5% текста AUC0.634069,
10%0.627653, оба проиграли6/6. Parse/fit не повторялись; текст остаётся вне поставки.

## Ревью и оставшиеся действия

Для прежнего pkg1 дополнительное read-only агентное ревью критичных кода/артефактов/OOF/метрик завершилось PASS. Для нового quality sprint выполнены авторские проверки; независимое внешнее ChatGPT review и новая приёмка владельца не проводились. Все12 новых fold моделей отдельно проверены по train/validation AUC/AP/gap, nested splits, hashes и OOF. Исторический ML-4 tuning закрыт как исследование, но внешнее ML-4 review не объявляется проведённым.

Владелец сообщил о принятой ручной отправке Submission 1 и public ROC-AUC 0.64459. Автоматическая отправка ранее не состоялась из-за ограничения upload-инструмента. Выбранный Submission 1 не менять до проверки и одобрения Submission 2. Фактическая установка backend во внутреннем контуре ещё не проверена; fallback пакет готов для передачи по выбранному командой каналу.

Ограничения модели: анонимные признаки, неизвестное время их формирования и горизонт события; score не откалиброван. Baseline holdout ранее просмотрен; независимой финальной оценки нет. Код, агрегаты и checkpoints опубликованы отдельно от приватных данных/моделей/CSV.

# Checkpoint 1 — табличная поставка и Submission 1

Дата 2026-10-21. Scope: `ml/inference.py`, `ml/delivery.py`, `ml/make_submission.py`, `ml/package_delivery.py`, synthetic tests; контракт [ML_HANDOFF](ML_HANDOFF.md), срочный [план](exec-plans/active/ML-DELIVERY.md).

## Результат

Full TRAIN rows65, 191 дерево — заранее фиксированная медиана девяти prior stopping iterations, без нового tuning. Версия `rows65-tabular-20261021T124857Z-440cc2cd`; исходный локальный пакет `artifacts/delivery/20261021T124857Z-440cc2cd/` сохраняется неизменным. Internal AUC на шести дополнительных folds 0.639551 ± 0.008700, AP 0.024494; это прежние CV результаты, не новая final independent evaluation.

Submission `artifacts/submissions/20261021T125017Z-33955f0a/submission.csv`, SHA-256 `28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057`. Validator: PASS, 106274 строк/unique index, точный порядок TEST, comma/header, finite [0,1], нет pandas index, max abs discrepancy с загруженной моделью **0.0**. Manifest исходного model package SHA-256 `345c09afc54e3234658db53a7a47d38ac7442bb9f9a439c41f51596b2522b37f`; JSON provenance рядом с CSV.

All Cups: https://cups.online/ru/workareas/Code_of_the_Time/1924/3664 . Авторизованная форма доступна, до загрузки 10 попыток, решений нет. `upload_file` отверг локальный путь как вне configured workspace roots браузерного инструмента. Отправка **не состоялась**; leaderboard score отсутствует. Обход доступа не выполнялся.

## Review

Дополнительный read-only агент проверил ML-4 границы, inference, delivery и submission: блокеров нет. Исправлен TypeError скалярного text_counts, добавлены negative cases. Проверены >2**53 index, неверные header/order/count/score и повреждение модели. Это агентное ревью не заменяет внешнюю сессию ChatGPT по Git-снимку. Финальные offline smoke и текстовый эксперимент ещё выполняются.

Для внешнего reviewer: проверить frozen median selection/full TRAIN fit, schema/type semantics и точность index, provenance/неперезапись, отсутствие source rows/model artifacts в публичном Git. Git checkpoint сохраняет только код, документацию и агрегаты; все приватные артефакты остаются локально.

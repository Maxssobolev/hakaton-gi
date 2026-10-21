# Статус проекта

Дата: 2026-10-21. Прямое срочное задание владельца разрешило [ML-DELIVERY](exec-plans/active/ML-DELIVERY.md): freeze **22 октября 09:00 МСК**, ML handoff **до 10:00 МСК**. ML Foundation принят ранее; ML-4 tuning прекращён с базовым кандидатом `lightgbm_rows65`. Историческое внешнее ML-4 ревью ещё не проводилось; новое разрешение позволяет поставку без ожидания старых ворот.

## Рабочий табличный компонент

- Полный TRAIN refit: 247972 строки, 191 дерево (медиана девяти сохранённых stopping iterations), без нового tuning/holdout-подбора. Версия `rows65-tabular-20261021T124857Z-440cc2cd`. Исходный immutable пакет `artifacts/delivery/20261021T124857Z-440cc2cd/`.
- Реализованы строгая схема 1367 признаков, load без fit, single/batch, reserved text input и точный tabular fallback. Контракт для backend: [ML_HANDOFF](ML_HANDOFF.md).
- Submission 1: `artifacts/submissions/20261021T125017Z-33955f0a/submission.csv`; SHA-256 `28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057`. 106274 unique index, точный порядок TEST; повторные scores загруженного пакета совпали с max error 0.0.
- All Cups показывает 10 оставшихся попыток и форму загрузки, но upload_file отверг локальный путь как находящийся вне configured workspace roots браузерного инструмента. CSV **не отправлен**, публичного score **нет**. Это ограничение инструмента, а не отрицательная оценка платформы.
- Final offline wheelhouse и чистое окружение готовятся отдельно; исходный пакет и submission сохраняются.

## Исследование и проверки

Полный streaming текстовый аудит/hash CSR запущен. Fixed SGD baseline и 50/50 OOF blend будут сравниваться с rows65 на тех же 6 folds repeats 1/2. Нет дальнейшего LightGBM tuning. Метрики текста пока не получены.

Дополнительное read-only агентное критическое ревью кода проведено параллельно: исправлены тип batch text_counts, запрет невалидного text cache и проверка порядка index; блокеров для запуска нет. Оно не заменяет внешнее ChatGPT review по опубликованному снимку.

## Ограничения

Нет времён события/формирования данных и словаря 1367 признаков; score не откалиброван. Baseline holdout ранее просмотрен, новой независимой финальной оценки нет. Внутренняя repeated CV описывает качество кандидата (AUC 0.639551 ± 0.008700 на 6 дополнительных folds), а не full-TRAIN training quality. Код и агрегаты можно публиковать по новому разрешению; исходные данные, модель, CSR и CSV остаются локальными.

## Проверенный offline fallback

Архив `artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz` (56.7 MB), SHA-256 `fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b`. Чистый offline venv, три synthetic controls и запрет доступа к репозиторию/исходным данным/сети — PASS; smoke 1.69 s, RSS 64.625 MiB. Только numpy/scipy/lightgbm; 22 unit tests PASS. Состав и доказательства — [checkpoint inference](CHECKPOINT_INFERENCE.md). Текстовый эксперимент ещё не завершён; готовый tabular пакет можно интегрировать уже сейчас.

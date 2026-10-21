# ML QUALITY SPRINT RESULT — 2026-10-21

Автор: Codex. Scope: P0–P5 прямого задания владельца, в рамках ML-DELIVERY.
Fallback rows65 и Submission 1 сохранены. Public ROC-AUC **0.64459** сообщён
владельцем; эта оценка не использовалась для выбора признаков или параметров.

## Протокол и диагностика P0

Только development: 197960 строк, 2674 positives. Прежние групповые folds repeat0,
отдельный nested stop split и refit на полном train fold. Validation не участвует
в stopping; source/cache hashes и digest всех позиций проверяются. Новый holdout
не создавался, внешний holdout не оценивался. Полный протокол зафиксирован до
новых результатов в [ML_VALIDATION](ML_VALIDATION.md).

- По среднему нормированному gain девяти rows65 моделей лидируют `feature_41`
  (2.68%), `feature_1094` (2.41%), `feature_1122` (2.21%), `feature_363` (2.02%),
  `feature_337` (1.90%). Это training importance, не causal effect и не
  подтверждение содержательного смысла анонимных столбцов.
- Сильного одномерного сигнала общей активности нет: AUC количества ненулевых
  0.472856, обратного направления/доли нулей 0.527144; количества -1 — 0.509796.
  Среднее число ненулевых: target0 362.242, target1 355.413. Это диагностические
  marginal associations на development, не CV-качество обученной модели.
- Ячеек <−1 на development нет. Доля -1: target0 2.1571%, target1 2.2191%.
  Raw -1/0 сохранены; нет оснований объявлять -1 пропуском или заменять его
  без отдельного эксперимента. Нефинитные значения запрещены кешем и API.
- Девять historical stopping iterations 73…301; evaluated=best+40, все до лимита
  1000. На stop лишь 176…185 positives; stop AUC в лучших точках 0.63094…0.70388.
  Train/validation gap 0.08757…0.18024 поддерживает гипотезу переобучения и
  нестабильной stopping-оценки. Ошибки, где validation управлял бы остановкой,
  не обнаружено. Patience40 не исключает более позднего улучшения после stop.
- Более сильная регуляризация ранее улучшила rows65 относительно leaves7;
  утверждение «всё объясняется слишком агрессивной регуляризацией» не
  подтверждено. Уменьшение gap само по себе не является целью выбора.

## Feature engineering P1

`activity-v1`: 13 per-row статистик без target или fit на данных. Nonzero count,
zero fraction, -1 count, positive sum/log1p sum/mean/max/std/log1p std, counts
выше 1/10/100/1000. Empty positive set → 0, float64 accumulation → явное
ограничение диапазоном float32; raw признаки неизменны. Пороги фиксированы до
метрик. Суммы чувствительны к разному масштабу анонимных столбцов, поэтому
дополнены log-статистиками и counts. Семантика столбцов не придумывалась.

LGB-activity проиграл 3/3 folds: mean AUC 0.640477, delta −0.003760.
Снижение gap 0.160471 → 0.140948 не улучшило validation AUC. Предзаданное
условие продолжения не выполнено: additional folds для него не запускались.
Совокупная доля gain агрегатов по folds — 1.5313% / 1.5495% / 2.2342%
(8 / 27 / 31 splits): они используются, но устойчивого выигрыша это не дало.
Колонки и feature sampling изменяются при добавлении агрегатов; это сравнение
двух полных pipelines, а не изолированная causal оценка каждого агрегата.

## Сохранённый текст P3

Только готовые OOF шести folds repeats1/2, exact tabular fallback при отсутствии
usable text. Хэши OOF и lookup/CSR indices/indptr проверены; missing mask
воспроизвёл прежний blend в точности. TSV не читался, текст не переобучался.

| Text weight | Mean CV AUC | Mean CV AP | Δ AUC к tabular | Победы | Δ mean repeats1 / 2 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0% | 0.639551 | 0.024494 | 0 | — | 0 / 0 |
| 5% | 0.634069 | 0.023810 | −0.005482 | 0/6 | −0.005291 / −0.005673 |
| 10% | 0.627653 | 0.022435 | −0.011898 | 0/6 | −0.011692 / −0.012104 |

Текст исключён из текущей поставки. Это отрицательный результат этих моделей
и смесей, а не доказательство бесполезности всех возможных текстовых подходов.
Проверка нескольких весов на той же CV не создаёт независимой final evaluation.

## XGBoost и выбор подтверждаемого кандидата P2

Одна заранее фиксированная конфигурация CPU hist: depth3, eta0.05,
min_child_weight10, alpha1/lambda20, rows0.65, columns0.8, max_bin128, максимум
1000 деревьев, patience40. Raw 1367 features, без activity aggregates.
Repeat0: XGBoost AUC0.642405, Δ−0.001832, победа1/3; fixed50/50 с rows65
дал AUC0.646311, Δ+0.002073, победы3/3. Выбран для confirmation именно этот
фиксированный blend. Новых конфигураций/весов после выбора не добавлялось;
второй вариант XGBoost с агрегатами не запускался, поскольку P1 проиграл3/3,
а проверка устойчивости уже выбранной смеси полезнее расширения поиска.

На первом дополнительном repeat blend дал AUC0.637379, Δ+0.001116, победы2/3.
XGBoost отдельно на третьем fold ухудшился на0.028609, blend на0.005631:
предварительный выигрыш не является доказанным устойчивым улучшением.
Gate: mean Δ дополнительных folds≥0.002, оба repeat means положительны,
минимум5/6 побед; правило не ослабляется после просмотра результатов.

Для repeat2 устранена лишняя ~1GiB приватная копия raw development: source
read-only memmap адресуется теми же development-only global positions. Значения,
порядок строк, folds, параметры, seeds и preprocessing не меняются.
Синтетический тест local copy vs global memmap получил побайтово одинаковые
модели и точно одинаковые train/validation predictions. Старые runner snapshots
сохранены, verifier проверяет их хэши и неизменность transform/split helpers.

## Общая таблица P4 и решение

Repeat0 — первоначальное сравнение, использованное для выбора кандидата:

| ID | Признаки | Mean AUC | Foldwise AUC | CV AP | Mean gap | Stop+refit, s | Peak RSS, MiB | Δ к rows65 | Inference |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| A0 | raw1367 | 0.644238 | 0.649306 / 0.643277 / 0.640130 | 0.025447 | 0.160471 | 100.93 | 3099.68 | +0.000000 | fallback готов |
| B0 | raw1367 +13 | 0.640477 | 0.639665 / 0.642337 / 0.639430 | 0.024758 | 0.140948 | 136.85 | 4175.10 | -0.003760 | fold reload PASS; пакет не выпускается |
| X0 | raw1367 | 0.642405 | 0.646160 / 0.643740 / 0.637316 | 0.025311 | 0.137510 | 402.65 | 4323.11 | -0.001832 | fold reload PASS; research |
| E0 | 0.5A0 +0.5X0 | 0.646311 | 0.650446 / 0.646520 / 0.641967 | 0.025838 | 0.157374 | reuse A0/X0; no fit | не измерен совместно | +0.002073 | OOF; итог confirmation FAIL |

Дополнительные шесть folds, в порядке repeat1 fold0/1/2, repeat2 fold0/1/2:

| ID | Признаки | Mean AUC | Foldwise AUC | CV AP | Mean gap | Stop+refit, s | Peak RSS, MiB | Δ к rows65 | Inference |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| A12 | raw1367 | 0.639551 | 0.627834 / 0.648981 / 0.631973 / 0.644388 / 0.636644 / 0.647485 | 0.024494 | 0.129014 | 183.06 | 3240.05 | +0.000000 | fallback готов |
| X12 | raw1367 | 0.634225 | 0.634117 / 0.652166 / 0.603363 / 0.631743 / 0.633640 / 0.650321 | 0.024048 | 0.122778 | 1007.95 | 3694.12 | -0.005326 | fold reload PASS; research |
| E12 | 0.5A12 +0.5X12 | 0.641047 | 0.632799 / 0.652997 / 0.626341 / 0.644396 / 0.638073 / 0.651676 | 0.024785 | 0.131486 | reuse A12/X12; no fit | не измерен совместно | +0.001496 | gate FAIL; не выпускается |

A — ранее сохранённый rows65; B — activity-v1; X — одна конфигурация
XGBoost; E — фиксированный blend. Stop+refit исключает подготовку/хэши кеша,
prediction, serialization и OOF arithmetic. RSS — измеренный накопительный пик
отдельного training процесса (для двух repeats показан максимум их пиков).
Для blend дополнительного fit нет: использованы те же A/X, совместный peak
pipeline не измерялся; это не 0 MiB. Совместный train gap пересчитан по
in-sample прогнозам обоих refit learners, validation — только OOF.

На дополнительных folds лучшая новая смесь: **AUC 0.641047030**,
baseline **0.639550904**, Δ **+0.001496126**.
Оба repeat mean положительны (+0.001116 / +0.001876), победы5/6, но одна
потеря −0.005631, а одна победа всего +0.000008. Средний прирост ниже
предзаданного порога +0.002. **Gate FAIL: сохранить rows65 и Submission1.**
Это небольшой положительный наблюдаемый эффект, а не доказанное существенное
улучшение; статистическая значимость не заявляется. Folds зависимы, repeats
используют один development, baseline ранее сравнивался на них. Ни новый
внешний holdout, ни leaderboard для выбора не использовались.

Реально обучены 3 LGB-activity fold refits и 9 XGBoost fold refits; перед каждым
был отдельный train-only stopping fit. Blend не обучает дополнительных моделей.
Повторного text fit нет. Старый A не переобучался. Все 12 новых fold моделей
сохранены и прошли отдельную проверку train/validation AUC/AP/gap/OOF.

## Поставка P5 и проверки

Submission2 **не создан**; нового full-TRAIN fit и нового release inference
package нет. Производственные `ml/inference.py`, `ml/make_submission.py`,
`ml/package_delivery.py` и runtime requirements совпадают с base checkpoint.
Кандидатный runtime был проверен на synthetic fixtures во время CV, но остался
только в ignored `artifacts/quality-sprint/prototype/`, поскольку gate провален.
Его готовность не объявляется готовностью нового ML package.

Проверенный CSV fallback:
`artifacts/submissions/20261021T125017Z-33955f0a/submission.csv`, SHA-256
`28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057`.
106274 строк, точный index/order TEST, finite continuous score [0,1],
max prediction discrepancy0.0 — ранее проверены по pkg1; CSV hash теперь
повторно подтверждён. Выбранный Submission1 не менялся.

Готовый fallback archive:
`artifacts/handoff/rows65-tabular-20261021T124857Z-440cc2cd-pkg1.tar.gz`, SHA-256
`fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b`.
Прежний clean offline smoke остаётся применимым, поскольку архив не изменился;
в этом sprint он не выдаётся за новый запуск. Полные прежние доказательства —
[CHECKPOINT_INFERENCE](CHECKPOINT_INFERENCE.md).

Рекомендация: **оставить текущий Submission1 и передавать прежний pkg1**.
Новая отправка на All Cups не выполнялась. Прежняя public0.64459 сообщена
владельцем; никаких предположений о public score смеси не делается.

## Воспроизведение и review

Ветка `ml-quality/activity-xgboost`, base `1232cf6`.
[GitHub compare](https://github.com/Maxssobolev/hakaton-gi/compare/1232cf6...ml-quality/activity-xgboost).
Внешнее ChatGPT review и новая приёмка владельцем не проводились.
Этот файл — компактный review package: протокол, диагностика, реальные метрики,
решение и ссылки на приватные доказательства. Исходные данные, модели, OOF,
CSV и архивы не публикуются. Независимый reviewer может проверить код и
агрегаты опубликованной ветки; доступ к локальным артефактам нужен для
повторного пересчёта метрик на приватных данных.

Код: `ml/activity_features.py`, `ml/quality_sprint.py`,
`ml/sprint_diagnostics.py`, `ml/sprint_blend.py`, `ml/sprint_selection.py`,
`ml/verify_sprint.py`, `tests/test_activity.py`;
исследовательская зависимость `xgboost-cpu==2.1.4` в `requirements-sprint.txt`.
XGBoost использует CPU histogram и явную регуляризацию согласно
[официальным параметрам](https://xgboost.readthedocs.io/en/stable/parameter.html).
Команды — [VERIFICATION](VERIFICATION.md). Существующий inference API и
runtime requirements не менялись.

Runs внутри `artifacts/quality-sprint/`:

- B0: `20261021T135545Z-b8634451`.
- X0/E0: `20261021T135922Z-276fb0f8`.
- X1/E1: `20261021T140758Z-36d5d6f0`.
- X2/E2: `20261021T142056Z-c8dc2f20`.

Приватные доказательства: `artifacts/quality-sprint/diagnostics.json`,
`text-blends.json`, `selection.json`, `summary.json`, `final-checks.json`,
`native-parameter-audit.json`, `fallback-before.json`, `unit-tests.log`, уникальные run
каталоги с models/OOF/snapshots/results/training history/verification.

Финальные проверки: **26 unit tests PASS**, compiler/diff checks PASS,
24 Markdown файла без сломанных относительных ссылок; native parameters всех
9 XGBoost моделей соответствуют preset. Повторно совпали SHA-256 всех трёх
исходных файлов, старого archive/CSV/delivery manifest. Production source и
runtime requirements не изменены. Ни исходные строки, ни приватные артефакты
в Git checkpoint не включены.

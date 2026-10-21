# Checkpoint 2 — результат проверки text_data.tsv

Дата 2026-10-21. **Текст в финальный пакет не включается.** Проверенный baseline и заранее фиксированный ансамбль уступили tabular на всех шести folds. Это отрицательный результат конкретного быстрого подхода, а не доказательство бесполезности текста при любых других моделях. Дальнейших экспериментов перед поставкой не запускаем.

## Протокол и provenance

Протокол записан до текстовых метрик в [ML-DELIVERY](exec-plans/active/ML-DELIVERY.md) и [ML_VALIDATION](ML_VALIDATION.md). Streaming parse реальных частот, log1p, FeatureHasher 2**18/alternate_sign=False, L2 и отдельный missing indicator; без learned vocabulary/IDF. До оценки исключены основы банкрот/несостоят/конкурсн/bankrupt/insolvenc. SGD log_loss, alpha=1e-4, average=True, seed42, 5 epochs, batch2048, shuffle только train. Веса не подбирались.

A — ранее сохранённый rows65, B — text SGD на train fold, C — fixed 0.5 A + 0.5 B по OOF validation; при отсутствии usable text C=A. Использованы те же сохранённые group folds repeats1/2, включая проверку digest/positions и source/cache/index integrity. A повторно загружен и воспроизводит прежние метрики с atol1e-12; внешнего holdout и leaderboard в отборе нет.

Run `20261021T132201Z-39481bde`, локальный каталог `artifacts/text-experiments/20261021T132201Z-39481bde/`. `results.json`, шесть joblib моделей и шесть NPZ OOF с SHA сохранены приватно. Parser/cache: `artifacts/text/20261021T125246Z-822ce8c9/`; полный аудит — [ML_DATA](ML_DATA.md). Код до результатов опубликован на commit `a59b8b8` (первые inference/submission checkpoint — `347d7a8`, offline checkpoint — `6830de5`).

## Результаты

| Repeat/fold | A AUC | A AP | B AUC | B AP | C AUC | C AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/0 | 0.627834 | 0.023194 | 0.540174 | 0.015169 | 0.583372 | 0.017072 |
| 1/1 | 0.648981 | 0.025234 | 0.560482 | 0.016351 | 0.608943 | 0.019346 |
| 1/2 | 0.631973 | 0.023428 | 0.545533 | 0.016058 | 0.582945 | 0.017607 |
| 2/0 | 0.644388 | 0.025578 | 0.545040 | 0.015906 | 0.593978 | 0.018706 |
| 2/1 | 0.636644 | 0.023802 | 0.530198 | 0.014878 | 0.578652 | 0.016805 |
| 2/2 | 0.647485 | 0.025729 | 0.574262 | 0.016911 | 0.618381 | 0.018962 |

| Вариант | Mean AUC ± SD | Mean AP |
| --- | ---: | ---: |
| A | 0.639551 ± 0.008700 | 0.024494 |
| B | 0.549281 ± 0.015676 | 0.015879 |
| C | 0.594379 ± 0.016047 | 0.018083 |

Средний C−A AUC **-0.045172**, выигрыш **0/6**. Оба repeats ухудшились; заранее заданное правило (оба repeats и минимум5/6 побед) не выполнено. SD ddof1 — описательный разброс зависимых CV folds, не стандартная ошибка независимых наблюдений.

Собственно обучение/оценка шести folds и TEST smoke заняли **68.37 s**, peak process RSS **4148.20 MiB**, включая memory-mapped CSR. Полный предварительный parse занял 1276.85 s. Все шесть text моделей перезагружены: OOF scores совпали с atol1e-12. На всех 106274 TEST строках last-fold text model выдаёт конечные [0,1]; это проверка transform/score, а не выпуск full-TRAIN текстовой модели.

## Лексический экран и ограничения

Предзаданный отдельный screen проверил 274842 текстовых записи; text SHA совпал с parser/cache. Оценка слов выполнялась только на development. Для основы банкрот presence AUC 0.506853: доля target0 4.164%, target1 5.535%; прочие заданные outcome/context stems — около0.5 (min 0.491855, max 0.506853). Нулевых частот в источнике нет. Результат не выявил почти прямого раскрытия target в этом ограниченном наборе слов; он не доказывает отсутствия других или временных утечек. Presence также смешивается с разницей текстового покрытия (77.746%/76.627%). Времён формирования текста и события нет.

Решение: поставить уже проверенный **rows65 tabular**, без text model/CSR/TSV и без Submission2. Первая отправка остаётся тем же CSV; владелец выбрал ручную отправку. Код и данные эксперимента сохраняются, не удаляются. Финальный пакет — [checkpoint inference](CHECKPOINT_INFERENCE.md). Внешнее ChatGPT review по опубликованному снимку не объявляется проведённым; дополнительный read-only агент проверяет критичные результаты.

Дополнительное read-only агентное ревью результата: **PASS**, blocker/important нет. Независимо сверены SHA/bytes cache arrays и шести моделей/OOF, saved/deterministic fold positions, полная partition197960 development rows в каждом repeat без holdout, formula C и точный missing fallback. Все AUC/AP/repeat means/SD/positives/usable counts воспроизведены с atol1e-12. Лексический screen SHA/агрегаты проверены; temporal caveat сохраняется. Решение release_text=False подтверждено.

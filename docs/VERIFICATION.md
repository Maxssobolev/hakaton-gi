# Проверка текущего этапа

Проверка честная и поэтапная. Нет фиктивных тестов и команд, всегда возвращающих успех. Текущие исполняемые команды описаны в README.md и активном плане.

Для контрольного пакета ML Foundation (ML-1–ML-3 дорожной карты) при наличии данных и зависимостей:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ml.audit --data-dir . --output artifacts/data-audit.json
.venv/bin/python -m ml.sample_eda --data-dir . --output artifacts/sample-eda.json
.venv/bin/python -m ml.train_baselines --data-dir . --output-dir artifacts/ml-1
.venv/bin/python -m ml.verify --artifact-dir artifacts/ml-1 --data-dir .
git diff --check
git status --short
```

Для ML-4 выполнять один тяжёлый запуск за раз; runner повторно проверяет хэш TRAIN и кеш, сохраняет внутренние folds и метрики после каждого завершённого fold:

```bash
.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.diagnose_ml4
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_baseline
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_depth3
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_l2_10
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_base
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_leaves7
.venv/bin/python -m py_compile ml/*.py tests/*.py
git diff --check
git status --short
```

Результаты обучения сверять с `artifacts/runs/<run_id>/results.json` и журналом; папки запусков не включать в Git. Эти команды не читают внешний holdout для выбора модели. Проверка полной проекции исходного CSV на кеш уже выполнена в ML Foundation и повторяется при изменении кеша/источника, а не после каждого preset.

Для выбранных завершённых запусков повторно загрузить модели и пересчитать validation ROC-AUC/AP: `.venv/bin/python -m ml.verify_ml4 --run-dir artifacts/runs/<run_id>`.

Синтетические unit tests быстрые и не требуют датасетов. Аудит проверяет реальные файлы, обучение должно вывести метрики и сохранить модели, verify проверяет повторную загрузку, схему, хэши кеша и полное потоковое соответствие кеша TRAIN.csv. При смене версии кеша его можно отдельно пересобрать вызовом `ml.cache.prepare_cache` без переобучения; после этого выполнить `ml.verify` и сравнить baseline-метрики с журналом. Большие проверки не запускаются параллельно. Если команда не выполнена, пишите «не запускалась» с причиной; если завершилась ошибкой, сообщайте ошибку. Не объявляйте приложение, CI или интеграционные проверки прошедшими до появления соответствующих компонентов.

Дополнительно при каждом изменении: проверить относительные Markdown-ссылки, отсутствие ссылок на универсальный harness как на обязательную внешнюю инструкцию, игнорирование датасетов, отсутствие секретов, полный diff новых и изменённых файлов. Для неотслеживаемых файлов git diff --check их не видит — инспектировать отдельно.

Перед публикацией review checkpoint отдельно проверить список staged файлов и staged diff, исключение `TRAIN.csv`, `TEST.csv`, `text_data.tsv`, локальных `artifacts/` и любых чувствительных материалов. Публичная ветка и PR создаются только после разрешения владельца. Результаты независимого ревью и решения владельца записывать в активный план; до их получения план остаётся активным.

Ограниченная вторая серия ML-4 расширяет тот же runner, не требует нового кеша:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_min500
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_l1l2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_rows65
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_depth2
```

Дополнительная CV: `--repeat 1` и `--repeat 2` для `lightgbm_leaves7` и единственного кандидата, выбранного по заранее записанному правилу из [ML_VALIDATION](ML_VALIDATION.md). Каждая команда создаёт новый run ID. Эти команды описывают воспроизведение, а не необходимость повторять уже завершённое обучение при каждом изменении документации.

Расширенный `ml.verify_ml4` использует seeds из конкретного run, проверяет все сохранённые позиции, хэши source/cache и runner, пересчитывает **train и validation ROC-AUC/AP и gap**. Для новых LightGBM runs дополнительно проверяет хэши моделей / training evidence, best iteration по stopping history, patience, число refit деревьев и соответствие native parameters запрошенным. Синтетический тест меняет только validation labels и проверяет неизменность выбранного числа деревьев, сохранённой модели и score; инварианты всех трёх repeats проверяются отдельно от реальных данных. Исторические runs без stopping history проверяются в пределах имеющихся артефактов.

## Срочная поставка 2026-10-21

Табличный refit не повторять при каждом изменении документации. Команды реального запуска:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.delivery
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m ml.make_submission --model-dir artifacts/delivery/20261021T124857Z-440cc2cd
.venv/bin/python -m ml.make_submission --model-dir artifacts/delivery/20261021T124857Z-440cc2cd --validate artifacts/submissions/20261021T125017Z-33955f0a/submission.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ml.text_features
# Затем после полного cache:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ml.text_experiment --text-cache artifacts/text/<run_id>
```

Portable runtime проверяется в отдельном venv, устанавливаемом через `--no-index` из wheelhouse. Из копии архива в отдельном каталоге запускается `python -I smoke_test.py`, без TRAIN/TEST/text и без пути к рабочему репозиторию. SHA-256 manifest сверяет весь payload; archive hash сохраняется отдельно. Финальные факты и review checkpoints — в STATUS и ML_EXPERIMENTS. Unit suite включает повреждения артефактов, reordered schema, invalid types/NaN/overflow, batch semantics, submission failures и stateless text transform.

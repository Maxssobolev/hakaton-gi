# Прогнозирование банкротства физических лиц

ML-подсистема для хакатона: исследование данных, обучение бинарного классификатора и подготовка переносимого Python-инференса. Backend и frontend разрабатывают другие команды. [Оригинальная постановка](docs/HACKATHON_BRIEF.txt) хранится локально в репозитории.

## Состояние

ML Foundation принят. **ML HANDOFF READY:** `lightgbm_rows65` без текста, public ROC-AUC0.64459 по сообщению владельца. Quality sprint завершён: activity aggregates не улучшили CV, fixed50/50 rows65/XGBoost дал лишь +0.001496 на дополнительных6 folds и не прошёл заранее заданный release gate. Сохраняются Submission1 и прежний offline pkg1; [полный результат](docs/QUALITY_SPRINT_RESULT.md). **Freeze: 22 октября 09:00 МСК; передача ML до 10:00 МСК.** Артефакты и проверки — в [STATUS](docs/STATUS.md), контракт backend — [ML_HANDOFF](docs/ML_HANDOFF.md).

```python
from ml.inference import BankruptcyPredictor
model = BankruptcyPredictor.load('/path/to/model_package')
score = model.predict(features)
scores = model.predict_batch(records)
```

Inference требует только `requirements-inference.txt`, модель и schema/metadata/manifest из локального пакета; TRAIN/TEST/text и обучение не нужны. Индекс остаётся внешним ключом backend.

## Данные

Положите TRAIN.csv, TEST.csv, text_data.tsv в корень проекта. Файлы исключены из Git. Их проверенные характеристики и открытые вопросы описаны в [ML_DATA](docs/ML_DATA.md). Табличный baseline использует только TRAIN.csv; текст — отдельная ветка исследований.

## Окружение и запуск

Python 3.8–3.11 с версиями библиотек из `requirements-ml.txt`. Для изолированного окружения:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
```

Проверка доступных файлов и схемы:

```bash
.venv/bin/python -m ml.audit --data-dir . --output artifacts/data-audit.json
.venv/bin/python -m ml.sample_eda --data-dir . --output artifacts/sample-eda.json
```

Обучение двух baseline и проверка сохранённых моделей:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ml.train_baselines --data-dir . --output-dir artifacts/ml-1
.venv/bin/python -m ml.verify --artifact-dir artifacts/ml-1 --data-dir .
```

Команды могут занимать значительное время и память: входные файлы занимают около 8 ГБ. Обучение создаёт локальный `float32` memmap около 1,3 ГБ, чтобы не держать полную таблицу и её копии в RAM. Артефакты локальные и исключены из Git. Запускайте один тяжёлый ML-процесс за раз. Полный контракт проверки — в [VERIFICATION](docs/VERIFICATION.md), методология — в [ML_VALIDATION](docs/ML_VALIDATION.md).

Для следующих отдельных запусков baseline используйте `--output-dir artifacts/runs --run-id auto`: каждому запуску создаётся свой каталог, существующий ID не перезаписывается. Текущий `artifacts/ml-1/` сохранён для воспроизведения исходных результатов.

Внутренние эксперименты ML-4 читают проверенный кеш и только сохранённый development. Каждый preset создаёт отдельный run ID и записывает `folds.npz`, `results.json`, снимок кода runner и модели folds. Последовательно, при ограничении потоков:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.diagnose_ml4
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_baseline
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_depth3
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset cat_l2_10
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_base
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 .venv/bin/python -m ml.optimize_tabular --preset lightgbm_leaves7
```

`--cache-dir`, `--data-dir`, `--output-dir`, `--run-id` и `--threads` позволяют задать пути и параметры запуска. `--repeat 0|1|2` выбирает заранее зафиксированное повторение групповой CV; без аргумента сохраняется исторический repeat 0. Для LightGBM нужна версия из `requirements-ml.txt`. Ограниченная серия регуляризации и правило выбора кандидата описаны в [ML_VALIDATION](docs/ML_VALIDATION.md), команды — в [VERIFICATION](docs/VERIFICATION.md).

Загруженные модели и метрики конкретного запуска проверяются командой `.venv/bin/python -m ml.verify_ml4 --run-dir artifacts/runs/<run_id>`.

Для продолжения работы агентом начните с [AGENTS.md](AGENTS.md). Рабочие инструкции проекта находятся в этом репозитории.

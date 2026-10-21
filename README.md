# Прогнозирование банкротства физических лиц

ML-подсистема для хакатона: исследование данных, обучение бинарного классификатора и подготовка переносимого Python-инференса. Backend и frontend разрабатывают другие команды. [Оригинальная постановка](docs/HACKATHON_BRIEF.txt) хранится локально в репозитории.

## Состояние

Техническая часть первого контрольного пакета (аудит, протокол валидации, два baseline) выполнена. Пакет соответствует этапам ML-1–ML-3 новой дорожной карты и ожидает независимого ревью и принятия владельцем. Точный прогресс — в [статусе](docs/STATUS.md), [активном плане](docs/exec-plans/active/ML-1.md) и [пакете ревью](docs/REVIEW_PACKAGE_ML_FOUNDATION.md). Будущий Python API описан в [ML_INTERFACE](docs/ML_INTERFACE.md).

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
.venv/bin/python -m ml.train_baselines --data-dir . --output-dir artifacts/ml-1
.venv/bin/python -m ml.verify --artifact-dir artifacts/ml-1 --data-dir .
```

Команды могут занимать значительное время и память: входные файлы занимают около 8 ГБ. Обучение создаёт локальный `float32` memmap около 1,3 ГБ, чтобы не держать полную таблицу и её копии в RAM. Артефакты локальные и исключены из Git. Запускайте один тяжёлый ML-процесс за раз. Полный контракт проверки — в [VERIFICATION](docs/VERIFICATION.md), методология — в [ML_VALIDATION](docs/ML_VALIDATION.md).

Для продолжения работы агентом начните с [AGENTS.md](AGENTS.md). Рабочие инструкции проекта находятся в этом репозитории.

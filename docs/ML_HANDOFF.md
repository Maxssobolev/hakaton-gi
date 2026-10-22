# Контракт ML-поставки

Срок: **22 октября 2026, 10:00 МСК**, freeze 09:00. API — `ml/inference.py`, класс `BankruptcyPredictor`; код backend
для формата 1 (rows65, pkg1) и формата 2 (entity-модель) одинаковый. Python **3.8.20 / Linux x86_64**, проверенная среда;
точные зависимости: numpy 1.24.4, scipy 1.10.1, lightgbm 4.5.0. Более новые Python требуют совместимых wheels
и отдельной проверки.

```python
from ml.inference import BankruptcyPredictor
model = BankruptcyPredictor.load('/path/to/package', num_threads=1)
features = {f'feature_{i}': 0.0 for i in range(1367)}  # synthetic example only
score = model.predict(features)                         # без текста: точный табличный score
score = model.predict(features, text_counts={'кредит': 2, 'платеж': 1})   # с частотами слов клиента
scores = model.predict_batch([features, features.copy()], text_counts=[None, {'долг': 3}])
version = model.model_version
```

## Вход

- Ровно 1367 ключей `feature_0`…`feature_1366`, реальные числа, конечные и представимые float32. Строки, bool, None,
  NaN/Inf, пропущенные и лишние ключи отклоняются. `0` и `-1` сохраняются, пропуски не заполняются. Порядок ключей
  произволен: массив собирается по схеме.
- `text_counts` — необязательно: словарь «слово → неотрицательная целая частота» в том же виде, что строки
  `text_data.tsv` (слово как есть, частота — целое ≥ 0). Регистр приводится внутри модели; слова с основами
  банкрот/несостоят/конкурсн/bankrupt/insolvenc игнорируются; частота 0 игнорируется. В batch — `None` либо
  последовательность словарей/`None` той же длины, что и записи.
- **Без текста** (`None`, пустой словарь или все слова отброшены) score **в точности равен табличному**.
- `index` — внешний ключ backend, не признак.

Пакетный запрос атомарен: либо список score целиком, либо исключение без частичного результата. Пустой список → `[]`.

## Выход

`float` — score класса target=1 в [0, 1], **калибровка не подтверждена**. Backend может вернуть
`{'model_version': model.model_version, 'risk_score': score}`. Пороги, категории, percentile и объяснения не поставляются.

## Что внутри формата 2 (`metadata.json → format_version: 2`)

TE15 (сглаженное среднее target по 15 категориальным столбцам) + признаки повторяющихся сущностей
(совпадения значений ID-подобных столбцов и почти-дубликаты по блокам из 32 столбцов — и, если в `metadata.json` указано
`band_widths: [32, 16]`, также из 16 столбцов — среди строк TRAIN с известной
меткой) → 3 модели LightGBM (214 деревьев, разные seed, среднее логитов) → при наличии текста бленд с Naive-Bayes
скором слов. Для объекта без похожих строк в TRAIN работают обычные табличные признаки. Обучения при загрузке
и прогнозе нет.

Если в `metadata.json` есть `negative_partner_rule` (релиз R3), то после бленда действует ещё одно правило.
Объект, у которого все близкие строки TRAIN (≥5 общих полос из 32 столбцов) имеют target=0, получает score×0.001.
Такой объект ранжируется ниже всех остальных. Для этих объектов score — только порядок, а не вероятность.

Файлы пакета: `model_0.txt`, `model_1.txt`, `model_2.txt`, `te_tables.npz`, `entity_tables.npz`, `nb_weights.npy`,
`schema.json`, `metadata.json`, `manifest.json`, `ml/__init__.py`, `ml/inference.py`, `ml/features_v2.py`,
`requirements-inference.txt`, `control_vectors.json`, `smoke_test.py`, этот документ и `wheelhouse/`.
Manifest проверяет SHA-256 каждого файла при load. Файлы готового пакета нельзя редактировать.

**Конфиденциальность:** `entity_tables.npz` и `te_tables.npz` содержат агрегаты, производные от меток TRAIN
(в том числе для отдельных значений и строк). Обращаться как с приватными данными, не публиковать.

## Установка offline из корня пакета

```bash
python3.8 -m venv /path/to/runtime
/path/to/runtime/bin/python -m pip install --no-index wheelhouse/pip-25.0.1-py3-none-any.whl
/path/to/runtime/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-inference.txt
/path/to/runtime/bin/python -I smoke_test.py
```

Linux: glibc ≥ 2.28 и `libgomp.so.1` (официальный manylinux_2_28 wheel LightGBM). Ресурсы формата 2: таблицы
сущностей держатся в памяти процесса; фактические RSS и время smoke-проверки в чистом окружении — в
`docs/CHECKPOINT_SUBMISSION3.md`. Формат 1 (rows65) по-прежнему около 65 МБ RSS.

## Ошибки

`InputValidationError` (ValueError) — схема, тип, пропуски, неверный `text_counts`; сообщение содержит позицию и поле,
без значений записи. `ModelLoadError` (ValueError) — отсутствующий файл, checksum, схема, metadata, неполные таблицы
или несовместимые версии numpy/lightgbm. `RuntimeError` — недопустимый score модели. Ошибку входа backend возвращает
клиенту как ошибку запроса; load failure — ошибка конфигурации сервиса.

Резервный пакет: `rows65-tabular-20261021T124857Z-440cc2cd-pkg1` (формат 1) — тот же API, текст игнорируется.
Пути, версии, SHA и результаты проверок — `docs/STATUS.md` и `docs/CHECKPOINT_SUBMISSION3.md`.

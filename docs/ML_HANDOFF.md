# Контракт срочной ML-поставки

Срок: **22 октября 2026, 10:00 МСК**, freeze 09:00. API реализован в `ml/inference.py`; поставляется `lightgbm_rows65`, без текста. Python **3.8.20 / Linux x86_64**, проверенная среда; точные зависимости: numpy 1.24.4, scipy 1.10.1, lightgbm 4.5.0. Backend сообщил: ограничений среды нет. Более новые Python требуют совместимых wheels и отдельной проверки; текущие pins не обещают Python 3.12+.

```python
from ml.inference import BankruptcyPredictor
model = BankruptcyPredictor.load('/path/to/package', num_threads=1)
features = {f'feature_{i}': 0.0 for i in range(1367)}  # synthetic example only
score = model.predict(features)  # float positive-class score, [0,1]
scores = model.predict_batch([features, features.copy()])  # list[float], same order
version = model.model_version
score_without_text = model.predict(features, text_counts=None)
```

Вход: ровно 1367 ключей `feature_0`…`feature_1366`, реальные числа, конечные и представимые float32. Строки с числами, bool, None, NaN/Inf, пропущенные и лишние ключи отклоняются. `0` и `-1` сохраняются. Порядок ключей словаря произволен; массив собирается по сохранённой схеме. Пропуски не заполняются. Пакетный запрос атомарен для вызывающего: либо список score целиком, либо исключение без частичного результата. `batch_size=256` ограничивает рабочую матрицу. Пустой список возвращает `[]`.

`index` — внешний ключ backend, вне features. По одному index модель признаков не извлекает. Backend сохраняет соответствие index → features и сопоставляет score с порядком входа. Для нового объекта нужны все рассчитанные признаки; исходные CSV/TSV для inference не требуются.

Выход `float` — score класса target=1, **калибровка не подтверждена**. Backend может обернуть его в `{'model_version': model.model_version, 'risk_score': score}`. Категории, пороги, percentile и объяснения пока не поставляются. Горизонт события и время формирования признаков неизвестны.

`text_counts` заранее зарезервирован: None либо словарь слово → неотрицательная целая частота; текущий tabular пакет валидирует и игнорирует его. В batch — None либо последовательность таких словарей/None той же длины. При отсутствии текста результат всегда точно табличный. Только новая версия пакета после подтверждения пользы может начать использовать текст; основные 1367 keys и формат score сохраняются.

Файлы в закрытый контур: `model.txt`, `schema.json`, `metadata.json`, `manifest.json`, `ml/__init__.py`, `ml/inference.py`, `requirements-inference.txt`, `control_vectors.json`, `smoke_test.py`, этот документ; offline wheelhouse с pip/numpy/scipy/lightgbm добавляется при финализации. Не нужны TRAIN.csv, TEST.csv, text_data.tsv, экспериментальные кеши или CatBoost. Manifest проверяет SHA-256 каждого payload-файла при load, archive SHA передаётся отдельно. Нельзя редактировать файлы готового пакета.

Установка offline из корня пакета:

```bash
python3.8 -m venv /path/to/runtime
/path/to/runtime/bin/python -m pip install --no-index wheelhouse/pip-25.0.1-py3-none-any.whl
/path/to/runtime/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-inference.txt
/path/to/runtime/bin/python -I smoke_test.py
```

Linux требуется glibc >= 2.28 и системная OpenMP runtime `libgomp.so.1`; используется официальный manylinux_2_28 wheel LightGBM. Offline bootstrap pip 25.0.1 включён для поддержки этого wheel tag. wheelhouse привязан к CPython 3.8/Linux x86_64. Другие ОС/архитектуры требуют соответствующих wheels. Предварительно: 1 CPU thread, 512 MiB RAM на процесс как стартовый бюджет; фактический RSS/время smoke фиксируются отдельно. Размер входной матрицы при batch 256 — около 1.34 MiB. Отдельный fit при загрузке/прогнозе отсутствует.

Ошибки: `InputValidationError` (ValueError) — схема/тип/пропуски/текст; сообщение содержит позицию и поле, без значений записи. `ModelLoadError` (ValueError) — отсутствующий файл, checksum, схема, metadata или несовместимая версия numpy/lightgbm. `RuntimeError` — недопустимый score модели. Ошибку входа backend возвращает клиенту как ошибку запроса; load failure — ошибка конфигурации сервиса.

Готовые пути, model version, SHA и результаты реальных проверок фиксируются в `docs/STATUS.md` и release checkpoint после сборки. Документ можно передать backend немедленно; сам чат не является отправкой команде.

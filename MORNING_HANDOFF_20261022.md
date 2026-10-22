# MORNING HANDOFF — 22.10.2026

> Финальная версия: 22.10, 02:00 МСК. Тяжёлые вычисления завершены (исследования остановлены сознательно: дальнейшие
> раунды на том же development усиливали бы адаптивный отбор при приросте порядка тысячных). Ниже только проверенные результаты.

## Статус: **READY** — R2 утверждён владельцем основным кандидатом Submission 3

Решение владельца (22.10, утро): основной — **R2**; **R1.2** — fallback и возможный Submission 4 для сравнения;
rows65 — fallback. R2 **не является строго доказанным улучшением**: исходный предрегистрированный gate раунда 2
его отклонил (seed 20261026: +0.0019 < +0.002), принятие по правилу раунда 3 — адаптивное.
Релизы зафиксированы и доступны только для чтения. CSV в All Cups загружает владелец; отправка не считается
выполненной без подтверждения платформы.

### R2 — основной (утверждён)

| что | путь (от `/home/makhzaun/Документы/Gi/hakaton/`) | SHA-256 |
| --- | --- | --- |
| **Submission 3** | `artifacts/release/R2-20261021T222359Z-22f845d4/submission3.csv` | `260bbb892ab86ad4bd872b38a531a0a2a1a4db2622c439d629c2fd48042d9668` |
| **Офлайн-пакет** (207 МБ) | `artifacts/release/R2-20261021T222359Z-22f845d4/entity-te15lkbp16-mcs100-nb-20261021T222359Z-22f845d4-pkg2.tar.gz` | `185a525339d8875b2eec3cc233a9c99f86b77c7d15da40178c26721770ad745d` |

### R1.2 — fallback и возможный Submission 4 для сравнения (проще, меньше памяти)

| что | путь | SHA-256 |
| --- | --- | --- |
| Submission 3 | `artifacts/release/R1.2-20261021T212050Z-7b1d4c1f/submission3.csv` | `c59aa747f37bcd126166a9f03b00ec75805d47550937c0359f7f49be00744f7d` |
| Офлайн-пакет (136 МБ) | `artifacts/release/R1.2-20261021T212050Z-7b1d4c1f/entity-te15lkbp-nb-20261021T212050Z-7b1d4c1f-pkg2.tar.gz` | `b22e94e007374123095157a4adbf9c9362dc2c34b182b7a3c390a336ce4765f3` |

### Fallback (в каждом релизе, папка `fallback-rows65/`)

| что | SHA-256 |
| --- | --- |
| Submission 1 rows65 (public 0.64459) | `28637aff6b1c4aaddf0a00f88673c3f859ac43d9b6b5f1ee31f4a09a02bc0057` |
| пакет rows65 pkg1 | `fa5a23cb7cdb8dd0213145666b1d0a83223b9f0a7aea0bacb06847c3749f950b` |

Целостность: в каталоге релиза `sha256sum -c SHA256SUMS`.

## Лучший ROC-AUC (построчная CV на development, 5 folds, с NB)

| seed | rows65 | R1.2 | **R2** |
| --- | --- | --- | --- |
| 20261024 | 0.6490 | 0.6904 | **0.6930** |
| 20261026 | 0.6557 | 0.6921 | **0.6940** |
| 20261027 | — | 0.6868 | **0.6904** |
| 20261028 | — | 0.6939 | **0.6967** |
| 20261029 | — | 0.6959 | **0.6984** |

- Внешний holdout (разведочно, только R1): 0.6593 против rows65 0.6342 (+0.025, 95% [0.014; 0.036]).
  Для выбора между R1.2 и R2 holdout не использовался.
- Ожидается заметный рост public относительно 0.64459 (≈ +0.03…0.04 по CV); точное значение заранее неизвестно.

**Как выбрать между R2 и R1.2.** R2 добавляет мелкие «полосы» по 16 столбцов для поиска почти-дубликатов и
`min_child_samples=100`. Прирост +0.0027 в среднем по пяти seed (21 из 25 folds), но принят адаптивно: раунд 2 по
строгому правилу его отклонил (seed 20261026: +0.0019 при пороге +0.002), подтверждение на трёх чистых seed
получено в раунде 3. R2 тяжелее: ~430 МБ RAM и 11 мс на запрос против 330 МБ и 9 мс у R1.2. Моя рекомендация — R2.
Если нужна максимальная строгость или экономия памяти — R1.2.

## Что за модель

TE15 + признаки повторяющихся сущностей (метки почти совпадающих строк TRAIN: совпадения ID-подобных столбцов,
точные дубликаты, ближайший партнёр по блокам столбцов) + 3×LightGBM (214 деревьев) + Naive-Bayes по тексту.
Основание: TRAIN/TEST — случайный построчный split (7.65% строк TEST имеют точный дубликат в TRAIN при 7.67%
ожидаемых), в данных повторные снимки одних и тех же людей. Используются только метки TRAIN — без меток TEST
(их нет), внешних данных и подбора по лидерборду; правила (§3.1 брифа) разрешают любые подходы к моделированию.
Подробно — `docs/CHECKPOINT_SUBMISSION3.md`.

## Проверки (каждого релиза)

- **Офлайн-установка в чистом venv** (Python 3.8.20, только wheelhouse, доступ к репозиторию/данным/сети запрещён):
  PASS — R2: 6.4 с; R1.2: 4.5 с. Квитанции — в `offline-verification/` каталога релиза.
- **Single-row = batch на всех 106274 строках TEST** (с текстом): max |Δ| **0.0**; батч = CSV: **0.0**;
  `predict(features, text_counts)` на 3000 строках: **0.0**; `predict_batch` 1/7/256: **0.0** —
  `checks/single_vs_batch.json`. Нормализация TE15+NB: все таблицы и z-константы заморожены в пакете.
- Пакет = **независимый исследовательский эталон** (max |Δ| 2.2e-16). Код пакета = код репозитория (для R2).
  40/40 unit tests.
- Cross-fitting: обучающая строка не видит своей метки в таблицах значений. Исключение — общий prior TE, где
  метка входит с весом ~1/248000. Метки validation/TEST в признаках не участвуют.

## Независимое ревью (Codex CLI, read-only)

- **Код поставки (2 прохода).** Первый проход — FAIL, 3 MAJOR:
  - №1 — research early stopping; финал проверен с фиксированными деревьями, поэтому не затронут;
  - №2 — переносимость формата 1; исправлено;
  - №3 — память `predict_batch`; исправлено.
  Второй проход подтвердил исправления.
- **R2: PASS**, BLOCKER/MAJOR нет. Единственное исправление — привязка `cv_evidence` к точной конфигурации; адаптивность отбора оговорена.
- Утечек меток validation/TEST и расхождений сабмита с пакетом ревью не нашло. Подробно —
  `docs/REVIEW_PACKAGE_ENTITY.md`; отчёты — `/home/makhzaun/Документы/Gi/hakaton-research/work/logs/codex_review*.md`.

## Инструкция для backend (на примере R2)

Полный контракт — `ML_HANDOFF.md` внутри архива. API тот же, что у rows65; `text_counts` теперь используется.

```bash
tar -xzf entity-te15lkbp16-mcs100-nb-20261021T222359Z-22f845d4-pkg2.tar.gz
cd entity-te15lkbp16-mcs100-nb-20261021T222359Z-22f845d4-pkg2
python3.8 -m venv /opt/ml-runtime
/opt/ml-runtime/bin/python -m pip install --no-index wheelhouse/pip-25.0.1-py3-none-any.whl
/opt/ml-runtime/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-inference.txt
/opt/ml-runtime/bin/python -I smoke_test.py          # ожидается {"status": "PASS", ...}
```

```python
from ml.inference import BankruptcyPredictor
model = BankruptcyPredictor.load('/path/to/pkg2', num_threads=1)   # один раз на процесс: R2 ~3.4 с, ~430 МБ RSS
score = model.predict(features_dict)                                  # без текста — точный табличный score
score = model.predict(features_dict, text_counts={'слово': 3, ...})   # с частотами слов клиента
scores = model.predict_batch(list_of_features, text_counts=list_of_dicts_or_None)
```

- Прогноз одной записи ~11 мс (R2) / ~9 мс (R1.2), 1 поток. Linux x86_64, glibc ≥ 2.28, `libgomp.so.1`;
  зависимости только numpy 1.24.4 / scipy 1.10.1 / lightgbm 4.5.0 (в wheelhouse).
- **Сценарий «по index» (§3.2 брифа):** для объектов **из TRAIN** модель находит саму строку с её меткой, поэтому
  score — не прогноз (в демо стоит оговорить). Честный прогноз — для объектов TEST и новых объектов.
- `entity_tables.npz` и `te_tables.npz` внутри пакета — приватные производные меток TRAIN; не публиковать.

## Где что лежит

- Код: `ml/features_v2.py`, `ml/inference.py`, `ml/entity_delivery.py`, `ml/make_submission_v2.py`,
  `ml/verify_single_batch.py`; тесты `tests/test_entity.py`; исследовательский код `research/entity_lookup/`.
- Сборка R2 заново: `ml.entity_delivery --band-widths 32,16 --min-child-samples 100`; полный цикл —
  `research/entity_lookup/build_release.sh`.
- Документы: `docs/STATUS.md`, `docs/CHECKPOINT_SUBMISSION3.md`, `docs/REVIEW_PACKAGE_ENTITY.md`,
  `docs/ML_VALIDATION.md` (раздел «Построчная CV»), `docs/ML_EXPERIMENTS.md`.
- Журнал исследований с предрегистрациями, логи, OOF, эталоны: `/home/makhzaun/Документы/Gi/hakaton-research/`.
- Более ранние проверенные сборки той же модели R1 (`R1-…`, `R1.1-…`) — в `artifacts/release/`.

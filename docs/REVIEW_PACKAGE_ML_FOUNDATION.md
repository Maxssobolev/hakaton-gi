# Review package: ML Foundation

**Дата подготовки:** 2026-10-21. **Stage ID:** ML-1–ML-3 по текущей [дорожной карте](ML_PLAN.md); один исходный [активный план ML-1](exec-plans/active/ML-1.md). **Название:** Data Audit / EDA + Validation Protocol + Baseline Models. **Статус:** исходный пакет R1 опубликован в draft PR; независимое ревью вернуло FIXES REQUIRED. Локальные исправления описаны в [пакете R2](REVIEW_PACKAGE_ML_FOUNDATION_R2.md), owner acceptance не получено.

## Цель и критерии принятия

Создать воспроизводимую исходную точку для дальнейшего ML: подтвердить схему данных, исключить известные пересечения идентичных векторов между development и holdout, обучить два baseline на одном split, сохранить модели и повторить прогнозы после загрузки. Критерии перечислены в [активном плане](exec-plans/active/ML-1.md). Это контрольный пакет, а не финальная модель или переносимый inference package.

## Git checkpoint

- Ветка ревью: `review/ml-foundation`.
- Base SHA (`main`): `1583412ba2bbf742ae28929315fd5a4ecb5170c8`.
- Head SHA первоначального проверенного снимка: `909ca6a5ae9f4c3306a896b4ec6c76dcaa8b8927`. Последующие коммиты с метаданными PR могут сдвинуть tip ветки; актуальный head указан в PR.
- [Draft PR #1](https://github.com/Maxssobolev/hakaton-gi/pull/1) в публичном [hakaton-gi](https://github.com/Maxssobolev/hakaton-gi). На момент открытия PR содержал 23 файла ML-кода и документации. Merge не выполнен.

## Файлы контрольного пакета

- Рабочие инструкции и рамки: `AGENTS.md`, `README.md`, `.gitignore`, `requirements-ml.txt`, `docs/PRODUCT.md`, `docs/ARCHITECTURE.md`, `docs/WORKFLOW.md`, `docs/REVIEW.md`, `docs/VERIFICATION.md`.
- ML-документы: `docs/ML_DATA.md`, `docs/ML_VALIDATION.md`, `docs/ML_PLAN.md`, `docs/ML_EXPERIMENTS.md`, `docs/ML_INTERFACE.md`, `docs/STATUS.md`, `docs/exec-plans/active/ML-1.md`, этот review package.
- Код: `ml/__init__.py`, `ml/common.py`, `ml/audit.py`, `ml/sample_eda.py`, `ml/train_baselines.py`, `ml/verify.py`.
- `docs/HACKATHON_BRIEF.txt` — копия исходного задания; владелец подтвердил право публикации 2026-10-21.
- `docs/HARNESS_PRINCIPLES.md`, `docs/MODULES.md`, `prompts/`, `templates/` существовали в рабочей папке до ML bootstrap. Они не требуются для работы ML-кода и не входят в checkpoint.
- Сырые `TRAIN.csv`, `TEST.csv`, `text_data.tsv`, локальные `artifacts/` и `.venv/` не входят в публикуемый пакет.

## Реально выполненные проверки

Запуски исходного этапа 2026-10-21 записаны в [плане](exec-plans/active/ML-1.md) и [журнале](ML_EXPERIMENTS.md):

| Проверка | Фактический результат |
| --- | --- |
| `ml.audit` | PASS: TRAIN 247 972, TEST 106 274, 1 367 признаков, 7 констант, 0 нефинитных значений, 0 лишних text index |
| `ml.sample_eda` | PASS: выборочные дубли, сильные корреляции и диапазоны описаны в ML_DATA |
| `ml.train_baselines` | PASS: обе модели и локальные результаты сохранены; первая попытка завершилась кодом 137 и отдельно записана в журнале |
| `ml.verify` | PASS: SHA TRAIN, схемы, внешний/внутренний split, загрузка моделей, контрольные прогнозы и полный пересчёт holdout ROC-AUC/AP |
| `py_compile` | PASS для исходного bootstrap; код дополнением не менялся |
| Повторный `ml.verify` после дополнения | PASS 2026-10-21: обе загруженные модели дали те же контрольные прогнозы и holdout-метрики; split, схема и хэш источника совпали |
| Markdown links после дополнения | PASS 2026-10-21: проверен 31 Markdown-файл вне `.venv`/`.git`, 0 битых относительных ссылок |
| `git diff --check` после дополнения и перед публикацией | PASS 2026-10-21 для staged файлов; перед первым commit список из 23 файлов проверен отдельно |
| `git check-ignore` после дополнения | PASS 2026-10-21: TRAIN.csv, TEST.csv, text_data.tsv и `artifacts/ml-1/results.json` игнорируются |

Этот раздел описывает проверки перед R1. Проверки переноса в другой Python-контур, backend/frontend и API отсутствуют, поскольку этих компонентов нет в текущем scope. Исправления и проверки после R1 — в [пакете R2](REVIEW_PACKAGE_ML_FOUNDATION_R2.md).

## Данные, конфигурация и метрики

- TRAIN SHA-256: `70063b1cad489056102f3452d838b02e3a2dcfd4a1ac1134c90079cc83f14c70`. Табличные признаки: `feature_0`…`feature_1366`, тип `float32`; `index`, `target` и текст исключены из X.
- Внешнее разделение: хэш вектора признаков, выбор 20% групп в holdout со стратификацией групп, seed 42. Development: 197 960 строк / 2 674 положительных; holdout: 50 012 / 672. Внутренний групповой split CatBoost: 10% групп development, seed 43.
- SGD Logistic Regression: `StandardScaler` только на development, `loss=log_loss`, `alpha=1e-4`, 8 эпох `partial_fit`, batch 4 096, `average=True`. Holdout ROC-AUC **0,5284581173876117**, Average Precision **0,01529609583263981**.
- CatBoost: depth 5, learning rate 0,05, L2 3, максимум 300 итераций, 4 CPU threads; early stopping выбрал 191 итерацию, затем переобучение на development. Holdout ROC-AUC **0,617529770952767**, Average Precision **0,020399761704370918**.
- Пиковый RSS обучения 4 296,28125 MiB. Версии библиотек и полные результаты — в [журнале](ML_EXPERIMENTS.md) и локальном `artifacts/ml-1/results.json`.
- Локальные артефакты: `artifacts/data-audit.json`, `artifacts/sample-eda.json`, `artifacts/ml-1/` (модели, схема, массивы, split, результаты). Они игнорируются Git и для reviewer через GitHub недоступны. Публичные агрегаты, конфигурации и команды записаны в документации; воспроизведение требует законного доступа к исходным файлам.

## Известные ограничения

- Внешний holdout уже просмотрен для двух baseline. Дальнейший подбор вести только внутри development; план независимой финальной оценки согласовать отдельно.
- Группы одинаковых `float32` векторов не гарантируют разделение разных записей одного субъекта. Время формирования признаков и горизонт target неизвестны.
- ROC-AUC CatBoost 0,6175 и AP 0,0204 — слабая стартовая точка, не подтверждение пригодности к продуктовой оценке риска.
- Score не калиброван; рабочие пороги, процентиль и человеческие объяснения не установлены.
- Текстовая ветка, финальная модель, переносимый Python API и submission ещё не реализованы.
- В публичном GitHub нельзя размещать сырые датасеты или чувствительные артефакты; право публикации исходного задания владелец подтвердил отдельно.

## Решения владельца

1. **Решено 2026-10-21:** владелец подтвердил публикацию исходного задания и разрешил отдельную ветку, commit, push и draft PR с проверенными кодом и документацией, без исходных данных и локальных артефактов.
2. После независимого вердикта владелец принимает пакет ML Foundation или запрашивает исправления. Только после принятия начинать ML-4.

## Результаты независимого ревью и принятия

R1 по HEAD `56835d381bc69588f4f2bea724db93407ffabb46`: **FIXES REQUIRED**, 0 Blocker / 2 Important / 3 Minor. I-01 — целостность кеша; I-02 — автоматические инвариантные тесты; M-01 — граница полной проверки текста; M-02 — Precision@K/Recall@K и распределение score в плане ML-4; M-03 — уникальные run IDs для следующих экспериментов. Подробности исправлений и реальные проверки — в [R2](REVIEW_PACKAGE_ML_FOUNDATION_R2.md). Повторного ревью и принятия владельцем ещё нет; план остаётся активным.

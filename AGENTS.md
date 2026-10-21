# Проект: ML-подсистема прогнозирования банкротства

Наша команда отвечает за данные, модели, валидацию и переносимый Python-инференс. Backend и frontend создают другие команды. Репозиторий, а не история чата, является источником текущих правил и результатов.

## Куда смотреть

| Вопрос | Файл |
| --- | --- |
| Оригинальная постановка | [docs/HACKATHON_BRIEF.txt](docs/HACKATHON_BRIEF.txt) |
| Назначение и ограничения | [docs/PRODUCT.md](docs/PRODUCT.md) |
| Текущая и целевая архитектура | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Данные и проверенные факты | [docs/ML_DATA.md](docs/ML_DATA.md) |
| Этапы ML и выводы из учебных материалов | [docs/ML_PLAN.md](docs/ML_PLAN.md) |
| Правила валидации | [docs/ML_VALIDATION.md](docs/ML_VALIDATION.md) |
| Контракт передачи ML-компонента | [docs/ML_INTERFACE.md](docs/ML_INTERFACE.md) |
| Активный этап и его проверки | [docs/STATUS.md](docs/STATUS.md) → docs/exec-plans/active/ |
| Пакет для независимой проверки | [docs/REVIEW_PACKAGE_ML_FOUNDATION.md](docs/REVIEW_PACKAGE_ML_FOUNDATION.md) |
| История реальных запусков | [docs/ML_EXPERIMENTS.md](docs/ML_EXPERIMENTS.md) |
| Рабочий процесс и границы полномочий | [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| Команды проверки | [docs/VERIFICATION.md](docs/VERIFICATION.md) |

## Перед изменениями

Прочитай относящиеся к задаче документы и активный execution plan; проверь git status и текущий diff. Делай минимальный проверяемый шаг, запускай реальные проверки, записывай только наблюдённые метрики и сверяй документацию с кодом. Существенные решения фиксируй в документации. Автор работы — Codex; независимое ML-ревью проводит ChatGPT в отдельной сессии по опубликованному Git-снимку; владелец принимает этап. После существенного этапа подготовь review package, дождись ревью и принятия владельцем перед следующим крупным этапом. Не делай git commit, push, merge или публикацию без отдельного разрешения владельца.

Оригинальные TRAIN.csv, TEST.csv, text_data.tsv не изменять и не публиковать. Их строки не помещать в логи и отчёты. Правила проведения экспериментов находятся в [docs/ML_VALIDATION.md](docs/ML_VALIDATION.md).

# Срочная ML-поставка — 22 октября 2026, 10:00 МСК

Основание: новое прямое задание владельца 2026-10-21. Оно разрешает немедленный inference, submission, текстовый эксперимент, необходимые Git-операции и параллельное критическое ревью. Старые межэтапные ворота не блокируют этот scope. ML Foundation принят; ML-4 tuning прекращён, кандидат `lightgbm_rows65`. Историческое внешнее ревью ML-4 не объявляем проведённым.

## Порядок и критерии

1. Табличный portable API, загрузка без fit, строгие 1367 float32 features, synthetic smoke; короткий контракт backend.
2. Full TRAIN refit rows65 без нового tuning: число деревьев — медиана девяти ранее сохранённых stopping iterations (191). Это заранее фиксированное правило, внешний holdout и TEST labels не используются. Локальная релизная модель, артефакты и submission никогда не публикуются в GitHub.
3. Submission 1: строгий потоковый CSV validator с повторным предсказанием по TEST, SHA-256 и provenance, уникальный каталог. Проверить доступ All Cups, отправить при доступной сессии.
4. Текст: однократный полный streaming аудит и hashing CSR с реальными log1p частотами; 2**18 bins, L2 row normalization, missing indicator. Нет fit vocabulary/IDF. До просмотра текстовых метрик исключаются слова с основами банкрот, несостоят, конкурсн, bankrupt, insolvenc как очевидные кандидаты на outcome leakage; считаются агрегаты их присутствия. Это не доказывает отсутствие других утечек. Fixed SGDClassifier(loss=log_loss, alpha=1e-4, average=True, random_state=42), 5 epochs partial_fit с перемешиванием train rows, без поиска. Причина: реально свободно около 5.9 GiB RAM, полный liblinear может потребовать несколько копий CSR; streaming SGD ограничивает память батчем 2048 строк.
5. A/B/C на сохранённых rows65 folds repeats 1 и 2 (6 folds). A — повторно загруженные модели, B — текст fold train only; C — заранее фиксированное 50/50 усреднение сырых OOF score, без обучения весов. Формула та же при инференсе. Это не stacking, вес не подбирается. Для отсутствующего текста C равен A. При отсутствии текста любой финальный ансамбль возвращает точный табличный fallback.
6. Включение текста только при устойчивом улучшении (оба repeats и минимум 5/6 folds), отсутствии очевидной утечки, воспроизводимом TEST inference и готовом переносе. Неустановленное время формирования текста — явное ограничение, подозрительные outcome слова исследуются до решения.
7. Финальный пакет: runtime source, model/schema/metadata, requirements и offline wheels, synthetic controls, smoke, SHA-256 manifest, архив; чистая среда без доступа к TRAIN/TEST/text. Freeze 2026-10-22 09:00 МСК, handoff до 10:00.

## Ревью и checkpoints

Параллельное агентное read-only ревью — дополнительная проверка, не подмена независимого ChatGPT review по опубликованному снимку. Подготовить три checkpoints: submission; текст; финальный inference. Косметические замечания не задерживают пакет. Текущие чужие untracked harness-файлы не включать в commits.

## Результаты

**Техническая часть готова досрочно 2026-10-21.** Final candidate rows65 tabular; новые эксперименты не запускаются. План остаётся active для формальной приёмки владельцем/фактической передачи и результата ручной отправки.

- Табличный refit/inference и Submission 1 выполнены; [checkpoint 1](../../CHECKPOINT_SUBMISSION1.md).
- Offline пакет с wheelhouse, усиленный clean smoke и дополнительное агентное ревью — PASS; [checkpoint inference](../../CHECKPOINT_INFERENCE.md). Владелец выбрал ручную отправку CSV после отказа upload-инструмента; platform score пока отсутствует.
- Полный текстовый parse завершён: 274842 строк, формат/joins валидны, частоты 1…11583, 396 пустых; 1276.85 s, 358661005 CSR nnz. Запущены lexical screen и fixed A/B/C experiment.

- Text experiment run `20261021T132201Z-39481bde` завершён: A AUC0.639551/AP0.024494; B0.549281/0.015879; C0.594379/0.018083. Fixed blend проиграл6/6, оба repeats ухудшились; текст не включается, Submission2 не создаётся. Все106274 TEST text scores finite [0,1], но final text model не выпускается. [Checkpoint text](../../CHECKPOINT_TEXT.md).
- Дополнительное критическое агентное review финальных artifacts/OOF/метрик — PASS, blocker/important нет. Внешний ChatGPT review по опубликованному снимку и owner acceptance не выдумываются. Пакет pkg1 оставлен неизменным; manual platform submission и backend installation остаются внешними действиями.

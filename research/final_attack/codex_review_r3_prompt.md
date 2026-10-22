Ты — независимый ревьюер (только чтение; ничего не изменяй, не делай commit, не загружай датасеты TRAIN/TEST/text_data и
artifacts/ml-1, не запускай обучение и тяжёлые скрипты; разрешены чтение кода, `git diff` и
`.venv/bin/python -m unittest tests.test_entity`). Лимит — 7 минут: дай вердикт быстро.

Изменение R2 → R3 (кандидат на финальный сабмит, дедлайн через ~30 минут): к финальному score R2 (LightGBM×3 + NB-бленд)
добавлено постправило: строки, у которых все близкие band-партнёры в TRAIN (≥5 общих полос шириной 32: признаки
bp_n5>0 и bp_pos5==0, те же признаки, что уже вход модели R2) отрицательные, получают score×0.001, то есть опускаются
ниже всех остальных строк с сохранением внутреннего порядка.

Файлы:
- Код поставки: `git diff -- ml/inference.py tests/test_entity.py` (в репозитории /home/makhzaun/Документы/Gi/hakaton).
- Исследование (вне репозитория): /home/makhzaun/Документы/Gi/hakaton-research/work/final_attack/
  quick_oof.py (ансамбль R2+R1.2, отклонён), diag_entity.py (диагностика на seeds 24/26), compute_xv.py (признаки
  validation-строк из меток fit-части fold), confirm_rule.py (проверка правила), build_rule_csv.py (CSV = R2 CSV + маска
  TEST из таблиц пакета R2), make_r3_package.py (пакет R3 = пакет R2 + флаг в metadata + новый inference.py).
  Реализация признаков для CV: /home/makhzaun/Документы/Gi/hakaton-research/work/scripts/{band_partner.py,lookup_features.py,rowcv3.py}.
- Журнал с предрегистрацией (раздел FINAL ATTACK в конце): /home/makhzaun/Документы/Gi/hakaton-research/EXPERIMENTS.md.
Результат правила к R2 (с NB, те же folds): чистые seeds 27/28/29: +0.00411 (5/5), +0.00389 (5/5), +0.00409 (5/5);
seeds 24/26 (на них правило найдено): +0.00314 (4/5), +0.00272 (5/5). На TEST маска: 8668 из 106274 строк (8.2%),
в CV ~6.1% (fit-часть меньше: 158k против 248k строк TRAIN).

Проверь:
1. Утечка меток: признаки маски для validation-строк — только из меток fit-части fold; собственная метка не используется;
   OOF R2 и xv выровнены по одинаковым позициям (StratifiedKFold(5, shuffle, seed)).
2. Совпадает ли маска в inference.py (индексы bp_n5/bp_pos5 для W=32) с маской в build_rule_csv.py и в CV (band_partner.py).
3. Корректность изменения inference.py: single=batch, валидация метаданных, обратная совместимость (без флага — как R2).
4. Методологические риски: selection bias (правило найдено на 24/26), перенос на TEST (больше fit-строк → больше маска),
   жёсткий перенос в низ против мягкого сдвига; что может пойти не так на Public/Private.
Ответ по-русски, кратко: находки BLOCKER/MAJOR/MINOR/NOTE с файлом:строкой и вердикт PASS/FAIL.

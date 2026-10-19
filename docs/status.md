# Состояние проекта

Дата: 2026-10-16. Здесь только текущий этап, последние проверенные запуски,
следующие шаги и открытые вопросы. Где что лежит — [карта проекта](project_map.md).

## Фокус: основания SBP-SAT

`src/sbp_sat/` — отдельный NumPy-прототип: периодический x, свободные верхняя
и нижняя границы, точечные приёмники, точный дискретный adjoint и градиент по
log(Vp)/log(Vs) при фиксированной плотности. Его подключение к рабочему
SAFOD/DAS workflow — последующий этап. Формулы — [method.md](sbp_sat/method.md),
проверки и команды — [validation.md](sbp_sat/validation.md).

## Последние проверенные запуски

| Запуск | Что содержит | Итог |
| --- | --- | --- |
| [reference_v2](../results/validation/sbp_sat/reference_v2/validation_report.md) | результаты исходного пакета QC v2 (Python 3.12.14, NumPy 2.3.5): `oblique/`, `time_energy/`, `controls.json`, `sha256.json` | по отчёту прошли все 16 наклонных случаев и проверка полностью дискретной энергии; контроль с отключённым верхним SAT не проходит, как и должен |
| [sherlock_20261016](../results/validation/sbp_sat/sherlock_20261016/validation_report.md) | локальные диагностики Sherlock: `oblique/`, `time_energy/` | `passed = true` в обоих `summary.json` |

- `sherlock_20261016/time_energy/summary.json` побайтно совпадает с
  `reference_v2/time_energy/summary.json`. Повторный запуск `time_energy_qc`
  при реструктуризации 2026-10-16 воспроизвёл его точно; отдельная папка
  этого запуска в `results/` не сохранялась.
- `oblique/summary.json` двух запусков совпадают по структуре и флагам
  `passed`; числа отличаются в младших знаках (относительная разница не
  более 4e-10).
- `reference_v2/controls.json` (отключённый верхний SAT; изоляция нижней
  границы, глубина 800 и 1000 м) не создаётся ни одним скриптом проекта:
  `oblique_qc` и `time_energy_qc` его не пишут. Готовой проверки для этих
  контролей нет: в `oblique_qc` нет опции отключения SAT, а сравнение глубин
  800 и 1000 м скрипт сам не выполняет (есть только параметр `--depth`).
- `operator_qc` 2026-10-16 в окружении `fwi` проходит.

## Следующие шаги

В SBP-SAT уже реализованы 2D forward, точный дискретный adjoint и градиент.
Ближайшая задача — закрепить основания маленькими независимыми проверками:

1. Проверить 1D производные на полиномах, положительность норм, проекции
   на обе границы и SBP-тождество, включая минимальные 8 интервалов.
2. Для `rho * vx_t = sxz_z`, `sxz_t = mu * vx_z` вывести поток энергии и
   показать его компенсацию SAT. Неверный знак или отключение SAT должны
   обнаруживаться проверкой.
3. После этого разделять обязанности `elastic2d.py`, сравнивая поля до/после
   и сохраняя точность energy, dot-product и Taylor проверок.

Поглощающие границы, интерполяция, DAS, ускорение и SAFOD подключаются
последовательно, с проверкой сопряжённого оператора после каждого добавления.

## Открытые вопросы

1. **Событие 2026-10-03: v3 или v5.** Для nc75443276 есть два варианта
   сравнения DAS и MH030: `plot_oct03_m08_das_mh030_pb_v3.py` (нулевая фаза,
   `sosfiltfilt`; пишет `compare/das_mh030_comparison_v3_zerophase`) и
   `plot_oct03_m08_das_mh030_pb_v5.py` (каузальный `DASutils.bandpass2D_c`;
   пишет `compare/das_mh030_comparison_v5_causal`). Какой вариант итоговый,
   решает автор. Событие не внесено в `settings.EVENTS`, папки
   `results/events/20261003_75443276/` в проекте нет. Подробнее —
   [README экспериментов](../scripts/safod/experiments/oct03_2026/README.md).
2. **Общая начальная модель по умолчанию.** У `run_forward` по умолчанию
   `--initial-model bill_logs`, у `compare_event` —
   `hybrid_zhang2009_boness2006_bill_logs`. Пока выбор не сделан, оба скрипта
   запускаются с одинаковыми явными `--initial-model` и `--theta-deg`
   ([workflow.md](safod/workflow.md)).
3. **`scripts/legacy/safod_deep/plot_two_selected_events_pretty.py`** не
   импортируется: в `scripts/safod_deep/plot_catalog_events.py` больше нет
   `infer_reference_locus_offset`, `load_reference_channel_limits` и
   `physical_reference_axis_and_crop`. Его результат
   `results/safod_deep/event_qc_pretty_two/` сохранён. Восстановить скрипт или
   оставить как запись — решает автор.

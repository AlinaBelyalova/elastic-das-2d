# Результаты расчётов

Здесь лежит всё вычисленное: отчёты, JSON/CSV, массивы, рисунки и логи.
Отчёт, числа, рисунки и лог одного запуска хранятся в его папке. Методика —
в `docs/`, расположение кода — в [карте проекта](../docs/project_map.md).

Содержимое `results/` локальное и исключено из Git (`results/*` в
`.gitignore`); версионируется только этот README.

## Разделы

| Папка | Кто создаёт | Содержимое |
| --- | --- | --- |
| `validation/sbp_sat/<запуск>/` | `scripts.validation.sbp_sat.oblique_qc`, `time_energy_qc` с `--output-dir` | `oblique/`, `time_energy/` (`summary.json`, `traces.npz`, PNG с `--plot`); общий `validation_report.md` запуска пишется вручную, скрипты его не создают. Сейчас: `reference_v2/`, `sherlock_20261016/` |
| `validation/elastic/<проверка>/` | `scripts/validation/elastic/` | по папке на каждый скрипт, который пишет файлы: `boundary_absorption/`, `free_surface_normal_incidence/`, `magnitude_scaling/`, `moment_tensor_analytical/`, `moment_tensor_linearity/`, `pointforce_2d/`, `reciprocity/`, `source_radiation/` |
| `validation/fwi/<проверка>/` | `scripts/validation/fwi/` | `two_layer_numba/`, `two_layer_taylor/` |
| `examples/homogeneous_demo/` | `scripts.examples.homogeneous_demo` | однородный демонстрационный расчёт, `explosive_validation/` |
| `benchmarks/performance/` | `scripts.benchmarks.performance` | сейчас только `benchmark_scaling_free_surface_true.json`; PNG скрипт создаёт при каждом запуске (вместе с JSON, перезаписывая их) |
| `events/<YYYYMMDD_номер>/` | конвейер `scripts/safod/` | одно событие SAFOD, раскладка ниже |
| `safod/` | `scripts/safod/selection/` | `catalog_search/` (`find_catalog_events`, `check_catalog_coverage`); `event_screening.csv` (`screen_events`) |
| `safod_deep/` | `scripts/safod_deep/`, `web_safod` | архив SAFOD-deep, раскладка ниже |
| `legacy/` | прежние версии кода | старые расчёты и манифесты переносов, индекс ниже |

Проверки, которые только печатают результат (например `operator_qc`),
папок не создают; список — в
[scripts/validation/README.md](../scripts/validation/README.md).

## Событие SAFOD

Событие выбирается переменной `SAFOD_EVENT_KEY=nc<номер>` и должно быть
внесено в `scripts/safod/settings.py` (реестр `EVENTS` и `aliases`). Имя папки —
`EVENT_RESULT_KEY` = дата origin time и номер, например `nc75379261` →
`events/20260618_75379261/`. Порядок запуска — в
[docs/safod/workflow.md](../docs/safod/workflow.md).

```text
events/<YYYYMMDD_номер>/
├── real/                            prepare_event: real_das_event_window.npz, геометрия, превью
│   └── channel_registration_qc/     calibrate_channel_registration (--out-dir), если регистрация каналов другая
├── forward/dc<θ>/<initial_model>/   run_forward
├── compare/dc<θ>/<initial_model>/   compare_event
├── compare/<анализ>[_YYYYMMDD]/     прочие сравнения, например DAS и геофон
└── logs/                            вывод запусков (скрипты сами лог не пишут)
```

`dc<θ>` — значение `--theta-deg` (ориентация 2-D double couple) в целых
градусах, три цифры (`dc035` для 35°).
`<initial_model>` — значение `--initial-model`.

Сохраняются по соглашению handoff 2026-08-14 (не переименовываются):

- `events/20260401_75336802/{forward,compare}/dc035/legacy_unversioned/` —
  апрельские расчёты до разделения по именам моделей;
- `events/20260618_75379261/{forward,compare}/dc035/digitized_log/` — прежнее
  имя модели `bill_logs`.

В `events/20260401_75336802/` анализы с суффиксом `_20260807` — поколение
2026-08-07, перенесённое из `real/`; `compare/das_geophone_physical_pairs/` —
папка по умолчанию `plot_das_geophone_two_physical_comparisons`. Файлы
`real/*_0_15s*` (2026-07-08) названы по схеме прежнего скрипта подготовки
(как в `scripts/legacy/prepare_real_event_20260413.py`); текущий
`prepare_event` их не пишет.

## Архив SAFOD-deep

| Папка | Кто создаёт | Содержимое |
| --- | --- | --- |
| `safod_deep/catalog/` | `scan_nano_manifest`, `build_event_catalog`; `focal_mechanisms.json` — `web_safod.scripts.fetch_focal_mechanisms` | манифесты записей, каталог NCEDC, обзорные рисунки, `CATALOG_NOTES.md`; для `recording_manifest_pre_nano.csv` скрипта-источника в проекте нет |
| `safod_deep/geometry/` | `plot_catalog_events --prepare-geometry` | `acquisition_registration.json` и QC-рисунок: регистрация конфигураций записи на физическую геометрию кабеля. Для `nano_mapping.csv` и `nano_registration.*` скрипта-источника в проекте нет (уточнить у автора) |
| `safod_deep/event_qc/` | `plot_catalog_events`, `plot_catalog_events_array.sbatch` | QC-галерея событий и `metadata/`; её читает `web_safod` (`web_safod/config/explorer.json`) |
| `safod_deep/event_qc_1_20Hz/`, `event_qc_3_15Hz/`, `event_qc_5_30Hz/`, `event_qc_adaptive/` | более ранние запуски (август 2026) | прежние варианты галереи; `web_safod` их не читает |
| `safod_deep/event_qc_pretty_two/` | `scripts/legacy/safod_deep/plot_two_selected_events_pretty.py` | рисунки двух выбранных событий; скрипт сейчас не запускается |
| `safod_deep/logs/` | Slurm | логи `plot_catalog_events_array.sbatch` |

## Legacy

| Папка | Содержимое |
| --- | --- |
| `legacy/mechanism_compare_20260718/` | сравнения реальных и синтетических данных для 20260401_75336802 при θ = 0, 15, 30, 45, 60, 75° (июль 2026, прежняя схема имён) |
| `legacy/migrations/` | `results_reorg_20260807_223722.txt` — манифест переноса результатов 2026-08-07 |
| `legacy/numpy_homogeneous_baseline_20260604/` | прежний однородный расчёт NumPy (2026-06-04) |
| `legacy/real_event_20260413_M153/` | подготовка события 2026-04-13 скриптом `scripts/legacy/prepare_real_event_20260413.py`; какие файлы он пишет — в [scripts/legacy/README.md](../scripts/legacy/README.md) |
| `legacy/safod_initial_model_20260629/` | прежняя начальная модель SAFOD: `safod_initial_model.npz` и рисунки Vp, Vs, rho |

Индекс старого кода — [scripts/legacy/README.md](../scripts/legacy/README.md),
прежние handoff и переносы — [docs/history/README.md](../docs/history/README.md).

## Правила

- Папка результата = имя скрипта без `_qc`: `free_surface_normal_incidence_qc`
  → `validation/elastic/free_surface_normal_incidence/`. Для SBP-SAT — папка
  запуска `validation/sbp_sat/<метка>_<YYYYMMDD>/` с подпапками `oblique/`,
  `time_energy/`.
- Каждый новый запуск получает новое имя. `oblique_qc` и `time_energy_qc`
  отказываются писать в непустую папку без `--overwrite`; `run_forward`
  без `--overwrite` не заменяет существующий forward NPZ. Архивные
  `reference_v2/` и `sherlock_20261016/` не перезаписываются.
- Логи Slurm — в `results/<область>/logs/` (для события —
  `events/<YYYYMMDD_номер>/logs/`). Slurm сам папку не создаёт: `mkdir -p`
  перед `sbatch`. Логи сборки сайта — в `web_safod/cache/`, сам сайт — в
  `web_safod/site/`.
- Исключение: QC подготовки скоростных моделей лежит рядом с входами в
  `data/safod/velocity_models/<источник>/qc/`.
- Большие запуски на Sherlock можно вести в `$SCRATCH` или `$GROUP_SCRATCH`
  с той же раскладкой. Эти пространства очищаются, поэтому единственная
  сохраняемая копия должна лежать на постоянном хранилище.

## SBP-SAT

- [Локальный запуск Sherlock 2026-10-16](validation/sbp_sat/sherlock_20261016/validation_report.md)
- [Исходный эталон QC v2](validation/sbp_sat/reference_v2/validation_report.md)
- Методика и команды запуска — [docs/sbp_sat/validation.md](../docs/sbp_sat/validation.md);
  сверка запусков — [docs/status.md](../docs/status.md)

# SAFOD: порядок работы

Шаги, команды, входы и выходы SAFOD-части проекта. Научные соглашения
(2-D разрез, вертикальный датум, политика сравнения, выбор модели) здесь не
пересказываются: они в
[handoff 2026-08-14](../history/FWI_SAFOD_FULL_HANDOFF_2026-08-14.md),
§5–6, §9, §24–32. `docs/history/` хранится локально и не версионируется;
пути внутри handoff даны на дату документа.

## Две разные задачи

| | Моделирование событий | Архив SAFOD-deep |
| --- | --- | --- |
| Назначение | Несколько выбранных землетрясений: подготовка DAS, forward по начальной модели, сравнение | Инвентаризация непрерывных записей, каталог NCEDC, QC-рисунки всех записанных событий |
| Код | `scripts/safod/`, `src/safod/` | `scripts/safod_deep/`, `src/safod_deep/`, `config/safod_deep/roots.json` |
| Данные DAS | файлы событий в `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/` | корни из `roots.json` в `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/` |
| Результаты | `results/events/<EVENT_KEY>/`, `results/safod/` | `results/safod_deep/` |
| Дальше | синтетика, FWI | `web_safod/` |

«deep» в `safod_deep` — имя архива непрерывных записей глубокого волокна
SAFOD (`SAFOD-deep/`, файлы `SAFOD-Deep-...h5` и `SAFOD_Deep_Nano-...h5`), а
не глубина событий: в каталог попадают события любой глубины в радиусе
запроса (`build_event_catalog --max-radius-km`, по умолчанию 50 км). Режим
`deep_saf` в `run_forward.py` (старый синтетический источник у разлома; в
коде выбран `catalog_event`) к архиву не относится.

## Общие условия

- Команды запускаются из корня проекта: пути `results/` и `data/` в коде
  относительные.
- Окружения `fwi` и `geo`; состав, проверка и внешние пакеты — в
  [setup.md](../setup.md). `geo` нужно всему, что импортирует `DASutils`
  (собранный `pyDAS` из DAS-utilities, Python 3.9) напрямую или через
  `DAS_db.py` и `prepare_event`.
- Модули, импортирующие `scripts.safod.settings` (`prepare_event`,
  `run_forward`, `compare_event`, `selection.screen_events`,
  `velocity_models.zhang2009.qc_registration`,
  `velocity_models.zhang2009.extract_safod_section`), при импорте ищут HDF5
  активного события на `/oak` и без них падают с `FileNotFoundError`. Всё,
  что читает `/oak` или считает, запускается на вычислительном узле
  (`sh_dev` или `sbatch`).
- Активное событие задаёт `SAFOD_EVENT_KEY`. Без переменной молча берётся
  `DEFAULT_EVENT_KEY = "nc75379261"`, поэтому её указывают в каждой команде.
- Каталог, время в очаге и записи геофонов скрипты запрашивают у NCEDC (FDSN).

Сокращения в командах ниже:

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
```

`<nc_key>` — ключ реестра в `scripts/safod/settings.py` (например,
`nc75379261`); `<EVENT_KEY>` — имя папки события (`20260618_75379261`).
`GEO_XLSX` — `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx`,
одна физическая геометрия кабеля для всех событий.

## A. Моделирование события

Пути скриптов — относительно `scripts/safod/`.

| Этап | Скрипт | Окр. | Вход | Выход | Команда |
| --- | --- | --- | --- | --- | --- |
| 1. Поиск в каталоге | `selection/find_catalog_events.py` | fwi | NCEDC, с 2026-04-01 до момента запуска; `GEO_XLSX` | `results/safod/catalog_search/`: `safod_catalog_events_since_april_2026_ranked.csv`, `candidates_crossline_lt{1,2,3}km.csv`, `safod_real_2d_profile_geometry.csv` | `$FWI -m scripts.safod.selection.find_catalog_events` |
| 2. Покрытие записями DAS | `selection/check_catalog_coverage.py` | geo | `..._ranked.csv` этапа 1; заголовки HDF5 по рекурсивному glob, включая `/oak/stanford/groups/ettore88/data/SAFOD/**/*.h5` | `results/safod/catalog_search/`: `catalog_candidates_with_das_coverage.csv`, `..._coverage_only.csv`, `all_candidate_das_headers.csv` | `$GEO -m scripts.safod.selection.check_catalog_coverage` |
| 3. Screening по 2-D геометрии | `selection/screen_events.py` | geo | заголовки HDF5 в `SAFOD_events/`; NCEDC | `results/safod/event_screening.csv` | `$GEO -m scripts.safod.selection.screen_events` |
| 4. Регистрация события | `settings.py`, вручную | — | выбранное событие | записи в `EVENTS` и `aliases` | см. ниже |
| 5. Регистрация каналов (если менялась конфигурация интеррогатора) | `calibrate_channel_registration.py` | geo | HDF5 события, время в очаге, `GEO_XLSX` | `results/events/<EVENT_KEY>/real/channel_registration_qc/` | см. ниже |
| 6. Подготовка наблюдений | `prepare_event.py` | geo | `input_patterns`, `GEO_XLSX`, NCEDC, CSV из `channel_mapping_csv` | `results/events/<EVENT_KEY>/real/`: `real_das_event_window.npz`, `SAFOD_Phase2_projected_from_georef.csv`, `full_cable_qc_registration.csv`, PNG | `SAFOD_EVENT_KEY=<nc_key> $GEO -m scripts.safod.prepare_event` |
| 7. Входы начальных моделей (один раз) | `velocity_models/...` | fwi | `data/safod/velocity_models/*/raw` и др. | `data/safod/velocity_models/*/{processed,qc}` | раздел «Начальные модели» |
| 8. Forward | `run_forward.py` | fwi | `real/real_das_event_window.npz`, входы модели | `results/events/<EVENT_KEY>/forward/dc<θ>/<model>/`: `outputs_safod_initial_forward.npz`, PNG; без `--no-gif` ещё `snapshots_vz.npz` и GIF | `SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.run_forward --theta-deg <θ> --initial-model <model>` |
| 9. Сравнение с синтетикой | `compare_event.py` | fwi | `real/real_das_event_window.npz`, `real/SAFOD_Phase2_projected_from_georef.csv`, пакет этапа 8 с теми же θ и моделью | `results/events/<EVENT_KEY>/compare/dc<θ>/<model>/` | `SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.compare_event --theta-deg <θ> --initial-model <model>` |
| 10. DAS и скважинный геофон | `plot_das_geophone_two_physical_comparisons.py` | geo | заданы в коде: HDF5 события NC75336802, канал 1694, SF.MH029.01.GP1; NCEDC; `GEO_XLSX` | `--output-dir`, по умолчанию `results/events/20260401_75336802/compare/das_geophone_physical_pairs/` | `$GEO -m scripts.safod.plot_das_geophone_two_physical_comparisons --output-dir results/events/20260401_75336802/compare/<analysis>` |

У этапов 1–3 и 6 нет параметров командной строки: настройки заданы
константами в начале файлов. Этап 2 обходит весь каталог SAFOD на Oak,
включая архив SAFOD-deep, поэтому его запускают только в задаче Slurm.

### Регистрация события в settings.py

- Ключ реестра — `nc<номер NCEDC>`. `normalize_event_key()` принимает только
  значения из явного словаря `aliases` (для каждого события `"<номер>"` и
  `"nc<номер>"`), поэтому новое событие вносится в два места: `EVENTS` и
  `aliases`. Иначе — `ValueError: Unknown SAFOD event key`.
- Поля записи (по существующим): `event_id` (`"NC<номер>"`),
  `ncedc_event_id` (int), `origin_time` (ISO, UTC), `magnitude`,
  `magnitude_type`, `depth_km`, `event_tag`, `input_patterns`,
  `channel_mapping_csv`.
  - `input_patterns` — пути или glob-шаблоны HDF5; при импорте `settings`
    каждый должен найтись.
  - `channel_mapping_csv` — `None`, если строки HDF5 совпадают с номерами
    каналов эталонной геометрии (так у апрельских событий); иначе путь
    относительно `results/events/<EVENT_KEY>/real/`, как у `nc75379261`:
    `channel_registration_qc/mapped_downleg_model_geometry.csv`.
  - `event_tag` обязателен (его читает `settings.py`), но больше нигде не
    используется; во всех записях он равен `EVENT_RESULT_KEY`.
- `EVENT_RESULT_KEY = <YYYYMMDD>_<номер>` вычисляется из `origin_time` и
  `event_id` без префикса `NC` и задаёт папку `results/events/<EVENT_KEY>/`:
  `nc75379261` → `20260618_75379261`.
- Сейчас в реестре: `nc75336802` (`20260401_75336802`), `nc75379261`
  (`20260618_75379261`, `DEFAULT_EVENT_KEY`), `nc75336682`
  (`20260401_75336682`, подготовлен только `real/`). Какие события подходят
  для 2-D сравнения — handoff §24, §28–30.
- `prepare_event` читает HDF5 (`DASutils.readFile_HDF`). Событие 2026-10-03
  записано в Sintela protobuf и в реестр не внесено (см. «Эксперименты»).

### Регистрация каналов

Нужна, когда строки HDF5 нового события не совпадают с номерами каналов
эталонной геометрии. Скрипт не импортирует `settings`, поэтому его можно
запустить до внесения события в реестр; `<EVENT_KEY>` для `--out-dir`
вычисляется по правилу выше.

```bash
$GEO -m scripts.safod.calibrate_channel_registration \
    --h5 <HDF5 события> \
    --origin-time <origin_time, UTC> \
    --reference-xlsx /home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
    --out-dir results/events/<EVENT_KEY>/real/channel_registration_qc
```

В `--out-dir` пишутся `channel_registration.json`,
`channel_registration_qc.png`, `turnaround_score.csv`,
`mapped_full_dual_pass_geometry.csv`, `mapped_downleg_geometry.csv` и
`mapped_downleg_model_geometry.csv`; последний указывается в
`channel_mapping_csv`. Остальные параметры (`--fmin`, `--fmax`, окна,
`--target-fs` и др.) — в `--help`. Для архива SAFOD-deep та же задача решается
отдельно: `plot_catalog_events --prepare-geometry` (часть B).

### run_forward и compare_event

- У `--initial-model` разные значения по умолчанию: `run_forward` —
  `bill_logs`, `compare_event` — `hybrid_zhang2009_boness2006_bill_logs`
  (угол по умолчанию у обоих `DEFAULT_THETA_DEG = 35.0`). `compare_event`
  ищет `forward/dc<θ>/<model>/outputs_safod_initial_forward.npz`: если пакета
  нет, он останавливается с `Synthetic package not found`, если модель в
  пакете другая — с `ValueError`. Поэтому `--theta-deg` и `--initial-model`
  всегда передаются обоим явно и одинаково. Выбор модели и угла — научное
  решение (handoff §31–32); этот документ его не делает.
- Допустимые модели (`AVAILABLE_INITIAL_MODELS` в
  `src/safod/models/factory.py`): `smooth_prior`, `bill_logs`, `zhang2009`,
  `hybrid_zhang2009_bill_logs`, `zhang2009_boness2006`,
  `hybrid_zhang2009_boness2006_bill_logs`,
  `hybrid_zhang2009_boness2006_bill_logs_smooth`.
- `<θ>` — целое число градусов, `0 <= θ < 90`: папку `dc<θ>` из трёх цифр
  (`dc035`) строит `theta_dir_name()`, нецелый угол даёт ошибку. Все
  сохранённые расчёты сейчас лежат в `dc035`.
- `run_forward` без `--overwrite` не перезаписывает существующий пакет;
  `--no-gif` отключает снимки поля и GIF; `--nt` — число шагов (по умолчанию
  12000).
- `run_forward` — полный расчёт, его запускают в задаче Slurm. Готового
  `.sbatch` в проекте нет; время и память не записаны.
- Скрипты не пишут лог сами. Для `20260618_75379261` вывод сохранён вручную в
  `results/events/20260618_75379261/logs/` под именами
  `forward_<nc_key>_<model>.log` и `compare_<nc_key>_<model>.log`.

### Сравнения наблюдений

Анализы без синтетики (DAS и скважинный геофон) кладутся в
`results/events/<EVENT_KEY>/compare/<analysis>/`; правила имён — в
[results/README.md](../../results/README.md).

- `plot_das_geophone_two_physical_comparisons.py` настроен на одно событие:
  NC75336802, канал DAS 1694, SF.MH029.01.GP1, файл HDF5 и полоса
  (`FMIN_HZ`, `FMAX_HZ`, сейчас 5–30 Гц) заданы константами в коде; из
  командной строки меняется только `--output-dir`. Методика и вариант
  1–20 Гц описаны в handoff §26.
- `results/events/20260618_75379261/compare/das_geophone_common_workflow/`
  создаёт ноутбук `notebooks/NC75379261_DAS_geophone.ipynb`.
- Каким кодом создана `results/events/20260401_75336802/compare/das_geophone_5_30Hz_usgs_time/`,
  в текущем проекте не найдено.

## B. Архив SAFOD-deep и web_safod

Пути скриптов — относительно `scripts/safod_deep/`, папки результатов —
относительно `results/safod_deep/`.

| Этап | Скрипт | Окр. | Вход | Выход | Команда |
| --- | --- | --- | --- | --- | --- |
| 1. Инвентаризация Nano-записей | `scan_nano_manifest.py` | fwi | `--root`, по умолчанию `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/2026-06-Nano`; метка, приоритет и примечание по умолчанию повторяют запись `2026-06_nano` в `roots.json` | `catalog/recording_manifest_nano.csv` | `$FWI -m scripts.safod_deep.scan_nano_manifest` |
| 2. Каталог событий и покрытие | `build_event_catalog.py` | fwi | `config/safod_deep/roots.json`, `GEO_XLSX`, NCEDC | `catalog/`: `recording_manifest.csv`, `ncedc_events_raw.csv`, `catalog_all.csv`, `catalog_recorded.csv`, `catalog_2d_direct_source_candidates.csv`, `event_windows_for_plotting.csv`, `CATALOG_NOTES.md`, PNG | см. ниже |
| 3. Регистрация геометрии по конфигурациям | `plot_catalog_events.py --prepare-geometry` | geo | `catalog/catalog_recorded.csv`, `catalog/recording_manifest.csv`, `GEO_XLSX` | `geometry/acquisition_registration.json`, `geometry/acquisition_registration_qc.png` | `$GEO -m scripts.safod_deep.plot_catalog_events --prepare-geometry` |
| 4. QC-рисунки событий | `plot_catalog_events_array.sbatch` | geo | то же и реестр этапа 3 | `event_qc/`: PNG, `metadata/*.json`, `status_shard_<i>_of_008.csv`; логи Slurm — `logs/` | `mkdir -p results/safod_deep/logs && sbatch scripts/safod_deep/plot_catalog_events_array.sbatch` |
| 5. Локальная HTML-галерея | `plot_catalog_events.py --build-index-only` | geo | `event_qc/` | `event_qc/index.html`, `event_qc/gallery_manifest.csv` | `$GEO -m scripts.safod_deep.plot_catalog_events --build-index-only` |
| 6. Веб-просмотр | `web_safod/` | fwi | `catalog/catalog_recorded.csv`, `event_qc/`, `event_qc/metadata/` (пути в `web_safod/config/explorer.json`) | `web_safod/site/` | [web_safod/README.md](../../web_safod/README.md) |

Новая папка записей на Oak добавляется в `config/safod_deep/roots.json`
(`label`, `path`, `priority`, `notes`).

### Этап 2: build_event_catalog

```bash
$FWI -m scripts.safod_deep.build_event_catalog \
    --roots-config config/safod_deep/roots.json \
    --geometry /home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
    [--wellhead-elevation-m <м>] [--reuse-manifest] [--reuse-events]
```

- Без `--reuse-manifest` скрипт заново сканирует заголовки HDF5 всех корней
  из `roots.json` и перезаписывает `catalog/recording_manifest.csv`; с флагом
  читает существующий файл. `--reuse-events` берёт `catalog/ncedc_events_raw.csv`
  вместо запроса к NCEDC.
- `scan_nano_manifest` пишет отдельный файл, который не читают ни
  `build_event_catalog`, ни `plot_catalog_events`. Текущий
  `recording_manifest.csv` содержит все 95 903 строки
  `recording_manifest_pre_nano.csv` и 5 366 из 5 368 строк
  `recording_manifest_nano.csv`. Как выполнено объединение, в коде не
  записано — уточнить у автора, прежде чем пересобирать манифест.
- `--wellhead-elevation-m` нужен, если в геометрии есть TVD, но нет
  абсолютной отметки; без него 3-D расстояния помечаются неизвестными
  (заголовок скрипта: «Do not guess wellhead elevation»). В примере в
  заголовке стоит `684.0`, в `settings.py` —
  `SAFOD_SURFACE_ELEVATION_M = 660.46`, а в текущем `catalog_recorded.csv`
  `closest_cable_elevation_m + closest_cable_tvd_m` = 660.5 м во всех 291
  строках. Какое значение брать — уточнить у автора.

### Этапы 3–5: plot_catalog_events

- Этап 3 повторяют после изменения манифеста или каталога: если в каталоге
  есть конфигурация интеррогатора без проверенной регистрации, этап 4
  останавливается с `Unvalidated configurations ... Run --prepare-geometry`.
- `.sbatch` отправляют из корня проекта: пути `--output` и `--error`
  относительные. Он вызывает
  `plot_catalog_events --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID --overwrite`;
  ресурсы и массив заданий указаны в самом файле. При `--num-shards` больше 1
  галерея не строится, для неё есть этап 5.
- Полоса, окна, P/S-направляющие и отбор событий задаются флагами
  (`--fmin`, `--fmax`, `--phase-guide-mode`, `--phase-model`, `--event-id`,
  `--geometry-class`, `--limit`, `--output-dir` и др., см. `--help`).
  P/S-направляющие по умолчанию считаются по модели
  `hybrid_zhang2009_boness2006_bill_logs_smooth` (`--phase-model`) через
  EikoHypoTomo (`/home/groups/ettore88/alina/packages/EikoHypoTomo`).
- Для папок `event_qc_1_20Hz/`, `event_qc_3_15Hz/`, `event_qc_5_30Hz/` и
  `event_qc_adaptive/` команды не сохранены; параметры обработки каждого
  события записаны в их `metadata/*.json`. `web_safod` по умолчанию читает
  только `event_qc/`.
- Каким кодом созданы `geometry/nano_registration.json`,
  `geometry/nano_registration_qc.png` и `geometry/nano_mapping.csv`, в
  текущем проекте не найдено.
- `web_safod/scripts/fetch_focal_mechanisms.py` пишет
  `catalog/focal_mechanisms.json`. Сборка и публикация сайта описаны в
  [web_safod/README.md](../../web_safod/README.md).

## Начальные модели

Модели строятся в `src/safod/models/`: `factory.build_initial_model(model_name=...)`
выбирает построитель по имени из `AVAILABLE_INITIAL_MODELS`, по одному модулю
на модель. Модули читают готовые файлы из `data/safod/velocity_models/`
(пути по умолчанию — в `factory.py`, `zhang2009.py`, `boness_zoback2006.py`).
Модели нужны `run_forward` и P/S-направляющим в `plot_catalog_events`.

Входы для моделей готовятся один раз скриптами `scripts/safod/velocity_models/`
и записываются в `data/`, а не в `results/`. `data/safod/velocity_models/`
исключён из Git.

| Набор | Скрипт (`scripts/safod/velocity_models/`) | Окр. | Выход (`data/safod/velocity_models/`) |
| --- | --- | --- | --- |
| Zhang et al. 2009, нативная сетка | `zhang2009/parse_model.py` | fwi | `zhang_thurber_bedrosian_2009/processed/zhang2009_native_grid.npz`, `.../zhang2009_native_nodes.csv`, `.../qc/` |
| Zhang 2009, проверка горизонтальной привязки | `zhang2009/qc_registration.py` | fwi | `zhang_thurber_bedrosian_2009/qc/zhang2009_registration_summary.txt`, `.../zhang2009_local_geographic_grid.png` |
| Zhang 2009, 2-D разрез SAFOD | `zhang2009/extract_safod_section.py` | fwi | `zhang_thurber_bedrosian_2009/processed/zhang2009_safod_section_2d.npz`, `.../zhang2009_safod_section_coordinates.csv`, `.../qc/` |
| Ellsworth & Malin 2011, Fig. 3a | `digitize_ellsworth_malin_fig3a.py` | fwi | `ellsworth_malin_2011/fig3a_digitized.csv`, QC PNG |
| Boness & Zoback 2006 | скрипта в проекте нет, `raw/` пуст | — | готовый `boness_zoback_2006/processed/boness_zoback2006_fig3_4_vp_vs_digitized.csv` |

```bash
Z=data/safod/velocity_models/zhang_thurber_bedrosian_2009
$FWI -m scripts.safod.velocity_models.zhang2009.parse_model --raw-dir $Z/raw --output-dir $Z
$FWI -m scripts.safod.velocity_models.zhang2009.qc_registration
SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.velocity_models.zhang2009.extract_safod_section

$FWI -m scripts.safod.velocity_models.digitize_ellsworth_malin_fig3a \
    --image <PNG рисунка 3a> \
    --csv data/safod/velocity_models/ellsworth_malin_2011/fig3a_digitized.csv \
    --qc data/safod/velocity_models/ellsworth_malin_2011/qc/fig3a_digitization_qc.png
```

- `qc_registration` и `extract_safod_section` импортируют `settings`, им
  нужен `/oak`. `extract_safod_section` по умолчанию берёт `--geometry-csv`
  активного события,
  `results/events/<EVENT_KEY>/real/SAFOD_Phase2_projected_from_georef.csv`,
  поэтому запускается после `prepare_event` этого события.
- Исходного PNG для оцифровки Fig. 3a в проекте нет.
- Источники данных описаны в
  `data/safod/velocity_models/zhang_thurber_bedrosian_2009/README.md` и
  `data/safod/velocity_models/ellsworth_malin_2011/README.md` (локально, вне
  Git); научное содержание моделей — в handoff §7–19.

## Эксперименты и устаревший код

- `scripts/safod/experiments/` — разовые диагностики вне основного порядка
  работы:
  - `diagnose_geometry.py` проверяет, монотонна ли TVD в `GEO_XLSX` при
    сортировке по номеру канала, и печатает результат:
    `$FWI -m scripts.safod.experiments.diagnose_geometry`.
  - `oct03_2026/` — событие 2026-10-03 (nc75443276), Sintela protobuf и
    MH030: [README](../../scripts/safod/experiments/oct03_2026/README.md).
- `scripts/legacy/safod_deep/plot_two_selected_events_pretty.py` — рисунки
  двух выбранных событий архива. Не импортируется: в `plot_catalog_events.py`
  больше нет трёх нужных ему функций. Его сохранённый результат —
  `results/safod_deep/event_qc_pretty_two/`. Об устаревшем коде в целом —
  [scripts/legacy/README.md](../../scripts/legacy/README.md).

## Внешний потребитель: SAFOD_WaveSearch

Проект `/home/groups/ettore88/alina/SAFOD/SAFOD_WaveSearch` читает геометрию
по абсолютному пути
`/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project/results/events/20260618_75379261/real/SAFOD_Phase2_projected_from_georef.csv`
(проверено grep 2026-10-16): `scripts/wavesearch/run_feasibility_pipeline.sh`
(значение `GEOMETRY` по умолчанию), `configs/data/oak_safod_wave_search.yaml`,
`configs/data/fwi_boness_dev.yaml`, `configs/paths/sherlock.example.yaml`,
а также `README.md` и `docs/NEXT_STEPS.md`.

Поэтому `results/events/`, папку `20260618_75379261` и этот файл не
переименовывают и не переносят без одновременной правки SAFOD_WaveSearch.
`real/channel_registration_qc/channel_registration.json` того же события тоже
хранит абсолютные пути внутри проекта. `results/` исключён из Git, так что
файл существует только в рабочей копии на Sherlock.

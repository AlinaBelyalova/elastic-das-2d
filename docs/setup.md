# Окружение, данные и тесты

Как работать с проектом на Sherlock: окружения Python, локальные пакеты не из
PyPI, входные данные, зашитые в код внешние пути, запуск и тесты. Всё ниже
проверено 2026-10-16 в задании на вычислительном узле (импорт, `ls`, `grep`,
`importlib.metadata`); расчёты не запускались. Где что лежит в проекте —
[project_map.md](project_map.md), шаги SAFOD — [safod/workflow.md](safod/workflow.md).

## 1. Окружения

Два conda-окружения. Активировать их не нужно: достаточно вызывать
интерпретатор по полному пути (`<окружение>/bin/python`).

|  | `fwi` | `geo` |
|---|---|---|
| Путь | `/home/groups/ettore88/alina/.envs/fwi` | `/home/groups/ettore88/alina/.envs/geo` |
| Python | 3.11.13 | 3.9.23 |
| numpy | 2.0.1 | 1.26.4 |
| scipy | 1.16.0 | 1.13.1 |
| matplotlib | 3.10.8 | 3.9.4 |
| numba | 0.65.1 | 0.60.0 |
| pandas | 3.0.2 | 2.3.3 |
| h5py | 3.14.0 | 3.14.0 |
| obspy | 1.5.0 | 1.4.2 |
| pyproj | 3.6.1 | 3.6.1 |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 |
| openpyxl | 3.1.5 | 3.1.5 |
| Pillow | 12.1.1 | 11.3.0 |
| pytest | 9.0.2 | нет |
| pykonal | нет | 0.4.1 |
| ipykernel / jupyterlab / notebook | 7.2.0 / 4.5.5 / 7.5.4 | 6.30.1 / 4.4.6 / нет |

Метапакета `jupyter` нет ни в одном окружении, но команда `jupyter` есть в
обоих. В `fwi` также стоит devito (editable-установка); кодом проекта он не
импортируется, но из-за него вне корня проекта находится чужой пакет
`scripts` (раздел 5).

**`fwi`** — основное окружение: `src/`, `scripts/validation/`,
`scripts/examples/`, `scripts/benchmarks/`, `scripts.safod.run_forward`,
`scripts.safod.compare_event`, `web_safod/`, тесты (pytest есть только здесь).

**`geo`** нужен всему, что импортирует `DASutils`: тот импортирует `pyDAS`,
а `pyDAS` собран только под Python 3.9
(`DAS-utilities/build/pyDAS.cpython-39-x86_64-linux-gnu.so`). В `fwi` эти
модули падают с `ModuleNotFoundError: No module named 'pyDAS'`:

- уже при импорте: `scripts.safod.prepare_event`,
  `scripts.safod.selection.screen_events` (импортирует `prepare_event`),
  `scripts.safod.plot_das_geophone_two_physical_comparisons`,
  `scripts.safod_deep.plot_catalog_events`, все
  `scripts.safod.experiments.oct03_2026.*`;
- во время работы (импорт в `fwi` проходит):
  `scripts.safod.calibrate_channel_registration` (`DASutils` при чтении
  HDF5), `scripts.safod.selection.check_catalog_coverage` (`DAS_db.py`
  импортирует `DASutils`);
- ноутбук `notebooks/NC75379261_DAS_geophone.ipynb`.

P/S-годографы в `scripts.safod_deep.plot_catalog_events` считаются через
EikoHypoTomo, которому нужен `pykonal` (есть только в `geo`).

В `geo` импортируются все модули `src/`, `scripts/` (кроме `scripts/legacy/`)
и `web_safod/`, в `fwi` — все, кроме перечисленных выше. Остальным скриптам
`pyDAS` не нужен, для них берётся `fwi`. Импорт не значит, что скрипт в этом
окружении запускали: для `scripts.safod_deep.scan_nano_manifest`,
`scripts.safod_deep.build_event_catalog`,
`scripts.safod.selection.find_catalog_events`,
`scripts.safod.experiments.diagnose_geometry` и
`scripts.safod.velocity_models.*` окружение прежних запусков нигде не
записано.

### Ноутбуки и ядра Jupyter

| Ноутбук | `metadata.kernelspec` | `language_info.version` | Окружение |
|---|---|---|---|
| `notebooks/final_demo.ipynb` | `name: python3`, `display_name: Python 3` | 3.11.13 | `fwi` |
| `notebooks/NC75379261_DAS_geophone.ipynb` | `name: python3`, `display_name: geo` | 3.9.23 | `geo` |

Jupyter выбирает ядро по `name`, а `python3` — встроенное ядро того
окружения, из которого запущен сам Jupyter. Поэтому `final_demo.ipynb`
открывать в Jupyter из `fwi` (или с ядром `Python (fwi)`),
`NC75379261_DAS_geophone.ipynb` — в Jupyter из `geo`
(`/home/groups/ettore88/alina/.envs/geo/bin/jupyter lab`). Оба ноутбука сами
находят корень проекта, если открыты из корня или из `notebooks/`.

Пользовательские ядра лежат в `~/.local/share/jupyter/kernels/`, список —
`jupyter kernelspec list`. На 2026-10-16 ядро `geo` там указывает на
несуществующий `~/miniconda3/envs/geo/bin/python`, а `geo_cuda/kernel.json`
не читается (невалидный JSON); `fwi` указывает на
`/home/groups/ettore88/alina/.envs/fwi/bin/python`.

Jupyter — сервер, поэтому запускать его только в задании на вычислительном
узле (раздел 5), не на логин-узле.

## 2. Локальные пакеты не из PyPI

Оба пакета не установлены ни в одно окружение (`import DASutils`,
`import pyVector` вне проекта падают). Скрипты сами добавляют их каталоги в
`sys.path`, и путь зашит в каждом скрипте отдельно (раздел 4).

| Пакет | Что используется | Как код его находит |
|---|---|---|
| DAS-utilities, `/home/groups/ettore88/alina/packages/DAS-utilities` | `python/DASutils.py`, `python/DAS_db.py`; `build/pyDAS.cpython-39-x86_64-linux-gnu.so` (сборка CMake, только Python 3.9, т.е. `geo`) | скрипт вставляет `build/` и `python/` в начало `sys.path` и добавляет `build/` в `LD_LIBRARY_PATH`; `DAS_db.py` загружается по пути файла (`importlib.util.spec_from_file_location`): `settings.DAS_DB_PY` в `prepare_event` и `selection.screen_events`, своя константа в `selection.check_catalog_coverage`; у `calibrate_channel_registration` корень задаётся `--utilities-root` (по умолчанию тот же путь) |
| EikoHypoTomo, `/home/groups/ettore88/alina/packages/EikoHypoTomo` | `python/EikonalTomography2DMod.py` (импортирует `pykonal`), `external/python-solver/GenericSolver/python/pyVector.py` | только `scripts.safod_deep.plot_catalog_events`: `_load_eikonal_modules()` при первом eikonal-расчёте добавляет оба каталога в `sys.path` и, если `DATAPATH` не задан, ставит его в `$TMPDIR` (или `/tmp`); ошибки eikonal-расчёта перехватываются (`phase-guide eikonal failed ...`) |

Собственные зависимости пакетов описаны в их `environment.yml` (у
DAS-utilities также `requirements.txt`).

## 3. Входные данные

| Вход | Где | Кто читает |
|---|---|---|
| Записи DAS по событиям (HDF5; для эксперимента 2026-10-03 — файлы `.pb`) | `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events` | `scripts/safod/settings.py`: `SAFOD_EVENT_DATA_ROOT` и `EVENTS[<ключ>]["input_patterns"]` (ключи `nc75336802`, `nc75379261` — по умолчанию, `nc75336682`; выбор — `SAFOD_EVENT_KEY`); свои копии пути — в `scripts/safod/` `selection/check_catalog_coverage.py`, `selection/screen_events.py`, `experiments/oct03_2026/*`, `plot_das_geophone_two_physical_comparisons.py` |
| Непрерывный архив SAFOD-deep | 8 корней записи в `config/safod_deep/roots.json` (все под `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/`) | `scripts.safod_deep.build_event_catalog --roots-config` (параметр обязателен); `scripts.safod_deep.scan_nano_manifest --root` (по умолчанию корень `2026-06-Nano`) |
| Таблица георефер. каналов кабеля | `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx` (вне проекта и вне git) | `settings.GEO_XLSX` (`prepare_event`); свои копии пути в 6 скриптах (раздел 4); `geometry_xlsx` в `web_safod/config/*.json`; `build_event_catalog --geometry` (обязателен, без значения по умолчанию) |
| Перерегистрация каналов события `nc75379261` | `results/events/20260618_75379261/real/channel_registration_qc/mapped_downleg_model_geometry.csv` | `settings.CHANNEL_MAPPING_CSV` (`prepare_event`); пишет его `scripts.safod.calibrate_channel_registration` |
| Скоростные модели | `data/safod/velocity_models/<модель>/` | `src/safod/models/` (таблица ниже) |

Скоростные модели (пути внутри `data/safod/velocity_models/`; код читает их
по относительным путям, поэтому запуск только из корня):

| Модель | `raw/` | Что лежит ещё | Кто читает | Чем получено |
|---|---|---|---|---|
| `zhang_thurber_bedrosian_2009/` | `MOD.head`, `inversion_grid.dat`, `Vp_model.dat`, `Vs_model.dat`, `Vpvs_model.dat` (переданы C. Thurber, см. `README.md` модели) | `processed/` (`.npz`, `.csv`), `qc/` (`.png`, `.txt`) | `src/safod/models/zhang2009.py` (`processed/zhang2009_safod_section_2d.npz`) | `scripts.safod.velocity_models.zhang2009.{parse_model, extract_safod_section, qc_registration}` |
| `boness_zoback_2006/` | пуст | `processed/boness_zoback2006_fig3_4_vp_vs_digitized.csv`, `qc/boness_zoback2006_fig3_4_digitization_qc.png` | `src/safod/models/boness_zoback2006.py` | скрипта оцифровки в проекте нет — уточнить у автора |
| `ellsworth_malin_2011/` | нет (без `raw/processed/qc`) | `fig3a_digitized.csv`, `README.md` | `src/safod/models/factory.py` (`DEFAULT_BILL_LOGS_CSV`) | `scripts.safod.velocity_models.digitize_ellsworth_malin_fig3a --image <рисунок>`; исходного рисунка в проекте нет |

Что не попадает в git (`.gitignore`, `web_safod/.gitignore`):

- `data/safod/velocity_models/` целиком: в клоне репозитория скоростных
  моделей нет, их переносят отдельно;
- `results/*`, кроме `results/README.md`, — включая CSV перерегистрации
  каналов из таблицы выше;
- везде `*.npz`, `*.npy`, `*.gif`, `*.mp4`;
- `docs/history/`;
- в `web_safod/`: `config/paths.local.json`, `cache/*` и `site/*` (кроме
  их `README.md`), `reference/san_andreas_fault.geojson`.

`config/safod_deep/roots.json` в git есть.

## 4. Внешние пути в коде

Собрано командой (без `scripts/legacy/`; 64 совпадения):

```bash
grep -rnE '"/oak/|"/home/groups/|/oak/stanford|/home/groups/ettore88' \
  --include=*.py --include=*.sbatch --include=*.json --include=*.sh \
  scripts src web_safod/config web_safod/scripts config | grep -v '^scripts/legacy'
```

| Корень | Что | Где (файл:строка) |
|---|---|---|
| `/home/groups/ettore88/alina/packages/DAS-utilities` (и `build/`, `python/`, `python/DAS_db.py`) | DAS-utilities | `scripts/safod/settings.py:59`, `scripts/safod/prepare_event.py:23,24`, `scripts/safod/calibrate_channel_registration.py:42`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:34`, `scripts/safod/selection/check_catalog_coverage.py:19,20,37`, `scripts/safod_deep/plot_catalog_events.py:38`, `scripts/safod/experiments/oct03_2026/`: `compare_oct03_current_turnaround.py:54`, `compare_oct03_das_mh030_simple.py:58`, `diagnose_oct03_mh030_channels.py:75`, `find_current_mh030_das_channel.py:52`, `plot_oct03_das_geophone_like_scec.py:75`, `plot_oct03_m08_das_mh030_pb_v3.py:70`, `plot_oct03_m08_das_mh030_pb_v5.py:71` |
| `/home/groups/ettore88/alina/packages/EikoHypoTomo` | EikoHypoTomo | `scripts/safod_deep/plot_catalog_events.py:84` |
| `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx` | таблица каналов | `scripts/safod/settings.py:32`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:69`, `scripts/safod/selection/find_catalog_events.py:18`, `scripts/safod/experiments/diagnose_geometry.py:19`, `scripts/safod/experiments/oct03_2026/plot_oct03_m08_das_mh030_pb_v3.py:108`, `scripts/safod/experiments/oct03_2026/plot_oct03_m08_das_mh030_pb_v5.py:109`, `scripts/safod_deep/plot_catalog_events.py:57`, `web_safod/config/explorer.json:5`, `web_safod/config/paths.example.json:6`, `web_safod/config/paths.local.json:6` |
| `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events` | записи событий | `scripts/safod/settings.py:27`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:64`, `scripts/safod/selection/check_catalog_coverage.py:42`, `scripts/safod/selection/screen_events.py:32`, `scripts/safod/experiments/oct03_2026/`: `compare_oct03_current_turnaround.py:81,85`, `compare_oct03_das_mh030_simple.py:87,91`, `diagnose_oct03_mh030_channels.py:106,110`, `find_current_mh030_das_channel.py:71,75`, `plot_oct03_das_geophone_like_scec.py:107,111`, `plot_oct03_m08_das_mh030_pb_v3.py:98,102`, `plot_oct03_m08_das_mh030_pb_v5.py:99,103` |
| `/oak/stanford/groups/ettore88/data/SAFOD` (`ActiveJune2026/**/*.h5`, `**/*.h5`) | расширенный поиск HDF5 | `scripts/safod/selection/check_catalog_coverage.py:43,44` |
| `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep` | архив SAFOD-deep | `config/safod_deep/roots.json:5,11,17,23,29,35,41,47`, `scripts/safod_deep/scan_nano_manifest.py:136` |
| `/home/groups/ettore88/alina/.envs/geo/bin/python`, `.../python3` | интерпретатор `geo` | `scripts/safod/settings.py:68`, `scripts/safod_deep/plot_catalog_events_array.sbatch:15` |
| `/home/groups/ettore88/alina/.envs/fwi/bin/python` | интерпретатор `fwi` | `web_safod/scripts/build_site.sbatch:13` |
| `/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project` | корень проекта (`cd` в задании) | `scripts/safod_deep/plot_catalog_events_array.sbatch:14`, `web_safod/scripts/build_site.sbatch:12` |
| `/home/groups/ettore88/alina/classes/geophys245/safod-event-explorer-pages` | checkout GitHub Pages (`pages_checkout`) | `web_safod/config/explorer.json:8` |
| `/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv` | старая геометрия | `scripts/safod/run_forward.py:1680` (комментарий), `:1686` (ветка `deep_saf`; `source_mode` в коде зафиксирован как `"catalog_event"`) |

Вне этого поиска: код ячеек `notebooks/NC75379261_DAS_geophone.ipynb`
содержит пути DAS-utilities и `SAFOD_events`.

Константы путей в `scripts/safod/settings.py`, которые другие модули не
импортируют: `DAS_UTILITIES_BUILD` (:62), `DESAMPLE_DAS_PY` (:65) и
`GEO_PYTHON` (:67) не используются нигде; `DAS_UTILITIES_ROOT` (:58) и
`DAS_UTILITIES_PYTHON` (:63) нужны только внутри `settings.py`, чтобы
собрать `DAS_DB_PY`; `SAFOD_EVENT_DATA_ROOT` (:26) — только для `EVENTS`.
Из путей `settings.py` другие модули импортируют только `DAS_DB_PY`
(`prepare_event`, `selection.screen_events`) и `GEO_XLSX` (`prepare_event`).
Собрать пути в одном конфиге — возможный следующий шаг.

## 5. Как запускать

- Из корня проекта, модулем: `python -m scripts.<...>`. Пути к `results/` и
  `data/` в коде относительные. Кроме того, вне корня пакет `scripts` в `fwi`
  находится в devito: из `/tmp`
  `/home/groups/ettore88/alina/.envs/fwi/bin/python -c "import importlib.util; print(importlib.util.find_spec('scripts'))"`
  показывает `/home/groups/ettore88/alina/packages/devito/scripts`.
- Только на вычислительном узле: `sh_dev` (по умолчанию раздел `dev`, 1 ядро,
  4 ГБ, 1 ч; опции `-c`, `-m`, `-t`) или `sbatch` с явным `--time`. Разделы
  и лимиты — `sh_part`. На логин-узле — только редактирование и просмотр.
- Для коротких проверок ограничить потоки:
  `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  В `scripts/safod_deep/plot_catalog_events_array.sbatch` `OMP_NUM_THREADS`
  и `NUMBA_NUM_THREADS` равны `$SLURM_CPUS_PER_TASK`.
- Импорт `scripts/safod/settings.py` не обращается к Oak. Он выбирает событие
  `SAFOD_EVENT_KEY` (по умолчанию `nc75379261`); неизвестный ключ вызывает
  `ValueError`. `prepare_event.main()` вызывает `get_selected_files()` до
  запроса каталога и чтения данных: отсутствие HDF5 по-прежнему вызывает
  `FileNotFoundError`. Независимые тесты модели могут работать без `/oak`.
- `sbatch` — только из корня: `#SBATCH --output/--error` в файлах проекта
  относительные, Slurm отсчитывает их от каталога отправки и каталог для
  логов не создаёт:

  ```bash
  cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
  mkdir -p results/<область>/logs
  sbatch <путь/к/файлу>.sbatch
  ```

  | Файл | Окружение | Логи |
  |---|---|---|
  | `scripts/safod_deep/plot_catalog_events_array.sbatch` | `geo` | `results/safod_deep/logs/` |
  | `web_safod/scripts/build_site.sbatch` | `fwi` | `web_safod/cache/` |

  Оба файла делают `cd` в абсолютный корень проекта (раздел 4). Куда класть
  логи новых заданий — [results/README.md](../results/README.md); когда и с
  какими параметрами запускать — [safod/workflow.md](safod/workflow.md) и
  [web_safod/README.md](../web_safod/README.md).

## 6. Тесты

Из корня, в `fwi`, на вычислительном узле:

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/groups/ettore88/alina/.envs/fwi/bin/python -m pytest -q
```

[`pytest.ini`](../pytest.ini): `testpaths = tests web_safod/tests`,
`pythonpath = .`, в `norecursedirs` — `results`, `data`, `notebooks`,
`web_safod/site`, `web_safod/cache`. Весь набор занимает несколько секунд
(замер 2026-10-16).

| Файл | Что проверяет | Условия |
|---|---|---|
| `tests/test_project_layout.py` | текст проекта: нет старых имён модулей и путей; цели `python -m` в `.md`, `.sbatch`, `.sh`, `.ipynb` находятся; относительные ссылки в `.md` не битые и не выходят из проекта | код проекта не исполняет |
| `tests/test_safod_smooth_prior_explicit_bounds.py` | явные границы сетки в `src.safod.models.smooth_prior.build_smooth_prior_model` | синтетическая геометрия, `/oak` не нужен |
| `tests/test_safod_settings.py` | импорт настроек без поиска данных; строгая проверка файлов при подготовке события | `/oak` не нужен |
| `web_safod/tests/test_web_safod.py` | builder `web_safod` ([web_safod/README.md](../web_safod/README.md)) | — |

Голый `pytest` работает, пока подхватывается `pytest.ini` (запуск из корня или
с путями внутри проекта). Если ini не используется (например, `-c` с другим
файлом), корня нет в `sys.path`, `scripts` находится в devito и сбор падает с
`No module named 'scripts.safod'`. Поэтому — `python -m pytest` из корня.

Тесты `web_safod` отдельно:

```bash
/home/groups/ettore88/alina/.envs/fwi/bin/python -m unittest web_safod.tests.test_web_safod
node --test web_safod/tests/test_app.cjs
```

Node.js в `fwi` нет. Проверить, есть ли `node` в оболочке:
`command -v node && node --version` (в оболочке агента 2026-10-16 находился
`/home/groups/ettore88/alina/.envs/cursor-node/bin/node`, v20.20.2; в `geo`
есть `bin/node`, v22.6.0). Модули Sherlock: `ml spider nodejs` (на
2026-10-16 — 20.20.0, 24.13.0, 25.3.0, загрузка `ml devel nodejs/20.20.0`).
Если функции `ml` в оболочке нет, — `bash -lc 'ml spider nodejs'`.
Проверки интерфейса описаны в [web_safod/README.md](../web_safod/README.md).

## 7. `requirements.txt`

[`requirements.txt`](../requirements.txt) — минимальный список для `fwi`, не
lock-файл и не описание `geo`:

- `numpy`, `scipy`, `matplotlib`, `numba` (с нижними границами версий) —
  решатели и графика; `jupyter`, `ipykernel` — ноутбуки;
- `pandas`, `h5py`, `obspy`, `pyproj`, `python-dateutil`, `openpyxl`,
  `Pillow`, `pytest` — без версий: данные SAFOD (HDF5, FDSN, координаты,
  Excel через `pandas.read_excel`), каталоги, `web_safod`, тесты. Версии в
  `fwi` на 2026-10-16 записаны в комментарии файла;
- DAS-utilities и EikoHypoTomo с PyPI не ставятся (раздел 2); `pykonal` для
  EikoHypoTomo есть только в `geo`.

Расхождения с `fwi`: метапакета `jupyter` в `fwi` нет (есть `jupyterlab`,
`notebook`, `ipykernel`); devito в `fwi` есть, но в файле не указан и кодом
проекта не импортируется. `web_safod/requirements.txt` — подмножество для
сайта (`pandas`, `openpyxl`, `Pillow`).

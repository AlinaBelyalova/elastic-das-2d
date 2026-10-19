# scripts/legacy — старый код

Скрипты сохранены только для истории: по ним видно, как получены старые
результаты. Код не поддерживается, под текущую структуру не обновляется и
исключён из проверок импорта; `tests/test_project_layout.py` тоже пропускает
`scripts/legacy/`. Новый код сюда не добавляется. Текущие скрипты —
в [docs/project_map.md](../../docs/project_map.md).

| Файл | Что делал | Состояние | Результат |
| --- | --- | --- | --- |
| `prepare_real_event_20260413.py` | Подготовка реальной DAS-записи события NC75343317 (2026-04-13, Md 1.53): два HDF5-файла с Oak, полоса 1–20 Гц, окно 0–15 с от origin time по каталогу, проекция каналов на 2-D геометрию | Заменён `scripts.safod.prepare_event` | `results/legacy/real_event_20260413_M153/` |
| `safod_deep/plot_two_selected_events_pretty.py` | Рисунки «для публикации» двух событий архива SAFOD-deep, 75343317 и 75371066: фильтр 5–30 Гц, по горизонтали — физический опорный канал | Сломан: не импортируется | `results/safod_deep/event_qc_pretty_two/` |

## prepare_real_event_20260413.py

- Появился в коммите `3b3f1bf` (2026-07-16) вместе с
  `scripts/prepare_real_event_20260401_75336802.py`. Функции этого скрипта
  (`read_das_db_headers`, `parse_beg_time_from_info`,
  `latlon_to_local_enu_m`, `build_channel_projection_mapping`,
  `project_event_to_model`, `main`) есть и во втором. Во втором добавлены
  `parse_catalog_time_to_datetime` и `fetch_event_from_ncedc`: координаты и
  магнитуда события запрашиваются в NCEDC, а здесь вписаны в код. Коммит
  `7f182aa` перенёс `prepare_real_event_20260401_75336802.py` в
  `scripts/safod/prepare_event.py`, а этот скрипт — сюда.
- `scripts.safod.prepare_event` с тех пор обобщён: событие берётся из `EVENTS`
  в `scripts/safod/settings.py` (выбор через `SAFOD_EVENT_KEY`), результат
  пишется в `results/events/<YYYYMMDD_номер>/real/`. Порядок запуска описан в
  [docs/safod/workflow.md](../../docs/safod/workflow.md).
- События NC75343317 в `settings.EVENTS` нет. В screening событий
  (handoff 2026-08-14, §28) у него стоит `geometry: poor`.
- После переноса в `7f182aa` в коде изменён только путь вывода (проход
  `structure_20261017T004955Z`): было `results/real_event_20260413_M153`,
  стало `results/legacy/real_event_20260413_M153`.
- Если запускать: скрипт импортирует `DASutils` из сборки DAS-utilities
  (нужно окружение `geo`, см. [docs/setup.md](../../docs/setup.md)), а пути
  к HDF5 на Oak, к `DAS_db.py` и к таблицам геометрии зашиты в код. Папку
  вывода он создаёт уже при импорте, относительно текущего каталога, а при
  запуске перезаписывает сохранённые там файлы с теми же именами.
- Скрипт пишет 5 файлов в папке результата:
  `SAFOD_Phase2_projected_from_georef.csv`,
  `real_cable_projected_geometry_from_georef.png`,
  `real_das_event_0_15s_raw_channels.png`,
  `real_das_event_0_15s_trace_normalized.png`,
  `real_das_event_window_0_15s.npz`. Шестой файл,
  `channel_projection_mapping_qc.csv`, этот скрипт не пишет. Откуда он взялся,
  не установлено: имя не встречается ни в текущем коде, ни в истории git.

## safod_deep/plot_two_selected_events_pretty.py

- Скрипт импортирует из `scripts/safod_deep/plot_catalog_events.py` три
  функции, которых там больше нет: `infer_reference_locus_offset`,
  `load_reference_channel_limits`, `physical_reference_axis_and_crop`.
  Они есть в версии из коммита `a7ffd71` и удалены коммитом `26daccc`.
  Посмотреть их можно так:
  `git show a7ffd71:scripts/safod_deep/plot_catalog_events.py`.
- По умолчанию (`--output-dir`) результат пишется в
  `results/safod_deep/event_qc_pretty_two/`. В коде окна показа — 7.0 с
  для 75343317 и 2.5 с для 75371066. Им соответствуют файлы
  `event_75343317_pretty_5_30Hz_tmax_7s.png` и
  `event_75371066_pretty_5_30Hz_tmax_2.5s.png`.
- Все 7 PNG в папке изменены 2026-08-05 (по дате файлов). Остальные 5
  (без суффикса `tmax` или с другими значениями), судя по именам, получены
  при других настройках. Таких версий скрипта в git нет: там только версия
  из `a7ffd71`, совпадающая с текущей (кроме комментария в шапке).
- Комментарий в шапке (7.5 с и 3.5 с) с кодом не совпадает.

## Бывшая папка archive/

`archive/` вынесена из проекта в
`/home/groups/ettore88/alina/project_archives/Elastic_das_project/archive_code_20261017T020246Z/`.
Содержимое совпадает с `archive/` в коммите `a7ffd71`. Коммит `b61b7d5` убрал
эту папку из репозитория. Посмотреть файлы из git:

```bash
git show a7ffd71:archive/legacy_models/safod_log_constrained_model.py
git show a7ffd71:archive/legacy_scripts/safod/build_initial_model.py
```

| Файл | Что это |
| --- | --- |
| `legacy_models/safod_log_constrained_model.py` | Прежняя начальная модель SAFOD. Шапка файла: модель по оцифрованным каротажам Vp/Vs Ellsworth & Malin (2011), Fig. 3a; в шапке указано старое имя `src/safod_digitized_log_model.py` |
| `legacy_scripts/safod/build_initial_model.py` | Прежний скрипт построения начальной модели SAFOD (ранее `scripts/build_safod_initial_model.py`, затем `scripts/safod/build_initial_model.py`) |

Оба файла не запускаются: они импортируют `src.safod_builder`, которого больше
нет. Его код перенесён в `src/safod/models/smooth_prior.py` (коммит `a7ffd71`).

Полная таблица старых и новых путей — в `docs/history/README.md`. Папка
`docs/history/` есть только в локальной копии проекта, в git её нет.

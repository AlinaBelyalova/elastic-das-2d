# Эксперименты 2026-10-03 (nc75443276)

Разовые диагностики события NCEDC 75443276 (2026-10-03, M0.8 по заголовкам
v3/v5): DAS Sintela и скважинный прибор SF.MH030.01.GP1. Данные DAS — два
файла Sintela protobuf в `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/`:
`timeseriesScaler_2026-10-03_04.51.40_UTC_073863.pb` и
`timeseriesScaler_2026-10-03_04.52.40_UTC_073864.pb`.

Событие не внесено в `scripts/safod/settings.py`: в `EVENTS` только
`nc75336802`, `nc75379261` и `nc75336682`. Основной `prepare_event` читает
HDF5, поэтому [порядок работы SAFOD](../../../../docs/safod/workflow.md) к
этому событию не применяется. Скрипты не импортируют `scripts.safod.settings`
и друг друга; пути к данным, DAS-utilities и параметры заданы в каждом файле.

## Запуск

- Окружение `geo`: все скрипты импортируют `DASutils` (нужен `pyDAS`).
  Окружения описаны в [setup.md](../../../../docs/setup.md).
- Из корня проекта, на вычислительном узле: данные лежат на `/oak`.
- Нужен доступ к NCEDC: время в очаге, волновые формы MH030, метаданные.
- У всех скриптов есть argparse; полный список опций — `--help`.

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
$GEO -m scripts.safod.experiments.oct03_2026.<скрипт> [опции]
```

Примеры из заголовков файлов, приведённые к форме `-m` (в заголовке
`diagnose_oct03_mh030_channels.py` запуск указан по пути к файлу):

```bash
$GEO -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v3 --das-channel 1694
$GEO -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v5 --das-channel 1694
$GEO -m scripts.safod.experiments.oct03_2026.diagnose_oct03_mh030_channels \
    --fmin 5 --fmax 40 --pmin 0.17 --pmax 0.34
```

## Скрипты

Назначение взято из заголовков файлов; опции — из `--help`, значения по
умолчанию — из кода.

| Скрипт | Назначение | Опции (по умолчанию) | Выход в `compare/` |
| --- | --- | --- | --- |
| `find_current_mh030_das_channel.py` | Найти текущие каналы Sintela DAS, ближайшие к MH030, по времени первого вступления, без старой карты каналов OptaSense | `--ch-min` (1050), `--ch-max` (1450), `--output-dir` | `current_channel_timing/` |
| `diagnose_oct03_mh030_channels.py` | Сравнить MH030 GP1 с текущими каналами Sintela 1276–1281, по панели на канал; лаг в P-окне считается для диагностики и не применяется | `--fmin` (5), `--fmax` (40), `--tmin`, `--tmax`, `--pmin`, `--pmax`, `--output-dir` | `mh030_channel_diagnostic/` |
| `compare_oct03_current_turnaround.py` | Сравнение вокруг текущего разворота Sintela (около каналов 1278–1280) с точным временем NCEDC 04:52:31.320Z вместо округлённого 04:52:31.000Z из ноутбука Ettore | `--channel` (1279), `--output-dir` | `current_turnaround_comparison/` |
| `compare_oct03_das_mh030_simple.py` | Простое сравнение DAS / MH030 в стиле прежнего рисунка SCEC: две оси Y, без нормировки, подгонки амплитуд и сдвига | `--channel` (1279), `--fmin` (5), `--fmax` (40), `--tmin`, `--tmax`, `--output-dir` | `das_mh030_two_physical_scales/` |
| `plot_oct03_das_geophone_like_scec.py` | Обработка, максимально близкая к `scripts/safod/plot_das_geophone_two_physical_comparisons.py`: причинный FIR, 201 отвод при 500 Гц (групповая задержка 0.200 с, как у прежних 401 отвода при 1000 Гц) | `--channel` (1279), `--fmin` (5), `--fmax` (40), `--fir-numtaps` (201), `--tmin`, `--tmax`, `--corr-tmin`, `--corr-tmax`, `--output-dir` | `das_geophone_like_scec/` |
| `plot_oct03_m08_das_mh030_pb_v3.py` | Сравнение DAS / MH030 с запасными метаданными MH029; полосы 5–40 и 5–200 Гц; фильтр zero-phase | `--das-channel` (1694), `--output-dir` | `das_mh030_comparison_v3_zerophase/` |
| `plot_oct03_m08_das_mh030_pb_v5.py` | То же, с фильтром `DASutils.bandpass2D_c(..., zerophase=False)`, как в ноутбуке Ettore | `--das-channel` (1694), `--output-dir` | `das_mh030_comparison_v5_causal/` |

## v3 и v5

- Фильтр. v3: Butterworth 4-го порядка через `scipy.signal.sosfiltfilt`
  (zero-phase, времена остаются привязаны к времени в очаге). v5:
  `DASutils.bandpass2D_c(..., zerophase=False)`, один и тот же причинный
  фильтр для DAS и геофона после приведения к общей временной сетке DAS.
- Остальное по заголовкам совпадает: чтение protobuf
  `readFile_protobuf(..., diff=False, system="Sintela")`, полосы 5–40 и
  5–200 Гц, пары strain / velocity и strain rate / acceleration, канал 1694
  по умолчанию.
- Выход: раньше оба писали одинаковые имена файлов в
  `compare/das_mh030_comparison/`; теперь v3 пишет в
  `das_mh030_comparison_v3_zerophase/`, v5 — в `das_mh030_comparison_v5_causal/`.
- Какой вариант итоговый — уточнить у автора.

## Расхождения между скриптами

- Канал по умолчанию: v3 и v5 берут 1694, номер канала прежнего сравнения с
  MH029 «Pod 1». По заголовкам `find_current_mh030_das_channel.py` и
  `compare_oct03_current_turnaround.py`, номера текущих каналов Sintela
  (3236 каналов, шаг около 2.5321 м) не совпадают со старыми номерами из
  `SAFOD_Phase2_GeoReferenced_Channels.xlsx`.
  `compare_oct03_current_turnaround.py`, `compare_oct03_das_mh030_simple.py`
  и `plot_oct03_das_geophone_like_scec.py` по умолчанию берут текущий
  канал 1279.
- Запасное время в очаге на случай, если NCEDC недоступен: в v3 и v5 —
  04:52:31.000Z, в остальных скриптах — 04:52:31.320Z.
- Метаданные отклика MH030 (StationXML), по заголовкам, тогда были
  недоступны. Все скрипты, кроме `find_current_mh030_das_channel.py` (он
  работает с сырыми отсчётами GP1), используют вместо них метаданные MH029
  Pod 1 — сразу или как запасной вариант.

## Результаты

Все выводы по умолчанию идут в
`results/events/20261003_75443276/compare/<папка из таблицы>/`. Имя
`20261003_75443276` следует правилу `EVENT_RESULT_KEY`, хотя событие не
зарегистрировано. Этой папки в проекте пока нет, прежние выводы в проекте не
найдены. Правила раскладки результатов —
[results/README.md](../../../../results/README.md).

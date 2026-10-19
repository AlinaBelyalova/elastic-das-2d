# Elastic DAS Project

Проект — 2-D моделирование упругих волн и отклика DAS: решатель на сдвинутой
сетке с оператором наблюдения DAS, FWI на стеке Numba, сравнение с реальными
событиями SAFOD, работа с архивом непрерывных записей SAFOD-deep и
статический веб-просмотрщик его событий. Текущий этап — прототип SBP-SAT;
состояние и ближайшие шаги — в [docs/status.md](docs/status.md).

## Правило размещения

| Где | Что | Как пользоваться |
| --- | --- | --- |
| `src/` | библиотека | только импорт: `from src.<пакет>.<модуль> import ...` |
| `scripts/` | точки входа | `python -m scripts.<область>.<имя>`, всегда из корня проекта |
| `docs/` | объяснения | методы, runbook, состояние, карта проекта |
| `data/`, `config/` | входы | скоростные модели, корни архива SAFOD-deep |
| `results/` | всё вычисленное | папка = имя скрипта без `_qc`; [правила](results/README.md) |
| `scripts/legacy/`, `results/legacy/`, `docs/history/` | старое | старый код, старые результаты, прежние handoff; только для справки (`docs/history/` есть только в локальной копии, вне Git) |

Имена областей `sbp_sat`, `elastic`, `fwi`, `safod`, `safod_deep` повторяются
в `src/`, `scripts/`, `results/` и `docs/`, поэтому по одному имени находятся
код, запуск, результаты и документация. Например: `src/sbp_sat/` →
`scripts/validation/sbp_sat/` → `results/validation/sbp_sat/` →
`docs/sbp_sat/`. Отступления: результаты событий SAFOD лежат в
`results/events/`, FWI описан в `docs/elastic/method.md`, SAFOD-deep — в
`docs/safod/workflow.md`.

## Структура

```text
src/          библиотека: решатели, DAS, FWI, SBP-SAT, модели SAFOD, каталог SAFOD-deep
scripts/      точки входа: проверки, примеры, бенчмарки, SAFOD, SAFOD-deep, legacy
tests/        быстрые автоматические тесты (pytest.ini в корне)
docs/         состояние, карта проекта, методы, runbook SAFOD, история
config/       корни архива SAFOD-deep на Oak
data/         локальные скоростные модели SAFOD; сырые записи DAS лежат на Oak
results/      все вычисленные результаты и логи (локально, вне Git)
notebooks/    демонстрация и исследования
web_safod/    статический просмотрщик событий SAFOD-deep
```

В корне также `requirements.txt`, `pytest.ini` и `AGENTS.md` (инструкции
для AI-агентов).

## С чего начать

| Задача | Документ | Код | Результаты |
| --- | --- | --- | --- |
| Текущий этап | [docs/status.md](docs/status.md) | `src/sbp_sat/` | `results/validation/sbp_sat/` |
| Окружение, данные, тесты | [docs/setup.md](docs/setup.md) | `tests/`, `web_safod/tests/` | — |
| SBP-SAT | [метод](docs/sbp_sat/method.md), [проверки](docs/sbp_sat/validation.md) | `src/sbp_sat/`, `scripts/validation/sbp_sat/` | `results/validation/sbp_sat/` |
| Прежний решатель и DAS | [docs/elastic/method.md](docs/elastic/method.md) | `src/elastic/`, `src/das/`, `src/common/`, `src/analytical/`, `scripts/validation/elastic/` | `results/validation/elastic/`, `results/examples/`, `results/benchmarks/` |
| FWI | [docs/elastic/method.md](docs/elastic/method.md) | `src/fwi/`, `scripts/validation/fwi/` | `results/validation/fwi/` |
| Событие SAFOD | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/`, `src/safod/` | `results/events/` |
| Поиск событий SAFOD | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/selection/` | `results/safod/` |
| Архив SAFOD-deep и сайт | [docs/safod/workflow.md](docs/safod/workflow.md), [web_safod/README.md](web_safod/README.md) | `src/safod_deep/`, `scripts/safod_deep/`, `web_safod/` | `results/safod_deep/`, `web_safod/site/` |
| Скоростные модели | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/velocity_models/`, `src/safod/models/` | `data/safod/velocity_models/` |
| Все проверки | [scripts/validation/README.md](scripts/validation/README.md) | `scripts/validation/` | `results/validation/` |
| Где что лежит | [docs/project_map.md](docs/project_map.md) | — | [results/README.md](results/README.md) |
| История | [docs/history/README.md](docs/history/README.md) | `scripts/legacy/` ([README](scripts/legacy/README.md)) | `results/legacy/` |

## Быстрая проверка

Из корня проекта, в окружении `fwi`, на вычислительном узле
([окружение](docs/setup.md)):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.validation.sbp_sat.operator_qc
```

Проверяет SBP-тождество и баланс энергии SAT, выполняется за доли секунды и
ничего не записывает. Остальные проверки SBP-SAT, их команды и критерии —
в [docs/sbp_sat/validation.md](docs/sbp_sat/validation.md).

## Прежний решатель

Velocity-stress схема на сдвинутой сетке (Virieux), leapfrog по времени,
бэкенды NumPy и Numba; DAS — интерполяция, разность на gauge length и проекция
на касательную к кабелю. Подробно — [docs/elastic/method.md](docs/elastic/method.md).

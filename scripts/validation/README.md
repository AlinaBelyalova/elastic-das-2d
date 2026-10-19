# Проверки решателей

Кратко: одна таблица всех проверок из `scripts/validation/{sbp_sat,elastic,fwi}`,
примера `scripts/examples` и бенчмарков `scripts/benchmarks`: что проверяет,
сколько идёт, когда считается пройденной и куда пишет.

## Как запускать

- Из корня проекта, в окружении `fwi` ([docs/setup.md](../../docs/setup.md)):
  `python -m <модуль из таблицы>`.
- Команды SBP-SAT с параметрами (`--output-dir`, `--plot`, `--overwrite`) есть
  только в [docs/sbp_sat/validation.md](../../docs/sbp_sat/validation.md).
- На Sherlock всё, кроме коротких проверок, запускается в задании Slurm на
  вычислительном узле, а не на login-узле.
- Методы: [docs/elastic/method.md](../../docs/elastic/method.md) (решатель на
  сдвинутой сетке, DAS, FWI), [docs/sbp_sat/method.md](../../docs/sbp_sat/method.md)
  (SBP-SAT). Раскладка и имена папок результатов:
  [results/README.md](../../results/README.md).
- Скрипты с фиксированной папкой результатов создают её сами и перезаписывают
  файлы в ней без предупреждения. Защиту от записи в непустую папку
  (`--overwrite`) имеют только `sbp_sat.time_energy_qc` и `sbp_sat.oblique_qc`.

Время — замеры 2026-10-16 на вычислительном узле Sherlock: окружение `fwi`,
`OPENBLAS_NUM_THREADS=1`, запуск без `--output-dir`. «Не измерено» значит, что
проверку при составлении таблицы не запускали.

## Таблица

| Проверка (`python -m ...`) | Что проверяет | Время | Критерий прохождения | Куда пишет |
| --- | --- | --- | --- | --- |
| `scripts.validation.sbp_sat.operator_qc` | QC 1: тождество SBP для сдвинутых операторов и полудискретная скорость изменения энергии с SAT | ~0.2–0.3 с | max невязки SBP ≤ 1e-12 и относительная невязка dE/dt ≤ 1e-12; печатает `PASSED` | только stdout |
| `scripts.validation.sbp_sat.adjoint_dot_product` | QC 3: точный дискретный сопряжённый оператор источник → данные, `<F s, q> = <s, F^T q>` | ~0.2–0.3 с | относительная ошибка ≤ 1e-11 | только stdout |
| `scripts.validation.sbp_sat.taylor` | QC 4: тест Тейлора для градиента по `log(Vp)`, `log(Vs)` при фиксированной `rho` | ~0.9 с | наклон R0 в (0.8, 1.2), наклон R1 в (1.7, 2.3), относительная ошибка производной по направлению против центральной разности ≤ 1e-5 | только stdout |
| `scripts.validation.sbp_sat.normal_incidence_qc` | QC 2: отражение P и SV гауссовых пакетов при нормальном падении на свободную поверхность; скорость на поверхности равна удвоенной падающей; h = 10, 5, 2.5 м | > 30 с (остановлен по `timeout 30`; полное время не измерено) | относительная L2-ошибка строго убывает при измельчении для P и SV; пик и traction печатаются без порога | только stdout |
| `scripts.validation.sbp_sat.time_energy_qc` | полностью дискретная энергия leapfrog: неоднородная модель 32×41, без источника и затухания, 0.5 с | ~1.5 с | энергии конечны и положительны, дрейф инварианта < 1e-9, изменение центрированной энергии < 5% | только stdout; с `--output-dir` — `summary.json`, `traces.npz`, с `--plot` — PNG |
| `scripts.validation.sbp_sat.oblique_qc` | наклонное падение P (30°, 60°) и SV (20°, 30°): эволюция полного поля против континуального решения; h = 10, 5, 2.5, 1.25 м | > 30 с (остановлен по `timeout 30`; полное время не измерено) | ошибки волновой формы и traction убывают на каждом шаге измельчения; на самой мелкой сетке пять метрик ≤ 0.01 (подробно в [validation.md](../../docs/sbp_sat/validation.md)) | только stdout; с `--output-dir` — `summary.json`, `traces.npz`, с `--plot` — PNG |
| `scripts.validation.elastic.boundary_absorption` | поглощение гауссовой губкой на боковых границах: однородная среда, изотропный источник, разные ширины губки и `gamma_s` с контролем `gamma_s=0` на той же сетке | не измерено | единого порога нет: ошибка только при NaN/Inf или несогласованном наборе конфигураций; `WARNING`, если энергия прямой волны отличается от контроля той же ширины больше чем на 2%; печатает лучшие конфигурации и рекомендацию с учётом стоимости | `results/validation/elastic/boundary_absorption/` |
| `scripts.validation.elastic.free_surface_graves_fs1_qc` | алгебра и порядок шагов Graves FS1: граничные уравнения для скоростей после продолжения; заполнение ghost-узлов до обновления скоростей (старый порядок) их нарушает; повторное заполнение после обновления восстанавливает; отражение напряжений точное | ~2 с при готовом кэше Numba; первый запуск с JIT-компиляцией дольше (~4 с) | все точные невязки ≤ 1e-11; невязка старого порядка должна быть > 1e-9 | только stdout |
| `scripts.validation.elastic.free_surface_normal_incidence_qc` | отражение плоских P и SV пакетов при нормальном падении на рабочей сетке с Numba-ядрами Graves FS1; поверхность = 2 × падающая волна; h = 10, 5, 2.5 м при фиксированной физической области | не измерено | автоматического критерия нет: печатает rel L2, peak_ratio, peak_dt, traction и наблюдаемые порядки; ожидаемые тренды выводятся в конце | `results/validation/elastic/free_surface_normal_incidence/` |
| `scripts.validation.elastic.magnitude_scaling` | линейность амплитуды по моменту: Mw = 0, 1, 2, 3, наклон `log10` пиковой огибающей против Mw для `vx`, `vz`, DAS; `numba_fused`, `free_surface=True` | не измерено | отклонение всех трёх наклонов от 1.5 ≤ 1e-3 | `results/validation/elastic/magnitude_scaling/` |
| `scripts.validation.elastic.moment_tensor_analytical` | решатель с тензором момента против аналитического линейного источника в однородном пространстве: кинематика P/S, диаграмма направленности и полярность, какой величине (U, V, A) соответствует скорость, общий амплитудный множитель, сходимость интегрирования по y | не измерено | лучший кандидат — `velocity`, корреляция ≥ 0.999, отличие множителя от 1 ≤ 3%, относительная ошибка ≤ 0.02, сдвиг ≤ 1 отсчёта и др.; без `--strict` нарушения печатаются как `QC warnings` без ошибки, с `--strict` — ошибка | `results/validation/elastic/moment_tensor_analytical/` (`--out-dir`) |
| `scripts.validation.elastic.moment_tensor_linearity` | отклик на линейную комбинацию `Mxx`, `Mzz`, `Mxz` равен комбинации откликов (`vx`, `vz`, DAS), при `free_surface` False и True | не измерено | худшая относительная L2-ошибка ≤ 1e-8 | `results/validation/elastic/moment_tensor_linearity/` |
| `scripts.validation.elastic.pointforce_2d` | NumPy-решатель с точечной силой против аналитического 2-D `v_z` для вертикальной силы; порядки 2, 4, 6, 8 | не измерено | автоматического порога нет: печатает rel L2 и max abs по порядкам и коэффициенты уменьшения ошибки; ошибка только при неверных параметрах или несовпадении осей времени | `results/validation/elastic/pointforce_2d/` |
| `scripts.validation.elastic.reciprocity` | взаимность `G_zz(x_A, x_B) = G_zz(x_B, x_A)` для вертикальной точечной силы (`solver_numpy_pointforce`) | не измерено | худшая относительная L2-ошибка ≤ 1e-8 | `results/validation/elastic/reciprocity/` |
| `scripts.validation.elastic.source_radiation` | диаграммы направленности 2-D источников в однородном пространстве по всему пути: тензор → билинейное распределение → инъекция → распространение → выборка; изотропный и double couple при `theta` = 0, 22.5, 45° | не измерено | double couple: сходство диаграмм P и SV ≥ 0.70; изотропный: rel L2 диаграммы P ≤ 0.35 и энергия SV/P ≤ 0.10 | `results/validation/elastic/source_radiation/` |
| `scripts.validation.fwi.two_layer_numba` | контролируемая двухслойная модель: прямая задача Numba для истинной и гладкой начальной моделей (`vx`, `vz`, DAS), печатает расхождение | не измерено | нет NaN/Inf; с `--benchmark-warm` повторный расчёт совпадает с относительной ошибкой ≤ 1e-12; печатает `PASSED` | `results/validation/fwi/two_layer_numba/` (`--out-dir`) |
| `scripts.validation.fwi.two_layer_adjoint_dot_product` | точный дискретный сопряжённый оператор на двухслойной модели: A) источник → `vx`/`vz`; B) источник → приёмники → DAS с конечной базой; `free_surface=False` | не измерено | максимальная относительная ошибка dot-product < `--tolerance` (по умолчанию 1e-8) | только stdout |
| `scripts.validation.fwi.two_layer_taylor` | производная по направлению и тест Тейлора для градиента DAS с контрольными точками по `log(Vp)`, `log(Vs)` | не измерено | только с `--strict`: лучшая относительная ошибка производной < 1e-4, наклон R0 в [0.85, 1.15], R1 в [1.75, 2.25]; без `--strict` нарушения не дают ошибки, скрипт печатает `completed` | `results/validation/fwi/two_layer_taylor/` (`--out-dir`) |
| `scripts.examples.homogeneous_demo` | демонстрация всей цепочки: однородная модель, double couple вне узла с билинейным распределением, `numba_fused`, прямой кабель DAS, strain rate, сейсмограммы, грубое сравнение времён вступлений, анимация; вторая проверка — взрывной источник на NumPy; `free_surface=True` | не измерено | критерия нет: ошибка только при NaN/Inf; печатает сравнение времён вступлений | `results/examples/homogeneous_demo/` |
| `scripts.benchmarks.backends` | NumPy против Numba fused на сетке 601×601, nt = 800, порядок 4, при `free_surface` False и True: совпадение и время (лучшее из 3) | не измерено | порога нет: печатает max abs, rel L2 и `allclose(atol=1e-7, rtol=1e-5)` против NumPy, время и ускорение | только stdout |
| `scripts.benchmarks.performance` | масштабирование NumPy и Numba fused при `free_surface=True` на сетках 201², 401², 601² (nt = 400, 600, 800): время, ускорение, пропускная способность | не измерено | порога нет | `results/benchmarks/performance/` |

Пояснения к критериям: `PASSED` и `FAILED` — строки, которые печатает скрипт;
«ошибка» — исключение (`RuntimeError` и др.) и ненулевой код выхода.

## Самопроверки модулей `src`

Модули `src.elastic.source`, `src.elastic.source_spreading`,
`src.elastic.source_injection`, `src.elastic.sampling`, `src.das.receivers`,
`src.das.strain`, `src.analytical.moment_tensor` и `src.fwi.misfit` содержат
`_self_test()`, который выполняется при `python -m <модуль>`. Файлов они не
пишут; время не измерено. Быстрые регрессионные тесты (`python -m pytest`)
описаны в [docs/setup.md](../../docs/setup.md).

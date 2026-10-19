# Упругий решатель на сдвинутой сетке (NumPy / Numba)

Кратко: 2-D изотропный velocity-stress решатель Virieux на сдвинутой сетке
с leapfrog по времени, бэкенды NumPy и Numba, источники в виде тензора
момента, выборка в приёмниках, оператор DAS и слой FWI. На нём построены
SAFOD forward (`scripts/safod/run_forward.py`, бэкенд `numba_fused`) и
FWI-проверки. Отдельный прототип SBP-SAT описан в
[docs/sbp_sat/method.md](../sbp_sat/method.md); текущий этап и следующие
шаги — в [docs/status.md](../status.md). Список проверок — в
[scripts/validation/README.md](../../scripts/validation/README.md).

Документ пересказывает код. Источник истины — заголовки и docstring модулей,
указанных в каждом разделе.

## Постановка и дискретизация

Код: `src/elastic/solver_numpy.py`, `src/elastic/solver_numba_fused.py`,
`src/common/grid.py` (`Grid2D`), `src/common/model.py` (`ElasticModel2D`).

- Уравнения первого порядка velocity-stress в плоскости (x, z), поля
  `vx, vz, sxx, szz, sxz`, массивы `[ix, iz]` формы `(nx, nz)`. Индекс `iz`
  растёт вглубь: свободная поверхность — верхняя граница.
- Сдвинутая сетка Virieux (1986), требуется `dx == dz = h`
  (иначе `ValueError`):

  | Поле | Положение узла |
  | --- | --- |
  | `sxx`, `szz` | `(i*h, j*h)` |
  | `vx` | `((i+1/2)*h, j*h)` |
  | `vz` | `(i*h, (j+1/2)*h)` |
  | `sxz` | `((i+1/2)*h, (j+1/2)*h)` |

- Leapfrog, второй порядок по времени: `t_sigma[n] = n*dt`,
  `t_v[n] = (n+1/2)*dt`; шаг `sigma^n, v^(n-1/2) -> v^(n+1/2) -> sigma^(n+1)`.
- Порядок по пространству `2M`, `M = half_order = 1..4` (порядки 2, 4, 6, 8;
  по умолчанию `half_order=2`). Коэффициенты: классические Taylor (C-SFD,
  по умолчанию) или TS-SFD (`use_ts_sfd=True`, МНК-подбор по дисперсии;
  зависят от представительного числа Куранта `mean(vp)*dt/dx`).
- Материалы: `mu = rho*vs²`, `lambda = rho*(vp² - 2*vs²)`; решатель требует
  `mu > 0`, `lambda >= 0` (то есть `Vs <= Vp/sqrt(2)`) и `lambda + 2*mu > 0`.
  На сдвинутые узлы: `1/rho` в узлах `vx` и `vz` — обратное среднее
  арифметическое двух соседних `rho`; `mu_xz` — гармоническое среднее
  четырёх соседей.
- Шаг по времени: решатель отклоняет `dt` больше границы
  `max_stable_dt(..., safety=1.0)` (консервативные пределы Куранта для
  каждого порядка заданы в коде; для TS-SFD запас дополнительно уменьшен).

Порядок операций на одном шаге (одинаков в NumPy и Numba):

1. обновление скоростей `sigma^n -> v^(n+1/2)`;
2. при `free_surface=True` — продолжение скоростей над поверхностью (Graves FS1);
3. обновление напряжений `-> sigma^(n+1)`;
4. инъекция источника в обновлённые напряжения;
5. при `free_surface=True` — отражение (imaging) напряжений FS1;
6. губка (sponge) для всех пяти полей;
7. выборка в приёмниках в момент `t_v[n]`.

## Бэкенды и сборка расчёта

| Модуль | Роль |
| --- | --- |
| `src/elastic/solver_numpy.py` | `run_elastic_solver_numpy`: базовая NumPy-реализация; источник только в узел `(ix, iz)` |
| `src/elastic/solver_numba_fused.py` | `run_elastic_solver_numba_fused`: рабочий Numba-бэкенд с тем же API и дополнительным `source_injection`; должен оставаться математически согласованным с NumPy |
| `src/elastic/solver_numpy_pointforce.py` | только для проверок (`pointforce_2d`, `reciprocity`): вертикальная точечная сила в уравнении для `vz`, порядки 2–8, без свободной поверхности (губка со всех сторон) |
| `src/elastic/simulator.py` | `run_forward_simulation(..., backend="numba_fused" \| "numpy", free_surface=False)`: выборка, инъекция, решатель, затем DAS; возвращает `(ElasticRunResult, DASResult)` |

Сравнение бэкендов по точности и скорости: `scripts.benchmarks.backends`,
`scripts.benchmarks.performance`.

## Свободная поверхность: Graves (1996) FS1

Код: `src/elastic/solver_numpy.py` (заголовок, блоки 1b, 2 и 3b цикла),
`src/elastic/solver_numba_fused.py` (`fill_velocity_ghosts_free_surface_numba`,
`update_stress_fused_free_surface_numba`, `mirror_stress_ghosts_free_surface_numba`).

- Планарная FS1 для схемы 2-4. При `free_surface=True` и `half_order != 2`
  оба бэкенда выдают `NotImplementedError`. Порядки 2, 6, 8 доступны только
  без свободной поверхности.
- Физическая поверхность — строка нормальных напряжений `iz = M`; узлы
  `iz = 0..M-1` — ghost-узлы (воздух). Верхняя губка отключена.
- Скорости над поверхностью строятся из граничных условий
  `sigma_zz = 0`, `sigma_xz = 0` второго порядка после обновления скоростей
  (на новом полушаге). На поверхности вертикальная производная в уравнении
  для горизонтального нормального напряжения — второго порядка.
- После инъекции источника: `szz[k] = 0`, `szz[k-1] = -szz[k+1]`,
  `sxz[k-1] = -sxz[k]`, `sxz[k-2] = -sxz[k+1]` (`k = M`).
- Источник тензора момента должен лежать на `iz >= M+1`, иначе он
  попадёт в `szz` на поверхности.
- В сопряжённом операторе и градиенте свободной поверхности нет:
  `src/fwi/adjoint_numba.py` и `src/fwi/gradient_numba.py` отклоняют
  `free_surface=True`.

Формулировка прежнего README «flat free-surface boundary condition using
ghost nodes» относилась к прежней смешанной реализации: ghost-скорости
заполнялись до обновления скоростей, то есть на старом полушаге. Её заменила
Graves FS1 (ghost-узлы остались, изменились формулы и порядок шагов);
причины — в handoff 2026-10-08, §H–§J. Проверки: `scripts.validation.elastic.free_surface_graves_fs1_qc`
(алгебра и порядок шагов), `scripts.validation.elastic.free_surface_normal_incidence_qc`
(отражение при нормальном падении).

## Поглощающие границы: губка

Код: `make_sponge_mask` в `src/elastic/solver_numpy.py`; Numba-бэкенд
использует ту же маску.

- PML нет. Реализована мультипликативная гауссова губка: в слое шириной
  `n_boundary` ячеек множитель `exp(-gamma_s*dt*w²)`, `w = (n_boundary - i)/n_boundary`
  (`i = 0` на внешнем краю); в углах множители перемножаются.
- Маска умножает `vx, vz, sxx, szz, sxz` в конце каждого шага. Скорости и
  напряжения относятся к разным моментам времени, поэтому в заголовке это
  названо приближением O(dt).
- При `free_surface=True` верх не демпфируется.
- Нужно `n_boundary > M`: разностные операторы не обновляют `M` крайних ячеек.
  Значения по умолчанию в решателях и `run_forward_simulation`:
  `n_boundary=40`, `gamma_s=300.0`.
- Источник должен лежать вне губки: `ix` в `[n_boundary, nx - n_boundary)`,
  `iz` в `[n_boundary, nz - n_boundary)`; при `free_surface=True` минимальный
  `iz` равен `M+1`.
- В части скриптов и в `src/das/receivers.py` ширина губки называется
  `n_pml`; это та же губка, не PML. В `receivers.py` параметр `n_pml`
  (по умолчанию 0) нужен только для проверки, что каналы не лежат в слое
  такой ширины у любой из четырёх границ: о `free_surface` эта проверка не
  знает и отклоняет каналы с `iz < n_pml` и у верхней границы.

Подбор `n_boundary` и `gamma_s`: `scripts.validation.elastic.boundary_absorption`.

## Источник

Код: `src/elastic/source.py`, `src/elastic/source_spreading.py`,
`src/elastic/source_injection.py`.

- Тензор момента `MomentTensor2D(Mxx, Mzz, Mxz)` по соглашению
  Aki & Richards (2002), гл. 3. Базовый double couple
  `M0 * [[0, 1], [1, 0]]` (нормаль `z`, подвижка `x`); `theta_deg` поворачивает
  разлом в плоскости x–z (`build_rotated_double_couple_2d`). Изотропный
  тензор — `build_isotropic_source_tensor_2d`. Построение источника:
  `build_source_2d`, `build_dc_source`, `build_isotropic_source`.
- Единицы: N·m в смысле 2-D модельного соглашения; согласование амплитуд
  2-D и 3-D (реальные моменты землетрясений) в коде отмечено как отдельный
  вопрос.
- Временная функция `SourceTimeFunction`: `values[n]` умножается на компоненту
  тензора (в `run_forward_simulation`) и на `dt/(dx*dz)` (в решателе) и
  добавляется к напряжениям на шаге `n`, то есть к `sigma^(n+1)`. Режим
  задаёт `source_time_mode`; при `source_time_mode=None` выбор делает
  `derivative_order`: 0 — `legacy_ricker` (по умолчанию), 1 —
  `legacy_ricker_derivative`. Docstring `build_source_time_function` называет
  физическим (рекомендуемым) режимом `ricker_moment`.
  - `source_time_mode="ricker_moment"`: `values = -dW/dt` в момент
    `(n+1/2)*dt`, где `W(t)` — история момента Рикера (`S_M = -dM_phys/dt`).
    В этом режиме скорость в приёмниках сравнима с аналитической скоростью
    для истории момента `W(t)`.
  - `legacy_ricker`, `legacy_ricker_derivative`: оставлены для
    воспроизводимости; это общие вейвлеты скорости напряжений, а не
    физическая история момента Рикера.
- Распределение по сетке (`spreading`):
  - `"nearest"` (по умолчанию): `sxx`, `szz` — в один узел; `sxz` — по 0.25
    в четыре окружающих узла `sxz`, центр совпадает с узлом источника;
  - `"bilinear"`: билинейные веса по физическим координатам источника, каждая
    компонента на своей подсетке (`Mxx -> sxx`, `Mzz -> szz`, `Mxz -> sxz`).
    В заголовке `source.py` этот режим указан для SAFOD и будущей FWI.
- `source_injection.py` переводит источник в шесть массивов формы `(4,)`; одно
  ядро инъекции обслуживает оба режима. Билинейный режим есть только в
  `numba_fused`; `numpy` переходит на `"nearest"` с `UserWarning`.

Проверки: `scripts.validation.elastic.source_radiation`,
`moment_tensor_linearity`, `magnitude_scaling`, `moment_tensor_analytical`.

## Приёмники и выборка

Код: `src/das/receivers.py`, `src/elastic/sampling.py`.

- `Receivers2D`: координаты центров каналов `x, z`, единичная касательная
  `tx, tz` в плоскости x–z, равномерная координата вдоль кабеля `s`
  (для скважинной геометрии подходит measured depth). Поля `ix, iz`
  устарели и оставлены для совместимости.
- `build_das_cable_from_waypoints` — синтетические равномерные каналы вдоль
  ломаной; `build_receivers_from_channel_centres` — заданные полевые центры
  каналов без пересэмплирования.
- `build_receiver_sampling` один раз вычисляет билинейные веса для `vx` и `vz`
  на их сдвинутых подсетках; в цикле используются только массивы.
  `scatter_adjoint` — точное алгебраическое транспонирование выборки.
  Записи приёмников относятся к моментам `t_v[n] = (n+1/2)*dt`.

## Оператор DAS

Код: `src/das/strain.py`.

- `compute_axial_strain_rate(vx, vz, receivers, gauge_length_m)` (из скоростей)
  и `compute_axial_strain(ux, uz, ...)` (из смещений):

  ```text
  eps_dot_ss(s) = ( [vx(s+L/2) - vx(s-L/2)] * tx(s)
                  + [vz(s+L/2) - vz(s-L/2)] * tz(s) ) / L
  ```

  `tx, tz` берутся в центре базы.
- Правило: сначала разность декартовых компонент на базе `L`, затем проекция
  на касательную в центре базы. При обратном порядке жёсткий сдвиг даёт
  ненулевой отклик на изогнутом кабеле.
- `L` — непрерывный параметр, не обязан быть кратным шагу каналов: значения
  в `s ± L/2` берутся линейной интерполяцией вдоль равномерного массива
  каналов. Выдаются только каналы, у которых `s ± L/2` лежит в пределах
  кабеля (`DASResult.channel_indices`).
- Сохраняются точно: жёсткий сдвиг даёт ноль; поле, линейное по `s`,
  восстанавливается точно (проверяется в `_self_test`).
- `das_adjoint` — точное транспонирование (dot-product тест в `_self_test`).
- В `run_forward_simulation` DAS считается после расчёта по `receiver_vx`,
  `receiver_vz` (strain rate).

## Аналитические эталоны

| Модуль | Содержание | Проверки |
| --- | --- | --- |
| `src/analytical/pointforce_2d.py` | 2-D функция Грина `v_z` от вертикальной точечной силы в однородной среде (plane strain, ядро по Kausel 2006) в частотной области; сдвиг на `dt/2` множителем `exp(+i*omega*dt/2)` для сравнения с leapfrog-скоростями | `scripts.validation.elastic.pointforce_2d` (порядки 2, 4, 6, 8) |
| `src/analytical/moment_tensor.py` | однородное полное пространство, симметричный тензор момента; 2-D решение — интеграл 3-D точечного решения вдоль `y`; основано на `AnalyticalSolution.py` Haipeng Li (Aki & Richards, 2002) | `scripts.validation.elastic.moment_tensor_analytical` |

В заголовке `pointforce_2d.py` ядро названо «practical analytical reference
candidate»: если график сходимости ведёт себя неожиданно, первым делом
перепроверять само ядро — множители, знаки и ближнее поле.

## Слой FWI (`src/fwi`)

| Модуль | Содержание |
| --- | --- |
| `two_layer.py` | ООП-слой над Numba-решателем (`TwoLayerFWIProblem`, `NumbaElasticDASForward` и спецификации модели, домена, решателя, DAS, источника); Python-объекты не входят в горячие циклы |
| `adjoint_numba.py` | `NumbaElasticDASAdjoint`: точный дискретный сопряжённый оператор Numba-решателя для `free_surface=False`; C-SFD и TS-SFD, губка, билинейные приёмники, источник `nearest`/`bilinear`; транспонирование DAS — через `das_adjoint`. Обратный шаг: разброс приёмников, губка, сбор источника, транспонирование обновлений напряжений и скоростей |
| `gradient_numba.py` | `NumbaElasticDASGradient`: градиент с контрольными точками (checkpointing) по `lambda, mu` с пересчётом в `vp, vs` и `log(vp), log(vs)`; `rho` фиксирована; транспонирование гармонического `mu_xz` учтено. `free_surface=True` и TS-SFD не поддерживаются (`NotImplementedError`). Целевая функция `J = 0.5*dt*dCh*||d_syn - d_obs||²` |
| `misfit.py` | `l2_das_misfit`: `J = 0.5*dt*dCh*sum(w*(syn - obs)²)` и `dJ/d(syn)`; градиентом пока не используется |

Проверки: `scripts.validation.fwi.two_layer_numba`,
`two_layer_adjoint_dot_product`, `two_layer_taylor`.

## Проверки и контекст

- Все проверки, критерии, время и папки результатов:
  [scripts/validation/README.md](../../scripts/validation/README.md).
- Окружение и запуск на Sherlock: [docs/setup.md](../setup.md).
- Предыстория (adjoint, градиент, временные уровни, замена свободной
  поверхности, вывод и проверки Graves FS1):
  `docs/history/FWI_SAFOD_FULL_HANDOFF_2026-10-08.md`,
  §E–§M. Файлы `docs/history/` локальные и в Git не входят; указатель —
  `docs/history/README.md`.

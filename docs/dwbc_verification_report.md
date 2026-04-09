# Отчёт сверки DWBC с чеклистом и с методом из статьи Deep Whole-Body Control

Сравнение реализации в `source/isaaclab_rl/rsl_rl/dwbc/` и конфига Husky в `source/isaaclab_tasks/.../full_body_control/config/husky/` с пайплайном статьи (Fu, Cheng, Pathak, CoRL 2022) и типичной реализацией (например, [MarkFzp/Deep-Whole-Body-Control](https://github.com/MarkFzp/Deep-Whole-Body-Control)).

**Проверка по коду:** каждый пункт чеклиста сопоставлен с конкретными файлами и строками логики. Прямое сравнение с репозиторием Deep-Whole-Body-Control требует клонирования репо и сверки файлов (ppo, actor_critic, runner, reward grouping) вручную — в отчёте отмечены только возможные отличия, известные по статье и типичным реализациям.

---

## 1. Две группы наград (leg / arm)

**Чеклист:** Имена в `REWARD_GROUPS` совпадают с полями в `HuskyDwbcRewardsCfg`; сумма по группе locomotion (manipulation) = «награда за базу» («награда за руку»).

**У нас:**

- **Конфиг групп:** [`FBC_dwbc_env_cfg.py`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/full_body_control/config/husky/FBC_dwbc_env_cfg.py) — `REWARD_GROUPS = {"locomotion": [...], "manipulation": [...]}`. Все перечисленные имена являются полями `HuskyDwbcRewardsCfg` (каждый элемент — `RewTerm(...)` с уникальным именем атрибута).
- **Wrapper:** [`wrapper_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/wrapper_dwbc.py):
  - `_build_reward_term_mapping()` строит `_term_name_to_idx` из `reward_manager._term_names` и проверяет, что каждый `term_name` из `reward_groups` есть в менеджере (иначе `ValueError`).
  - `_compute_group_reward(group_name)` суммирует `rm._step_reward[:, idx] * dt` по всем терминам группы — получается одна скалярная награда на env для «locomotion» и одна для «manipulation».

**Вывод:** Соответствует: две группы, имена терминов совпадают с конфигом наград, логика «база» vs «рука» зашита в состав групп (locomotion = стабилизация/колёса, manipulation = EE/рука).

**Отличие от репо:** В оригинале обычно те же две группы (leg/arm или locomotion/manipulation); способ агрегации (сумма по терминам с весами и dt) у нас явный в wrapper, в оригинале может быть внутри env.

---

## 2. Наблюдения policy и privileged

**Чеклист:** В observation_manager есть группы `policy` и `privileged`; wrapper отдаёт `obs_prop = obs_dict["policy"]`, `obs_priv = obs_dict["privileged"]`; размеры совпадают с ожиданиями ActorCriticDWBC (num_prop, num_priv).

**У нас:**

- **Конфиг:** [`FBC_dwbc_env_cfg.py`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/full_body_control/config/husky/FBC_dwbc_env_cfg.py) — `HuskyDwbcObservationsCfg`: атрибуты `policy = PolicyCfg()` и `privileged = PrivilegedCfg()`. Имена групп в менеджере наблюдений задаются этими атрибутами (см. `observation_manager` в Isaac Lab).
- **Wrapper:** [`wrapper_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/wrapper_dwbc.py) `_extract_obs()`: `obs_prop = obs_dict["policy"]`, `obs_priv = obs_dict.get("privileged", None)` (если нет — нулевой тензор длины 0). Возвращает `(obs_prop, obs_priv)`.
- **Сеть:** В `DwbcOnPolicyRunner` при создании `ActorCriticDWBC` передаются `num_prop = obs_prop.shape[-1]`, `num_priv = obs_priv.shape[-1]` после первого `env.reset()`, т.е. размеры приходят из реального env.

**Вывод:** Соответствует: группы `policy` и `privileged` заданы в конфиге, wrapper явно отдаёт их в (obs_prop, obs_priv), размеры для сети берутся из env.

---

## 3. Действия: сначала leg, потом arm (2 + 6 = 8)

**Чеклист:** В `HuskyDwbcActionsCfg` порядок: `vel_actions`, затем `arm_actions`. В `rsl_rl_dwbc_cfg.py`: `num_leg_actions=2`, `num_arm_actions=6`. В action_manager порядок и размерности совпадают (2 + 6 = 8).

**У нас:**

- **Конфиг окружения:** [`FBC_dwbc_env_cfg.py`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/full_body_control/config/husky/FBC_dwbc_env_cfg.py) — `HuskyDwbcActionsCfg`: первый атрибут `vel_actions` (DiffDriveVelocityActionCfg → 2 скаляра), второй `arm_actions` (JointPositionActionCfg для 6 суставов). В Isaac Lab порядок терминов в `action_manager` совпадает с порядком полей в конфиге, итоговая размерность — `total_action_dim` (сумма по терминам).
- **Конфиг агента:** [`rsl_rl_dwbc_cfg.py`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/full_body_control/config/husky/agents/rsl_rl_dwbc_cfg.py): `num_leg_actions=2`, `num_arm_actions=6`. В `actor_critic_dwbc.py` выход актора — `torch.cat([leg_out, arm_out], dim=-1)`, т.е. [leg, arm]; в `get_actions_log_prob` срез `[:, :num_leg_actions]` и `[:, num_leg_actions:]` — совпадает с порядком в конфиге действий.

**Вывод:** Соответствует: порядок leg → arm и размерности 2 + 6 = 8 согласованы между env и сетью.

---

## 4. Сеть: актор — один бэкбон, две головы

**Чеклист:** Вход актора — obs_prop и латент (priv или history); один backbone, два головы; выход — concat(leg, arm); размерности и num_leg_actions/num_arm_actions совпадают.

**У нас:** [`actor_critic_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/actor_critic_dwbc.py):

- Вход: `obs_prop` (num_prop) и `latent` (priv_encoder или history_encoder, размер `priv_encoder_dims[-1]` = 20).
- `actor_input_dim = num_prop + latent_dim`; `actor_backbone(obs_prop, latent)` через `torch.cat([obs_prop, latent], dim=-1)` и MLP.
- Две головы: `leg_action_head(backbone_out)` → num_leg_actions, `arm_action_head(backbone_out)` → num_arm_actions; выход `torch.cat([leg_out, arm_out], dim=-1)`.

**Вывод:** Соответствует статье (unified policy, separate leg/arm heads).

---

## 5. Критик: два value-выхода (batch, 2)

**Чеклист:** `evaluate()` возвращает тензор формы (batch, 2); при act() в transition.values записывается (N, 2).

**У нас:**

- [`actor_critic_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/actor_critic_dwbc.py): `evaluate(obs_prop, obs_priv)` → `torch.cat([leg_val, arm_val], dim=-1)` — форма (batch, 2).
- [`ppo_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/ppo_dwbc.py) в `act()`: `self.transition.values = self.actor_critic.evaluate(obs_prop, obs_priv).detach()` — при N env это (N, 2). В `DwbcRolloutStorage.Transition` поле `values` описано как (num_envs, 2).

**Вывод:** Соответствует.

---

## 6. Привилегированный энкодер (латент 20)

**Чеклист:** Размер латента = последний элемент `priv_encoder_dims` (20); выход priv_encoder = вход actor_backbone по второй координате (latent dim).

**У нас:** [`actor_critic_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/actor_critic_dwbc.py):

- `priv_encoder`: MLP из num_priv в `priv_encoder_dims` (последний слой → latent_dim = 20).
- `actor_input_dim = num_prop + latent_dim`; в `_actor_forward` используется `torch.cat([obs_prop, latent], dim=-1)` — вторая координата по латенту = latent_dim.

**Вывод:** Соответствует.

---

## 7. History encoder и буфер истории

**Чеклист:** Вход (batch, history_len, num_prop), выход (batch, latent_dim). Буфер `_obs_history_buf` размером num_envs * (history_len * num_prop); при act() форма совместима с `infer_hist_latent(obs_history.view(-1, history_len, num_prop))`.

**У нас:**

- [`actor_critic_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/actor_critic_dwbc.py): `infer_hist_latent(obs_history)` вызывает `self.history_encoder(obs_history.view(-1, self.history_len, self.num_prop))` — вход (batch, history_len, num_prop), выход (batch, latent_dim) (StateHistoryEncoder до linear_output даёт тот же latent_dim).
- [`runner_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/runner_dwbc.py): `_obs_history_buf` имеет форму `(num_envs, history_len * num_prop)`. При шаге вызывается `self.alg.act(obs_prop, obs_priv, self._obs_history_buf, hist_encoding=...)`; в `_update_obs_history` при dones буфер для сброшенных env обнуляется, затем сдвиг и добавление текущего obs_prop. В `infer_hist_latent` передаётся именно плоский буфер, который внутри сети переводится в view(-1, history_len, num_prop).

**Вывод:** Соответствует; при reset обнуление по done есть.

---

## 8. Два потока returns/advantages (T, N, 2)

**Чеклист:** В `compute_returns` GAE по двум компонентам; после вызова `self.returns` и `self.advantages` имеют форму (T, N, 2).

**У нас:** [`rollout_storage_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/rollout_storage_dwbc.py):

- `rewards`, `values`, `returns`, `advantages` объявлены как (T, N, 2).
- `compute_returns(last_values, gamma, lam)`: один цикл по шагам в обратном порядке; `delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]` — тензоры формы (N, 2), GAE применяется к обеим компонентам одинаково. Затем `self.returns[step] = advantage + self.values[step]`, `self.advantages = self.returns - self.values`, и одна общая нормализация: `(self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)` по всему тензору.

**Возможное отличие от оригинала:** В части реализаций преимущества нормализуют по каждой компоненте отдельно (отдельный mean/std для leg и arm). У нас — одна общая нормализация по всем (T*N*2) элементам. Это допустимый вариант, но может влиять на масштаб градиентов по разным подзадачам.

**Вывод:** Формы (T, N, 2) и двухкомпонентный GAE соответствуют; нормализация advantages — общая.

---

## 9. Advantage mixing

**Чеклист:** Используется `value_mixing_ratio = get_value_mixing_ratio()` и формула mixing_adv для обеих компонент; при mixing_schedule [0,0,0] ratio = 0; при ненулевом schedule ratio растёт — логировать mixing_ratio в TensorBoard.

**У нас:** [`ppo_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/ppo_dwbc.py):

- `get_value_mixing_ratio()`: `sched = self.mixing_schedule` (формат [max_ratio, start_iter, ramp_iters]); `ratio = min(max((self.counter - sched[1]) / max(sched[2], 1), 0), 1) * sched[0]`. При Husky `mixing_schedule=[0.0, 0.0, 0.0]` → ratio всегда 0 (только «свои» advantages).
- В update: `mixing_adv[..., 0] = advantages_b[..., 0] + value_mixing_ratio * advantages_b[..., 1]`; `mixing_adv[..., 1] = advantages_b[..., 1] + value_mixing_ratio * advantages_b[..., 0]` — как в статье (causal dependency: leg advantage подмешивается к arm и наоборот).
- В [`runner_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/runner_dwbc.py) в `log()` пишется `self.writer.add_scalar("Loss/mixing_ratio", locs["mixing_ratio"], locs["it"])` — mixing_ratio логируется в TensorBoard.

**Вывод:** Соответствует статье; при [0,0,0] cross-mixing отключён; ratio логируется.

---

## 10. Log-prob по компонентам (batch, 2)

**Чеклист:** `get_actions_log_prob` возвращает (batch, 2): сумма log_prob по leg и по arm; размерности в storage и в surrogate совпадают.

**У нас:** [`actor_critic_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/actor_critic_dwbc.py):

- `get_actions_log_prob(actions)`: `log_prob = self.distribution.log_prob(actions)` по всем 8 действиям; затем `leg_lp = log_prob[:, :num_leg_actions].sum(dim=-1, keepdim=True)`, `arm_lp = log_prob[:, num_leg_actions:].sum(dim=-1, keepdim=True)`, возврат `torch.cat([leg_lp, arm_lp], dim=-1)` — (batch, 2).
- В [`ppo_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/ppo_dwbc.py) в update: `actions_log_prob_b = self.actor_critic.get_actions_log_prob(actions_b)` и `old_actions_log_prob_b` из storage; оба (batch, 2); `ratio = torch.exp(actions_log_prob_b - old_actions_log_prob_b)` — (batch, 2); surrogate использует `mixing_adv * ratio` и т.д., размерности совпадают.

**Вывод:** Соответствует.

---

## 11. Privileged regularizer

**Чеклист:** В update() считаются priv_latent и hist_latent (inference); loss += priv_reg_coef * ||priv_latent - hist_latent.detach()||; расписание priv_reg_coef_schedule (например [0, 0.1, 3000, 7000]); coef в логах меняется.

**У нас:** [`ppo_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/ppo_dwbc.py):

- В update: `priv_latent_b = self.actor_critic.infer_priv_latent(obs_prop_b, obs_priv_b)`; `hist_latent_b = self.actor_critic.infer_hist_latent(obs_hist_b)` в `torch.inference_mode()`; `priv_reg_loss = (priv_latent_b - hist_latent_b.detach()).norm(p=2, dim=1).mean()`; `priv_reg_coef = self._get_priv_reg_coef()`; в loss добавляется `priv_reg_coef * priv_reg_loss`.
- `_get_priv_reg_coef()`: по расписанию [coef_start, coef_end, ramp_start_iter, ramp_duration] линейный рост от coef_start до coef_end за ramp_duration итераций начиная с ramp_start_iter. В Husky: [0, 0.1, 3000, 7000] — с итерации 3000 за 7000 итераций рост от 0 до 0.1.
- В runner при логировании пишутся `Loss/priv_reg_loss` и `Loss/priv_reg_coef` — можно отслеживать изменение coef.

**Вывод:** Соответствует Regularized Online Adaptation из статьи.

---

## 12. DAgger: каждые dagger_update_freq итераций

**Чеклист:** Каждые dagger_update_freq итераций rollout с hist_encoding=True, затем только update_dagger() (обновляется только history_encoder); в логах чередуются update() и update_dagger(); hist_latent_loss логируется в DAgger-итерациях.

**У нас:** [`runner_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/runner_dwbc.py):

- `hist_encoding = (it % self.dagger_update_freq == 0)` — раз в 20 итераций True (Husky: dagger_update_freq=20).
- В rollout всегда вызывается `self.alg.act(..., hist_encoding=hist_encoding)` — в DAgger-итерациях сбор с history encoder.
- После rollout: если `hist_encoding` — вызывается только `self.alg.update_dagger()` (обновляет только history_encoder через hist_latent_loss); иначе `self.alg.update()` (полный PPO).
- В `update_dagger()` в ppo_dwbc: минимизируется ||priv_latent.detach() - hist_latent||, градиенты только по history_encoder.
- В `log()` пишутся `Loss/hist_latent_loss` и для не-DAgger итераций — value_loss, surrogate_loss и т.д.; hist_latent_loss на DAgger-итерациях заполняется в `mean_hist_latent_loss` (в update_dagger), на остальных остаётся от предыдущего DAgger-шага или 0.

**Вывод:** Соответствует; чередование PPO / DAgger и логирование есть.

---

## 13. Min policy std

**Чеклист:** В конфиге задан min_policy_std для leg и arm; после каждого PPO update вызывается _enforce_min_std(); в логах leg/arm mean noise std не опускаются ниже минимумов.

**У нас:** [`rsl_rl_dwbc_cfg.py`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/full_body_control/config/husky/agents/rsl_rl_dwbc_cfg.py): `min_policy_std = [0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28]` (2 leg + 6 arm). [`ppo_dwbc.py`](../source/isaaclab_rl/isaaclab_rl/rsl_rl/dwbc/ppo_dwbc.py): в конце `update()` вызывается `self._enforce_min_std()`; там `new_std = torch.max(current_std, self.min_policy_std)` и присваивание в `self.actor_critic.std.data`. В runner в `log()` пишутся `Policy/leg_mean_noise_std` и `Policy/arm_mean_noise_std` (среднее по std для leg и arm действий) — при корректном enforce они не будут ниже 0.28.

**Вывод:** Соответствует; _enforce_min_std вызывается после каждого PPO update (не после update_dagger).

---

## Краткая сводка отличий от типичной реализации (статья / репо)

| Аспект | Наша реализация | Возможное отличие в оригинале |
|--------|------------------|-------------------------------|
| Нормализация advantages | Одна общая по (T,N,2) | Иногда нормализуют по каждой компоненте отдельно |
| Робот / симулятор | Husky+UR5, Isaac Lab | Статья: четвероногий манипулятор, Isaac Gym |
| Torque supervision | Не используется | В статье может быть вспомогательный loss по крутящему моменту |
| Имена групп наград | locomotion / manipulation | В репо могут быть leg / arm или другие имена при той же логике |

Итог: все пункты чеклиста по коду выполняются; отличия — в основном окружение и опциональные детали (нормализация advantages, torque), а не в ядре метода (два потока наград, два критика, advantage mixing, privileged/history encoders, DAgger, priv reg, min std).

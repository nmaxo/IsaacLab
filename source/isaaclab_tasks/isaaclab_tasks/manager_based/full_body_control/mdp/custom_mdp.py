import torch
from isaaclab_tasks.manager_based.navigation.mdp import (
    JointVelocityAction,
    JointVelocityActionCfg,
    ActionTerm,
)
from isaaclab.utils import configclass
from skrl.utils import logger


class DiffDriveVelocityAction(JointVelocityAction):
    """ActionTerm для дифференциального привода 4 колес.

    Ожидает от политики 2 действия (линейная и угловая скорость платформы):
    - action[0]: v_lin — нормированная линейная скорость вдоль X базы [-1, 1] -> м/с
    - action[1]: omega — нормированная угловая скорость (yaw) [-1, 1] -> рад/с

    Внутри преобразует (v_lin, omega) в угловые скорости 4 колёс через дифф. кинематику
    и записывает в _processed_actions для apply_actions (4 колёса).
    """

    cfg: "DiffDriveVelocityActionCfg"

    @property
    def action_dim(self) -> int:
        """Размер входа от политики: 2 (v_lin, omega). Не 4 (колёса)."""
        return 2

    def process_actions(self, actions: torch.Tensor):
        """
        Преобразует действия RL в угловые скорости колес.
        Args:
            actions: [num_envs, 2] — команды от политики
        """
        # ограничим диапазон [-1, 1]
        actions = actions.clamp(-1.0, 1.0)

        # масштабируем под физические значения
        # linear_velocity_sign: для Husky USD часто +X базы смотрит назад; при -1 положительная
        # команда "вперёд" даёт движение к цели (синяя стрелка совпадает с зелёной).
        v_lin = self.cfg.linear_velocity_sign * actions[:, 0:1] * self.cfg.max_linear_speed   # м/с
        omega = actions[:, 1:2] * self.cfg.max_angular_speed  # рад/с

        # параметры робота
        L = self.cfg.wheel_base      # расстояние между левыми и правыми колесами (м)
        R = self.cfg.wheel_radius    # радиус колеса (м)

        # дифференциальная кинематика
        v_left = (v_lin - omega * L / 2.0) / R
        v_right = (v_lin + omega * L / 2.0) / R

        # создаем вектор скоростей для 4-х колес
        # [front_left, front_right, rear_left, rear_right]
        wheel_vels = torch.cat([v_left, v_right, v_left, v_right], dim=1)

        wheel_vels = wheel_vels * self.cfg.scale
            # ← ЛОГИРОВАНИЕ через skrl:
        # logger.info(f"max_angular: {self.cfg.max_angular_speed}")
        # logger.info(f"omega max: {omega.max():.3f}")
        # logger.info(f"v_right - v_left: {(v_right - v_left).max():.3f}")

        # сохраняем обработанные действия
        self._processed_actions = wheel_vels


@configclass
class DiffDriveVelocityActionCfg(JointVelocityActionCfg):
    """Конфиг для DiffDriveVelocityAction"""

    class_type: type[ActionTerm] = DiffDriveVelocityAction

    # геометрия робота
    wheel_radius: float = 0.33 # м
    wheel_base: float = 0.8    # м (расстояние между левыми и правыми колесами)

    # максимальные скорости
    max_linear_speed: float = 10.0   # м/с
    max_angular_speed: float = 10.0  # рад/с
    scale: float = 1.0
    # знак линейной скорости: 1.0 = положительный action = движение по +X базы;
    # -1.0 = если в USD ось X базы смотрит назад (Husky), чтобы синяя стрелка шла к цели
    linear_velocity_sign: float = 1.0
    # дополнительные опции
    use_default_offset: bool = False

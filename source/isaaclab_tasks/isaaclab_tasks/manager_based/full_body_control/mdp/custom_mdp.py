import torch
from isaaclab_tasks.manager_based.navigation.mdp import (
    JointVelocityAction,
    JointVelocityActionCfg,
    ActionTerm,
)
from isaaclab.utils import configclass

class DiffDriveVelocityAction(JointVelocityAction):
    """ActionTerm для дифференциального привода 4 колес.

    Ожидает action от политики: [v_lin, omega]
    - v_lin: нормированная линейная скорость [-1, 1]
    - omega: нормированная угловая скорость [-1, 1]
    """

    cfg: "DiffDriveVelocityActionCfg"

    def process_actions(self, actions: torch.Tensor):
        """
        Преобразует действия RL в угловые скорости колес.
        Args:
            actions: [num_envs, 2] — команды от политики
        """
        # ограничим диапазон [-1, 1]
        actions = actions.clamp(-1.0, 1.0)

        # масштабируем под физические значения
        v_lin = actions[:, 0:1] * self.cfg.max_linear_speed   # м/с
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
    max_linear_speed: float = 1.0   # м/с
    max_angular_speed: float = 3.0  # рад/с

    # дополнительные опции
    use_default_offset: bool = False

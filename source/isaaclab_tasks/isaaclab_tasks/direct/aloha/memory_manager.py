import torch

class Memory_manager:
    def __init__(self, num_envs: int, embedding_size: int, action_size: int, history_length: int, device: torch.device):
        """
        Инициализация менеджера памяти для хранения эмбеддингов и действий как стека.

        Args:
            num_envs (int): Количество сред.
            embedding_size (int): Размер эмбеддинга (например, 512 для ResNet18).
            action_size (int): Размер действия (например, 2 для линейной и угловой скорости).
            history_length (int): Длина стека (n).
            device (torch.device): Устройство для хранения тензоров (CPU/GPU).
        """
        self.num_envs = num_envs
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.history_length = history_length
        self.device = device
                
        # Создаем индексы для выборки: [0, k, 2k, ..., (m-1)*k]
        m = 4
        k = 4
        indices = torch.arange(0, m * k, k, device="cpu", dtype=torch.long)  # [m]
        # Ограничиваем индексы длиной стека
        self.indices = torch.clamp(indices, 0, self.history_length - 1)

        # Инициализация стека для эмбеддингов: [num_envs, history_length, embedding_size]
        self.embedding_history = torch.zeros((num_envs, history_length, embedding_size), device="cpu")
        self.action_history = torch.zeros((num_envs, history_length, action_size), device="cpu")
        # Флаг, указывающий, что стек еще не заполнен (для первой инициализации)
        self.is_initialized = False

    def update(self, embeddings: torch.Tensor, actions: torch.Tensor):
        """
        Обновление стека: новый эмбеддинг и действие добавляются в начало, старые сдвигаются вправо.

        Args:
            embeddings (torch.Tensor): Эмбеддинги текущего шага, форма [num_envs, embedding_size].
            actions (torch.Tensor): Действия текущего шага, форма [num_envs, action_size].
        """
        # Если стек не инициализирован, заполняем его первым эмбеддингом и действием
        if not self.is_initialized:
            self.embedding_history = embeddings.cpu().unsqueeze(1).expand(-1, self.history_length, -1)
            self.is_initialized = True
        # Сдвигаем историю вправо (удаляем самый старый элемент)
        # Создаем новую историю, объединяя новый эмбеддинг/действие с частью старой истории
        self.embedding_history = torch.cat([embeddings.cpu().unsqueeze(1), self.embedding_history[:, :-1]], dim=1)
        self.action_history = torch.cat([actions.cpu().unsqueeze(1), self.action_history[:, :-1]], dim=1)

    def get_observations(self, m=4, k=4) -> torch.Tensor:
        """
        Получение m эмбеддингов и m действий с частотой k, начиная с последнего (индекс 0).

        Args:
            m (int): Количество возвращаемых эмбеддингов и действий.
            k (int): Шаг выборки (частота).

        Returns:
            torch.Tensor: Тензор формы [num_envs, m * (embedding_size + action_size)] с m эмбеддингами и действиями.
        """
        # Выбираем эмбеддинги и действия для всех сред одновременно
        selected_embeddings = self.embedding_history[:, self.indices].to(self.device)  # [num_envs, m, embedding_size]
        selected_actions = self.action_history[:, self.indices].to(self.device)  # [num_envs, m, action_size]

        # Объединяем эмбеддинги и действия по последней размерности
        combined = torch.cat([selected_embeddings, selected_actions], dim=-1)  # [num_envs, m, embedding_size + action_size]

        # Разворачиваем в плоский вектор
        output = combined.view(self.num_envs, -1).to(self.device)  # [num_envs, m * (embedding_size + action_size)]

        return output

    def reset(self):
        """
        Сброс стека для указанных сред или всех сред.
        """
        self.is_initialized = False
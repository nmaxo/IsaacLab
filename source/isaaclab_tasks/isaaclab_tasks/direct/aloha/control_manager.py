import torch
import math

class VectorizedPurePursuit:
    def __init__(self, num_envs, device='gpu', max_path_length=15, lookahead_distance=0.35, base_linear_velocity=1, max_angular_velocity=1.8, arrival_threshold=0.2):
        """
        Vectorized Pure Pursuit controller for following paths in a vectorized environment.
        
        :param num_envs: Number of environments (agents).
        :param device: Torch device ('cpu' or 'cuda').
        :param max_path_length: Maximum length of paths (pads shorter paths with NaN).
        :param lookahead_distance: Lookahead distance for pure pursuit.
        :param base_linear_velocity: Base linear speed.
        :param max_angular_velocity: Maximum angular speed.
        :param arrival_threshold: Distance threshold to consider path finished.
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_path_length = max_path_length
        self.lookahead_distance = lookahead_distance
        self.base_linear_velocity = base_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.arrival_threshold = arrival_threshold

        # Storage for paths: (num_envs, max_path_length, 2), padded with NaN
        self.paths = torch.full((num_envs, max_path_length, 2), float('nan'), dtype=torch.float32, device=self.device)
        
        # Actual lengths of paths: (num_envs,)
        self.path_lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        
        # Finished flags: (num_envs,)
        self.finished = torch.ones(num_envs, dtype=torch.bool, device=self.device)  # Start as finished (no path)
        self.target_positions = torch.full((num_envs, 2), float('nan'), dtype=torch.float32, device=self.device)

    def update_paths(self, env_indices, new_paths, target_positions):
        """
        Update paths for specific environments that have finished and need new paths.
        
        :param env_indices: List or tensor of environment indices to update.
        :param new_paths: List of new paths, each a list of [x, y] points (or tensor).
        """
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(env_indices, dtype=torch.int64, device=self.device)
        self.target_positions[env_indices] = torch.tensor(target_positions, dtype=torch.float32, device=self.device)
        for i, env_id in enumerate(env_indices):
            path = new_paths[i]
            length = path.shape[0]
            if length > self.max_path_length:
                raise ValueError(f"Path length {length} exceeds max_path_length {self.max_path_length}")
            self.paths[env_id, :length] = path
            self.paths[env_id, length:] = float('nan')
            self.path_lengths[env_id] = length
        self.finished[env_indices] = False

    def compute_controls(self, positions, orientations):
        """
        Compute linear and angular velocities for all environments.
        
        :param positions: Tensor (num_envs, 2) of current [x, y] positions.
        :param orientations: Tensor (num_envs,) of current yaw orientations in radians.
        :return: linear_vels (num_envs,), angular_vels (num_envs,)
        """
        linear_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        angular_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Active environments: have valid path (length >= 2) and not finished
        active = (self.path_lengths >= 2) & ~self.finished
        if not active.any() and not self.finished.any():
            return linear_vels, angular_vels

        # Subset for active envs
        active_indices = torch.where(active)[0]
        num_active = active_indices.shape[0]
        pos = positions[active_indices]
        ori = orientations[active_indices]
        paths_active = self.paths[active_indices]
        path_lens = self.path_lengths[active_indices]

        # Segments
        max_segments = self.max_path_length - 1
        segment_starts = paths_active[:, :-1, :]  # (num_active, max_segments, 2)
        segment_ends = paths_active[:, 1:, :]     # (num_active, max_segments, 2)
        segment_vecs = segment_ends - segment_starts
        segment_lengths = torch.norm(segment_vecs, dim=-1)  # (num_active, max_segments)

        # Mask valid segments
        segment_mask = torch.arange(max_segments, device=self.device).expand(num_active, max_segments) < (path_lens - 1).unsqueeze(1)

        # Compute closest points on all segments
        pos_exp = pos.unsqueeze(1)  # (num_active, 1, 2)
        to_starts = pos_exp - segment_starts
        projs = torch.sum(to_starts * segment_vecs, dim=-1) / (segment_lengths ** 2 + 1e-8)
        projs_clamped = torch.clamp(projs, min=0.0, max=1.0)
        closest_points = segment_starts + segment_vecs * projs_clamped.unsqueeze(-1)
        dists = torch.norm(pos_exp - closest_points, dim=-1)

        # Invalidate non-segments
        dists[~segment_mask] = float('inf')

        # Find closest segment per active env
        min_dists, min_segments = torch.min(dists, dim=1)

        # Get proj and seg len for min segment
        min_projs = projs_clamped[torch.arange(num_active, device=self.device), min_segments]
        min_seg_lengths = segment_lengths[torch.arange(num_active, device=self.device), min_segments]

        # Cumulative arclengths
        padded_segment_lengths = segment_lengths.clone()
        padded_segment_lengths[~segment_mask] = 0.0
        cum_lengths = torch.cat([torch.zeros(num_active, 1, device=self.device), padded_segment_lengths], dim=1)  # (num_active, max_segments + 1)
        cum_lengths = torch.cumsum(cum_lengths, dim=1)

        # Arclength to closest point
        cum_at_min_seg = cum_lengths[torch.arange(num_active, device=self.device), min_segments]
        closest_arclen = cum_at_min_seg + min_projs * min_seg_lengths

        # Target arclength for lookahead
        target_arclen = closest_arclen + self.lookahead_distance

        # Total path arclength
        total_lengths = cum_lengths[torch.arange(num_active, device=self.device), (path_lens - 1).clamp(max=max_segments)]

        # Beyond end?
        is_beyond = target_arclen >= total_lengths

        # Lookahead points
        lookahead_points = torch.zeros(num_active, 2, dtype=torch.float32, device=self.device)

        # Last points
        last_point_indices = (path_lens - 1).clamp(max=self.max_path_length - 1)
        last_points = paths_active[torch.arange(num_active, device=self.device), last_point_indices]

        lookahead_points[is_beyond] = last_points[is_beyond]

        # For not beyond
        not_beyond = ~is_beyond
        num_not_beyond = not_beyond.sum()
        if num_not_beyond > 0:
            target_arclen_nb = target_arclen[not_beyond]
            cum_lengths_nb = cum_lengths[not_beyond]
            segments_nb = torch.searchsorted(cum_lengths_nb, target_arclen_nb.unsqueeze(1)).squeeze(1) - 1
            segments_nb = torch.clamp(segments_nb, min=0, max=max_segments - 1)
            cum_at_seg_nb = cum_lengths_nb[torch.arange(num_not_beyond.item(), device=self.device), segments_nb]
            seg_lengths_nb = padded_segment_lengths[not_beyond, segments_nb]
            fracs_nb = (target_arclen_nb - cum_at_seg_nb) / (seg_lengths_nb + 1e-8)
            fracs_nb = torch.clamp(fracs_nb, min=0.0, max=1.0)
            starts_nb = segment_starts[not_beyond, segments_nb]
            vecs_nb = segment_vecs[not_beyond, segments_nb]
            lookahead_points[not_beyond] = starts_nb + vecs_nb * fracs_nb.unsqueeze(-1)

        # Compute controls
        to_targets = lookahead_points - pos
        target_angles = torch.atan2(to_targets[:, 1], to_targets[:, 0])
        alphas = target_angles - ori
        alphas = (alphas + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
        curvatures = 2 * torch.sin(alphas) / self.lookahead_distance
        ang_vels_active = (curvatures) * self.base_linear_velocity
        ang_vels_active = torch.clamp(ang_vels_active, -self.max_angular_velocity, self.max_angular_velocity)
        # Adjust linear velocity based on turning
        lin_vels_active = self.base_linear_velocity * (1 - torch.abs(ang_vels_active) / self.max_angular_velocity)
        # Check finished
        dists_to_end = torch.norm(pos - last_points, dim=1)
        finished_active = dists_to_end < self.arrival_threshold
        self.finished[active_indices[finished_active]] = True

        # Assign back
        linear_vels[active_indices] = lin_vels_active
        angular_vels[active_indices] = ang_vels_active
        # Check finished for active environments
        # Handle orientation alignment for all finished environments
        just_finished = torch.where(self.finished)[0]
        if len(just_finished) > 0:
            pos_jf = positions[just_finished]
            ori_jf = orientations[just_finished]
            target_pos_jf = self.target_positions[just_finished]
            # Пропустить окружения с невалидными целевыми позициями
            to_targets_jf = target_pos_jf - pos_jf
            target_angles_jf = torch.atan2(to_targets_jf[:, 1], to_targets_jf[:, 0])
            alphas_jf = target_angles_jf - ori_jf
            alphas_jf = (alphas_jf + math.pi) % (2 * math.pi) - math.pi  # Нормализовать в [-π, π]
            signs = torch.sign(alphas_jf)
            signs[signs == 0] = 1
            ang_vels_jf = torch.clamp(signs * 2.0, -self.max_angular_velocity, self.max_angular_velocity)  # Simple proportional control
            
            linear_vels[just_finished] = 0.0  # Нет линейного движения
            angular_vels[just_finished] = ang_vels_jf
            
            # Проверить, завершён ли доворот (ошибка ориентации < 0.05 радиан)
            ori_error = torch.abs(alphas_jf)
            alignment_done = ori_error < 0.1  # Порог ~2.86°
            self.finished[just_finished[alignment_done]] = True  # Оставить True, но можно использовать для внешней логики
        # linear_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # angular_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return linear_vels, angular_vels
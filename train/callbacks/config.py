from __future__ import annotations

from dataclasses import dataclass

@dataclass
class CurriculumStageInfo:
    stage_idx: int
    world_scale: float
    num_drones: int
    num_static_obstacles: int
    num_dynamic_obstacles: int
    stage_timesteps: int
    total_planned_timesteps: int

    @property
    def num_total_obstacles(self) -> int:
        return self.num_static_obstacles + self.num_dynamic_obstacles
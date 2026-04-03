from __future__ import annotations
import json
from pathlib import Path
from envs.createObjects import EnvObject, Dynamic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wrappers.env_3d import DroneSwarm3D

def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "interface" / "configs"

def _load_object_config(config_name: str) -> list[dict]:
    path = _configs_dir() / config_name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data

def get_complex_world() -> dict[str, str]:
    return {
        "wall_holes": "wall_holes.json",
        "two_walls_with_holes": "two_walls_with_holes.json",
        "hallway": "hallway.json",
    }
def apply_object_config(env: DroneSwarm3D, object_configs: list[dict]) -> None:
    """Spawn obstacles from JSON config into the environment."""
    env.remove_all_obstacles()
    env._obstacles = []

    static_count = 0
    dynamic_count = 0

    for cfg in object_configs:
        pos = [float(cfg["x"]), float(cfg["y"]), float(cfg["z"])]
        ori = [0.0, 0.0, float(cfg.get("yaw", 0.0))]
        size = max(0.1, float(cfg.get("size", 1.0)))
        shape = str(cfg.get("shape", "box"))

        if cfg.get("kind") == "dynamic":
            axis = int(cfg.get("axis", 0))
            min_pos = float(cfg.get("min", -1.0))
            max_pos = float(cfg.get("max", 1.0))
            if min_pos > max_pos:
                min_pos, max_pos = max_pos, min_pos
            speed = max(0.0, float(cfg.get("speed", 0.2)))
            direction = -1 if int(cfg.get("direction", 1)) < 0 else 1
            obj = Dynamic(axis, min_pos, max_pos, speed, direction, pos, size, ori)
            dynamic_count += 1
        else:
            obj = EnvObject(pos, size, ori)
            static_count += 1

        obj.create(shape, physics_call=env._call_physics)
        env._obstacles.append(obj)

    env.num_static_obstacles = static_count
    env.num_dynamic_obstacles = dynamic_count


def apply_complex_world(env: DroneSwarm3D, complex_world_name: str) -> None:
    worlds = get_complex_world()
    if complex_world_name not in worlds:
        raise ValueError(f"Unknown world: {complex_world_name}")
    object_configs = _load_object_config(worlds[complex_world_name])
    apply_object_config(env, object_configs)
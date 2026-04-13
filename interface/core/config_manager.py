"""
  Contains no Qt dependencies.
  Do not add any Qt imports here.
"""
import json
import random
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config.defaults import SWARM_EDGE_MARGIN
from envs.createWorld import SUPPORTED_SHAPES, generate_edge_xyz, random_obstacle_config
from interface.types import DroneSpawnConfig, ObjectConfig, SwarmConfig, TargetSpawnConfig

# Contains default colors for swarms, used as defaults and for validation when setting by name. Each entry is RGBA with values in [0, 1].
SWARM_COLOR_PRESETS: Dict[str, List[float]] = {
    "Green":   [0.0, 1.0, 0.0, 0.9],
    "Blue":    [0.0, 0.0, 1.0, 0.9],
    "Yellow":  [1.0, 1.0, 0.0, 0.9],
    "Magenta": [1.0, 0.0, 1.0, 0.9],
    "Cyan":    [0.0, 1.0, 1.0, 0.9],
    "Orange":  [1.0, 0.5, 0.0, 0.9],
    "Purple":  [0.5, 0.0, 1.0, 0.9],
}

AXIS_LABEL_TO_INT: Dict[str, int] = {"x": 0, "y": 1, "z": 2}
AXIS_INT_TO_LABEL: Dict[int, str] = {0: "x", 1: "y", 2: "z"}
KINDS = ("static", "dynamic")


class ConfigManager:
    """Owner of all entity configuration states."""

    def __init__(self, num_swarms: int = 1, drones_per_swarm: int = 1, world_scale: float = 5.0):
        self._lock = threading.Lock()
        self._num_swarms = max(0, int(num_swarms))
        self._drones_per_swarm = max(1, int(drones_per_swarm))
        self._world_scale = max(1.0, float(world_scale))

        self._object_configs: List[ObjectConfig] = []
        self._drone_spawn_configs: List[DroneSpawnConfig] = []
        self._swarm_configs: List[SwarmConfig] = []
        self._target_spawn_configs: List[TargetSpawnConfig] = []

        self._init_swarm_spawn_configs()

    # ---- Properties ----

    @property
    def num_swarms(self) -> int:
        return self._num_swarms

    @property
    def drones_per_swarm(self) -> int:
        return self._drones_per_swarm

    # ---- Getters ----

    def get_object_configs(self) -> List[ObjectConfig]:
        with self._lock:
            return [dict(cfg) for cfg in self._object_configs]

    def get_drone_spawn_configs(self) -> List[DroneSpawnConfig]:
        with self._lock:
            self._rebuild_drone_spawn_configs()
            return [dict(cfg) for cfg in self._drone_spawn_configs]

    #TODO: Reset targets after color or position changes to swarms, or add option to keep them in place?
    def get_target_spawn_configs(self) -> List[TargetSpawnConfig]:
        with self._lock:
            targets = [dict(cfg) for cfg in self._target_spawn_configs]
            # Ensure that targets match colour of swarms
            for i, cfg in enumerate(targets):
                if "color" not in cfg and 0 <= i < len(self._swarm_configs):
                    cfg["color"] = list(
                        self._swarm_configs[i].get("color")
                    )
            return targets

    def get_swarm_configs(self) -> List[SwarmConfig]:
        with self._lock:
            return [dict(cfg) for cfg in self._swarm_configs]

    def get_name(self, mode: str, index: int) -> str:
        """Returns the name of the entity at the given index, or a default if not set."""
        with self._lock:
            if mode == "object_spawns":
                configs, default = self._object_configs, f"Object {index + 1}"
            elif mode == "drone_spawns":
                configs, default = self._drone_spawn_configs, f"Drone {index + 1}"
            else:
                configs, default = self._target_spawn_configs, f"Target {index + 1}"
            if 0 <= index < len(configs):
                return str(configs[index].get("name", default))
        return default

    def count(self, mode: str) -> int:
        """Returns the count of entities for the given mode."""
        with self._lock:
            if mode == "object_spawns":
                return len(self._object_configs)
            elif mode == "drone_spawns":
                return len(self._swarm_configs)
            else:
                return len(self._target_spawn_configs)

    # ---- Setters ----

    def set_object_configs(self, configs: List[ObjectConfig]):
        with self._lock:
            self._object_configs = [
                self._ensure_object_fields(dict(cfg)) for cfg in configs
            ]

    # ---- Count management ----

    def sync_object_count(self, count: int):
        """
        Syncs the count of obstacles to the given count, adding default configs or removing from the end as needed.
        """
        with self._lock:
            current = len(self._object_configs)
            if count > current:
                for i in range(current, count):
                    self._object_configs.append(self._default_object_config(i))
            elif count < current:
                self._object_configs = self._object_configs[:count]

    def sync_target_count(self, count: int):
        """
        Syncs the count of targets to the given count, adding default configs or removing from the end as needed.
        """
        with self._lock:
            current = len(self._target_spawn_configs)
            if count > current:
                for i in range(current, count):
                    self._target_spawn_configs.append(
                        self._default_target_spawn_config(i)
                    )
            elif count < current:
                self._target_spawn_configs = self._target_spawn_configs[:count]

    def sync_swarm_count(self, count: int):
        """
        Syncs the count of swarms to the given count, adding default configs or removing from the end as needed.
        """
        count = max(1, int(count))
        with self._lock:
            self._num_swarms = count

            current = len(self._swarm_configs)
            if count > current:
                for i in range(current, count):
                    self._swarm_configs.append(self._default_swarm_config(i))
            elif count < current:
                self._swarm_configs = self._swarm_configs[:count]

            current_targets = len(self._target_spawn_configs)
            if count > current_targets:
                for i in range(current_targets, count):
                    self._target_spawn_configs.append(
                        self._default_target_spawn_config(i)
                    )
            elif count < current_targets:
                self._target_spawn_configs = self._target_spawn_configs[:count]

            for i, target_cfg in enumerate(self._target_spawn_configs):
                if "name" not in target_cfg:
                    target_cfg["name"] = f"Target Spawn {i + 1}"
                target_cfg["color"] = list(
                    self._swarm_configs[i].get("color", SWARM_COLOR_PRESETS["Green"])
                )

            self._rebuild_drone_spawn_configs()

    def randomize(self):
        """
        Re-randomize obstacle positions, swarm spawn edges, and target positions
        (targets are placed on the edge opposite their swarm).
        Names and colors are preserved.
        """
        with self._lock:
            # Re-randomize each obstacle position, keeping count
            self._object_configs = [
                self._default_object_config(i)
                for i in range(len(self._object_configs))
            ]
            # Re-randomize each swarm edge/position (keep name and color)
            for swarm_cfg in self._swarm_configs:
                edge = random.randint(0, 3)
                x, y, z = generate_edge_xyz(edge, self._world_scale, SWARM_EDGE_MARGIN)
                swarm_cfg["x"] = float(x)
                swarm_cfg["y"] = float(y)
                swarm_cfg["z"] = float(z)
                swarm_cfg["_edge"] = edge
            # Re-generate target positions based on the new swarm edges
            for i, target_cfg in enumerate(self._target_spawn_configs):
                new_cfg = self._default_target_spawn_config(i)
                target_cfg["x"] = new_cfg["x"]
                target_cfg["y"] = new_cfg["y"]
                target_cfg["z"] = new_cfg["z"]
            self._rebuild_drone_spawn_configs()

    def remove_at(self, mode: str, index: int) -> bool:
        """
        Removes the configuration at the given index for the specified mode.
        
        Returns:
          True if the removal was successful, False otherwise.
        """
        with self._lock:
            if mode == "object_spawns":
                if 0 <= index < len(self._object_configs):
                    del self._object_configs[index]
                    return True
            elif mode == "drone_spawns":
                if 0 <= index < len(self._swarm_configs):
                    del self._swarm_configs[index]
                    return True
            elif mode == "target_spawns":
                if 0 <= index < len(self._target_spawn_configs):
                    del self._target_spawn_configs[index]
                    return True
            else:
                return False

    # ---- Field setters ----

    def set_object_field(self, index: int, key: str, raw_value) -> bool:
        """
        Sets the given field of the object at the specified index to the provided value.
        The raw_value is coerced to the appropriate type based on the existing value for that field, or defaults if not set.
        """
        with self._lock:
            if not (0 <= index < len(self._object_configs)):
                return False
            cfg = self._object_configs[index]
            old = cfg.get(key)
            if key == "axis":   # Special handling for axis to allow both int and string inputs
                raw_str = str(raw_value).strip().lower()
                value = AXIS_LABEL_TO_INT.get(raw_str, self._coerce_int(raw_value, old))
                value = max(0, min(2, int(value)))  # 0 <= value <= 2
            else:               # For other keys, determine type based on existing value or defaults
                value = (
                    self._coerce_float(raw_value, old) if isinstance(old, float)
                    else self._coerce_int(raw_value, old) if isinstance(old, int)
                    else (str(raw_value).strip()) or str(old)
                )

            if key == "shape" and value not in SUPPORTED_SHAPES:
                value = old 
            if key == "kind" and value not in KINDS:
                value = old
            if key == "size":
                value = max(0.1, float(value))  # 0.1 <= size
            if key == "speed":
                value = max(0.0, float(value))
            if key == "direction":              # direction is absolute, either -1 or 1
                value = -1 if int(value) < 0 else 1

            cfg[key] = value
        return True

    def set_spawn_field(self, mode: str, index: int, key: str, raw_value) -> bool:
        """
        Sets the given field of the target at the specified index to the provided value.
        """
        with self._lock:
            configs = (
                self._drone_spawn_configs
                if mode == "drone_spawns"
                else self._target_spawn_configs
            )
            if not (0 <= index < len(configs)):
                return False
            old = configs[index].get(key)
            if key in ("x", "y", "z"):
                value = self._coerce_float(raw_value, old)
            else:
                value = str(raw_value).strip() if str(raw_value).strip() else str(old)
            configs[index][key] = value
        return True

    def set_swarm_field(self, swarm_idx: int, key: str, raw_value) -> bool:
        """
        Sets the given field of the swarm at the specified index to the provided value.
        """
        with self._lock:
            if not (0 <= swarm_idx < len(self._swarm_configs)):
                return False
            swarm_cfg = self._swarm_configs[swarm_idx]
            if key in ("x", "y", "z"):
                value = self._coerce_float(raw_value, float(swarm_cfg.get(key, 0.0)))
                swarm_cfg[key] = value
            elif key == "color_name":
                color_name = str(raw_value).strip()
                if color_name not in SWARM_COLOR_PRESETS:
                    color_name = self._default_swarm_color_name(swarm_idx)
                swarm_cfg["color_name"] = color_name
                swarm_cfg["color"] = list(SWARM_COLOR_PRESETS[color_name])
            else:
                value = str(raw_value).strip()
                if value:
                    swarm_cfg[key] = value

            if 0 <= swarm_idx < len(self._target_spawn_configs):
                self._target_spawn_configs[swarm_idx]["color"] = list(
                    swarm_cfg.get("color", SWARM_COLOR_PRESETS["Green"])
                )
            self._rebuild_drone_spawn_configs()
        return True

    def set_drone_offset_field(self, drone_index: int, key: str, raw_value) -> bool:
        """
        Sets the given local offset field of the drone at the specified index to the provided value.
        """
        with self._lock:
            if not (0 <= drone_index < len(self._drone_spawn_configs)):
                return False
            current = float(self._drone_spawn_configs[drone_index].get(key, 0.0))
            self._drone_spawn_configs[drone_index][key] = self._coerce_float(
                raw_value, current
            )
            self._rebuild_drone_spawn_configs()
        return True

    # ---- Persistence ----

    def save_json(self, path: Path):
        """
        Saves the current object configurations to a JSON file at the specified path.
        """

        configs = self.get_object_configs()
        with path.open("w", encoding="utf-8") as f:
            json.dump(configs, f, indent=4)

    def load_json(self, path: Path) -> Optional[List[ObjectConfig]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [
                    self._ensure_object_fields(dict(cfg))
                    for cfg in data
                    if isinstance(cfg, dict)
                ]
        except Exception:
            return None
        return None

    # ---- Defaults ----

    def _default_object_config(self, index: int) -> ObjectConfig:
        return random_obstacle_config(index, self._world_scale)

    def _default_target_spawn_config(self, index: int) -> TargetSpawnConfig:
        if 0 <= index < len(self._swarm_configs):
            swarm_edge = self._swarm_configs[index].get("_edge", random.randint(0, 3))
        else:
            swarm_edge = random.randint(0, 3)
        opposite_edge = {0: 1, 1: 0, 2: 3, 3: 2}[swarm_edge]
        x, y, z = generate_edge_xyz(opposite_edge, self._world_scale, SWARM_EDGE_MARGIN)
        return {
            "name": f"Target Spawn {index + 1}",
            "x": x,
            "y": y,
            "z": z,
        }

    def _default_swarm_color_name(self, index: int) -> str:
        names = list(SWARM_COLOR_PRESETS.keys())
        return names[index % len(names)]

    def _default_swarm_config(self, index: int) -> SwarmConfig:
        color_name = self._default_swarm_color_name(index)
        edge = random.randint(0, 3)
        x, y, z = generate_edge_xyz(edge, self._world_scale, SWARM_EDGE_MARGIN)
        return {
            "name": f"Swarm {index + 1}",
            "x": x,
            "y": y,
            "z": z,
            "color_name": color_name,
            "color": list(SWARM_COLOR_PRESETS[color_name]),
            "_edge": edge,
        }

    def _default_drone_offset(self, index: int) -> dict:
        row = index // 3
        col = index % 3
        return {
            "local_x": round((col - 1) * 0.35, 2),
            "local_y": round((row - 1) * 0.35, 2),
            "local_z": 0.0,
        }

    def _ensure_object_fields(self, config: dict) -> ObjectConfig:
        """ Ensures that the given object config has all required fields, filling in defaults as needed."""
        defaults = {
            "name": "Object",
            "kind": "static",
            "shape": "box",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "size": 0.5,
            "yaw": 0.0,
            "axis": 0,
            "min": -1.0,
            "max": 1.0,
            "speed": 0.2,
            "direction": 1,
        }
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
        return config

    def ensure_spawn_fields(self, config: dict, prefix: str) -> dict:
        defaults = {"name": f"{prefix} count", "x": 0.0, "y": 0.0, "z": 0.5}
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
        return config

    # ---- Internal rebuild ----

    def _init_swarm_spawn_configs(self):
        self._swarm_configs = [
            self._default_swarm_config(i) for i in range(self._num_swarms)
        ]
        self._target_spawn_configs = [
            self._default_target_spawn_config(i) for i in range(self._num_swarms)
        ]
        self._drone_spawn_configs = []
        for swarm_idx in range(self._num_swarms):
            for drone_local_idx in range(self._drones_per_swarm):
                offset = self._default_drone_offset(drone_local_idx)
                self._drone_spawn_configs.append(
                    {
                        "name": f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}",
                        "swarm_index": swarm_idx,
                        **offset,
                    }
                )
        self._rebuild_drone_spawn_configs()

    def _rebuild_drone_spawn_configs(self):
        """Must be called while holding self._lock."""
        expected = self._num_swarms * self._drones_per_swarm

        if len(self._drone_spawn_configs) > expected:
            self._drone_spawn_configs = self._drone_spawn_configs[:expected]
        elif len(self._drone_spawn_configs) < expected:
            for idx in range(len(self._drone_spawn_configs), expected):
                swarm_idx = idx // self._drones_per_swarm
                drone_local_idx = idx % self._drones_per_swarm
                offset = self._default_drone_offset(drone_local_idx)
                self._drone_spawn_configs.append(
                    {
                        "name": f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}",
                        "swarm_index": swarm_idx,
                        **offset,
                    }
                )

        for idx, cfg in enumerate(self._drone_spawn_configs):
            swarm_idx = int(cfg.get("swarm_index", idx // self._drones_per_swarm))
            swarm_idx = max(0, min(self._num_swarms - 1, swarm_idx))
            drone_local_idx = idx % self._drones_per_swarm
            swarm_cfg = self._swarm_configs[swarm_idx]

            local_x = float(cfg.get("local_x", 0.0))
            local_y = float(cfg.get("local_y", 0.0))
            local_z = float(cfg.get("local_z", 0.0))

            cfg["swarm_index"] = swarm_idx
            cfg["local_x"] = local_x
            cfg["local_y"] = local_y
            cfg["local_z"] = local_z
            cfg["name"] = f"Swarm {swarm_idx + 1} Drone {drone_local_idx + 1}"
            cfg["x"] = float(swarm_cfg.get("x", 0.0)) + local_x
            cfg["y"] = float(swarm_cfg.get("y", 0.0)) + local_y
            cfg["z"] = float(swarm_cfg.get("z", 0.8)) + local_z
            cfg["color"] = list(swarm_cfg.get("color", SWARM_COLOR_PRESETS["Green"]))

    # ---- Utilities ----

    @staticmethod
    def _coerce_float(text, fallback: float) -> float:
        try:
            return float(text)
        except Exception:
            return float(fallback)

    @staticmethod
    def _coerce_int(text, fallback: int) -> int:
        try:
            return int(float(text))
        except Exception:
            return int(fallback)

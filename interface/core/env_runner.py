# -*- coding: utf-8 -*-

import random
import threading
import time
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from envs.createObjects import Dynamic, EnvObject, safe_unit
from envs.createWorld import SUPPORTED_SHAPES
from interface.core.frame_buffer import FrameBuffer
from wrappers.env_3d import DroneSwarm3D as _DroneSwarm3D
from config.defaults import GOAL_REACH_DISTANCE, MIN_SPAWN_DIST_FROM_DRONE, SWARM_MIN_SEPARATION


class EnvRunner:
    def __init__(
        self,
        frame_buffer: FrameBuffer,
        object_configs_getter: Callable,
        drone_spawn_configs_getter: Callable,
        target_spawn_configs_getter: Callable,
        on_state: Callable,
        on_stopped: Callable,
        on_error: Callable,
        num_drones: int = 1,
        max_steps: int = 5000,
        lidar: bool = False,
        render_mode: str = "human",
        model_path: str = "",
        world_scale: float = 5.0,
        num_swarms: int = 1,
        seed: Optional[int] = None,
    ):
        self._frame_buffer = frame_buffer
        self._get_object_configs = object_configs_getter
        self._get_drone_spawn_configs = drone_spawn_configs_getter
        self._get_target_spawn_configs = target_spawn_configs_getter
        self._on_state = on_state
        self._on_stopped = on_stopped
        self._on_error = on_error

        self._num_drones = max(1, int(num_drones))
        self._max_steps = max(1, int(max_steps))
        self._lidar = bool(lidar)
        self._render_mode = render_mode
        self._world_scale = max(1.0, float(world_scale))
        self._num_swarms = max(1, int(num_swarms))
        self._seed = int(seed) if seed is not None else None

        self._running = False
        self._sim_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._speed_lock = threading.Lock()
        self._speed_multiplier = 1.0
        self._env_lock = threading.Lock()
        self._current_env = None
        self._episode = 0
        self._step = 0

        self._model_path = str(model_path or "").strip()
        self._model = None
        self._model_backend: Optional[str] = None
        self._model_lock = threading.Lock()

        self._last_warning: str = ""

        self._load_model(report_error=False)

    # ---- Public  ----

    @property
    def running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._frame_buffer.clear()
        self._close_current_env()
        self._running = True
        self._stop_event.clear()
        self._restart_event.clear()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

    def close(self):
        self.stop()
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=2.0)
        self._close_current_env()

    def request_restart(self):
        if self._running:
            self._restart_event.set()

    def set_model_path(self, model_path: str) -> Tuple[bool, str]:
        self._model_path = str(model_path or "").strip()
        return self._load_model(report_error=True)

    def get_model_display_name(self) -> str:
        if not self._model_path:
            return "Heuristic"
        return Path(self._model_path).name

    def set_speed_multiplier(self, speed_multiplier: float):
        with self._speed_lock:
            self._speed_multiplier = max(0.1, float(speed_multiplier))

    def get_speed_multiplier(self) -> float:
        with self._speed_lock:
            return float(self._speed_multiplier)

    def set_num_swarms(self, num_swarms: int):
        self._num_swarms = max(1, int(num_swarms))

    def get_replay_frame_count(self) -> int:
        return self._frame_buffer.get_count()

    def get_replay_index(self) -> int:
        return self._frame_buffer.get_index()

    def seek_replay_frame(self, index: int) -> bool:
        with self._env_lock:
            env_ref = self._current_env
        return self._frame_buffer.seek(index, env_ref)

    def get_replay_frame_meta(self, index: int):
        return self._frame_buffer.get_meta(index)

    # ---- Simulation loop ----

    def _simulation_loop(self):
        """Main simulation loop running in the background thread."""
        current_env = None
        try:
            while not self._stop_event.is_set(): # Loop over episodes
                configs = self._get_object_configs()
                if current_env is None:
                    current_env = self._make_env(configs)

                with self._env_lock:
                    self._current_env = current_env

                obs_dict, _ = current_env.reset(seed=random.randint(0, 2**31 - 1)) # Reset env with a random seed for variability
                self._last_warning = "" 
                self._apply_object_configs(current_env, configs)
                self._apply_spawn_configs(
                    current_env,
                    self._get_drone_spawn_configs() if callable(self._get_drone_spawn_configs) else [],
                    self._get_target_spawn_configs() if callable(self._get_target_spawn_configs) else [],
                )
                # print("Env made")
                self._episode += 1
                self._step = 0
                done = {agent: False for agent in current_env.possible_agents} 
                trunc = {agent: False for agent in current_env.possible_agents} 
                restart_requested = False 
                self._frame_buffer.record(current_env, self._episode, self._step) # Record initial state

                while not self._stop_event.is_set(): # Loop over steps within the episode
                    if self._restart_event.is_set():
                        restart_requested = True
                        self._restart_event.clear()
                        break

                    act_dict = self._build_action_dict(obs_dict) 
                    obs_dict, _, done, trunc, _ = current_env.step(act_dict)
                    self._step += 1
                    self._frame_buffer.record(current_env, self._episode, self._step) # Record after action
                    state = {
                        "episode": self._episode,
                        "step": self._step,
                        "drones": len(getattr(current_env, "agents", [])),
                        "obstacles": len(getattr(current_env, "_obstacles", [])),
                        "speed_multiplier": self.get_speed_multiplier(),
                        "recorded_frames": self.get_replay_frame_count(),
                        "replay_index": self.get_replay_index(),
                    }
                    if self._last_warning:
                        state["warning"] = self._last_warning
                    self._on_state(state)
                    time.sleep((1 / 30) / self.get_speed_multiplier())

                    if all(done[a] or trunc[a] for a in done): # Episode ends when all agents are done or truncated and input is required to continue
                        input("Simulation ended. Press Enter to continue.")
                        break

                if restart_requested:
                    current_env.close()
                    with self._env_lock:
                        self._current_env = None
                    current_env = None
                    continue

                if self._stop_event.is_set():
                    break

        except Exception as exc:
            print("ERROR:", exc)
            self._on_error(str(exc))
        finally:
            self._running = False
            self._on_stopped()

    # ---- Environment construction ----

    def _warn(self, message: str) -> None:
        """Store a placement warning; it will be included in the next telemetry state."""
        self._last_warning = message

    def _make_env(self, configs):
        static_count = sum(1 for cfg in configs if cfg.get("kind") == "static")
        dynamic_count = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")
        return _DroneSwarm3D(
            num_drones_per_swarm=self._num_drones,
            num_static_obstacles=static_count,
            num_dynamic_obstacles=dynamic_count,
            max_steps=self._max_steps,
            num_swarms=self._num_swarms,
            lidar=self._lidar,
            render_mode=self._render_mode,
            seed=self._seed,
            world_scale=self._world_scale,
        )

    def _close_current_env(self):
        with self._env_lock:
            env_ref = self._current_env
            self._current_env = None
        if env_ref is not None:
            try:
                env_ref.close()
            except Exception:
                pass

    def _apply_object_configs(self, env_instance, configs):
        p = getattr(env_instance, "_p", None)
        if p is None:
            return

        for obj in list(getattr(env_instance, "_obstacles", [])):
            body_id = getattr(obj, "body_id", None)
            if body_id is None:
                continue
            try:
                env_instance._call_physics(p.removeBody, body_id)
            except Exception:
                continue

        new_obstacles = []
        for cfg in configs:
            pos = [float(cfg["x"]), float(cfg["y"]), float(cfg["z"])]
            ori = [0.0, 0.0, float(cfg["yaw"])]
            size = max(0.1, float(cfg["size"]))
            shape = str(cfg.get("shape", "box"))
            if shape not in SUPPORTED_SHAPES:
                shape = "box"

            if cfg.get("kind") == "dynamic":
                axis = int(cfg.get("axis", 0))
                min_pos = float(cfg.get("min", -1.0))
                max_pos = float(cfg.get("max", 1.0))
                if min_pos > max_pos:
                    min_pos, max_pos = max_pos, min_pos
                speed = max(0.0, float(cfg.get("speed", 0.2)))
                direction = -1 if int(cfg.get("direction", 1)) < 0 else 1
                obj = Dynamic(axis, min_pos, max_pos, speed, direction, pos, size, ori)
            else:
                obj = EnvObject(pos, size, ori)

            try:
                obj.create(shape, physics_call=env_instance._call_physics)
                new_obstacles.append(obj)
            except Exception:
                continue

        env_instance._obstacles = new_obstacles
        env_instance.num_static_obstacles = sum(
            1 for cfg in configs if cfg.get("kind") == "static"
        )
        env_instance.num_dynamic_obstacles = sum(
            1 for cfg in configs if cfg.get("kind") == "dynamic"
        )

        # --- TODO: validate user-placed obstacle positions ---


    def _apply_spawn_configs(self, env_instance, drone_spawns, target_spawns):
        p = getattr(env_instance, "_p", None)
        if p is None:
            return

        def _apply_body_color(body_id: int, rgba):
            rgba = [max(0.0, min(1.0, float(c))) for c in rgba]
            try:
                num_joints = env_instance._call_physics(p.getNumJoints, int(body_id))
            except Exception:
                num_joints = 0
            for link_idx in range(-1, int(num_joints)):
                try:
                    env_instance._call_physics(
                        p.changeVisualShape,
                        int(body_id),
                        int(link_idx),
                        rgbaColor=rgba,
                    )
                except Exception:
                    continue

        drone_ids = list(getattr(env_instance, "_drone_ids", []))
        if drone_ids and drone_spawns:
            for i, body_id in enumerate(drone_ids[: len(drone_spawns)]):
                cfg = drone_spawns[i]
                pos = [
                    float(cfg.get("x", 0.0)),
                    float(cfg.get("y", 0.0)),
                    float(cfg.get("z", 0.8)),
                ]
                try:
                    env_instance._call_physics(
                        p.resetBasePositionAndOrientation,
                        int(body_id),
                        pos,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    env_instance._call_physics(
                        p.resetBaseVelocity,
                        int(body_id),
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    )
                    if "color" in cfg:
                        _apply_body_color(int(body_id), cfg["color"])
                except Exception:
                    continue

        seen: set = set()
        unique_targets = []
        for tgt in list(getattr(env_instance, "_targets", []) or []):
            body_id = getattr(tgt, "body_id", None)
            if body_id is None or body_id in seen:
                continue
            seen.add(body_id)
            unique_targets.append(tgt)

        if unique_targets and target_spawns:
            for i, target in enumerate(unique_targets[: len(target_spawns)]):
                cfg = target_spawns[i]
                pos = [
                    float(cfg.get("x", 0.0)),
                    float(cfg.get("y", 2.0)),
                    float(cfg.get("z", 0.5)),
                ]
                body_id = getattr(target, "body_id", None)
                if body_id is None:
                    continue
                try:
                    env_instance._call_physics(
                        p.resetBasePositionAndOrientation,
                        int(body_id),
                        pos,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    if hasattr(target, "start_pos"):
                        target.start_pos = np.array(pos, dtype=np.float32)
                    if "color" in cfg:
                        _apply_body_color(int(body_id), cfg["color"])
                        if hasattr(target, "color"):
                            target.color = cfg["color"]
                except Exception:
                    continue

                          # --- TODO: validate user-placed obstacle positions ---



    # ---- Model loading ----

    def _load_model(self, report_error: bool = True) -> Tuple[bool, str]:
        if not self._model_path:
            with self._model_lock:
                self._model = None
                self._model_backend = None
            return True, "No model selected."

        path = Path(self._model_path)
        if not path.is_file():
            msg = f"File not found: {path}"
            if report_error:
                self._on_error(msg)
            return False, msg

        try:
            from stable_baselines3 import PPO, SAC
        except Exception:
            msg = "Failed to import stable_baselines3"
            if report_error:
                self._on_error(msg)
            return False, msg

        errors = []
        for name, cls in (("PPO", PPO), ("SAC", SAC)):
            try:
                model = cls.load(str(path))
                with self._model_lock:
                    self._model = model
                    self._model_backend = name
                return True, f"Loaded {path.name} ({name})"
            except Exception as e:
                errors.append(f"{name}: {e}")

        msg = "All backends failed:\n" + "\n".join(errors)
        if report_error:
            self._on_error(msg)
        return False, msg

    # ---- Action selection ----

    def _heuristic_action(self, obs: np.ndarray) -> np.ndarray:
        to_target = obs[6:9]
        to_centroid = obs[12:15]
        dir_vec = 0.8 * safe_unit(to_target) + 0.2 * safe_unit(to_centroid)
        dir_vec = safe_unit(dir_vec)
        speed = float(np.clip(np.linalg.norm(to_target) / 5.0, 0.05, 1.0))
        u3 = speed * 2.0 - 1.0
        return np.array([dir_vec[0], dir_vec[1], dir_vec[2], u3], dtype=np.float32)

    def _predict_action(self, obs: np.ndarray) -> np.ndarray:
        with self._model_lock:
            model = self._model
        if model is None:
            return self._heuristic_action(obs)
        try:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size < 4:
                return self._heuristic_action(obs)
            return action[:4]
        except Exception:
            return self._heuristic_action(obs)

    def _build_action_dict(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {agent: self._predict_action(obs) for agent, obs in obs_dict.items()}

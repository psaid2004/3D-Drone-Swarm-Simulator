
import random
import threading
import time
import importlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from envs.createObjects import Dynamic, EnvObject
from wrappers.env_3d import DroneSwarm3D as env
from envs.createWorld import SUPPORTED_SHAPES 



class EnvironmentViewer:
    def __init__(
        self,
        object_configs_getter,
        on_state,
        on_stopped,
        on_error,
        num_drones=0,
        max_steps=5000,
        lidar=False,
        render_mode="human",
        model_path="",
    ):
        #### Initialize attributes
        # Basic attributes
        self._get_object_configs = object_configs_getter
        self._on_state = on_state
        self._on_stopped = on_stopped
        self._on_error = on_error
        self._num_drones = max(1, int(num_drones))
        self._max_steps = max(1, int(max_steps))
        self._lidar = bool(lidar)
        self._render_mode = render_mode

        # Simulation control attributes
        self._running = False
        self._sim_thread = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._speed_lock = threading.Lock()
        self._speed_multiplier = 1.0
        self._env_lock = threading.Lock()
        self._current_env = None
        self._episode = 0
        self._step = 0

        # Recording attributes
        self._record_lock = threading.Lock()
        self._recorded_frames: List[Dict[str, Any]] = []
        self._replay_index = 0
        self._max_recorded_frames = 8000

        ## Model management
        self._model_path = str(model_path or "").strip()
        self._model = None
        self._model_backend = None
        self._model_lock = threading.Lock()

        self._load_model(report_error=True)

    @property
    def running(self):
        return self._running

    def start(self):
        if self._running:
            return
        self._clear_recording()
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

    def set_model_path(self, model_path: str):
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

    def get_replay_frame_count(self) -> int:
        with self._record_lock:
            return len(self._recorded_frames)

    def get_replay_index(self) -> int:
        with self._record_lock:
            return int(self._replay_index)

    def seek_replay_frame(self, index: int) -> bool:
        with self._record_lock:
            if not self._recorded_frames:
                return False
            bounded_index = max(0, min(int(index), len(self._recorded_frames) - 1))
            frame = dict(self._recorded_frames[bounded_index])

        with self._env_lock:
            env_ref = self._current_env
            if env_ref is None:
                return False
            self._apply_frame(env_ref, frame)

        with self._record_lock:
            self._replay_index = bounded_index

        return True

    def _simulation_loop(self):
        current_env = None
        try:
            while not self._stop_event.is_set():
                configs = self._get_object_configs()
                current_env = self._make_env(configs)

                with self._env_lock:
                    self._current_env = current_env

                obs_dict, _ = current_env.reset(seed=random.randint(0, 2**31 - 1))
                self._apply_object_configs(current_env, configs)
                # print("Env made")
                self._episode += 1
                self._step = 0
                done = {agent: False for agent in current_env.possible_agents}
                trunc = {agent: False for agent in current_env.possible_agents}
                restart_requested = False
                self._record_frame(current_env)

                while not self._stop_event.is_set():

                    if self._restart_event.is_set():
                        restart_requested = True
                        self._restart_event.clear()
                        break
                    
                    act_dict = self._build_action_dict(obs_dict)
                    obs_dict, _, done, trunc, _ = current_env.step(act_dict)
                    self._step += 1
                    self._record_frame(current_env)
                    self._on_state(
                        {
                            "episode": self._episode,
                            "step": self._step,
                            "drones": len(getattr(current_env, "agents", [])),
                            "obstacles": len(getattr(current_env, "_obstacles", [])),
                            "speed_multiplier": self.get_speed_multiplier(),
                            "recorded_frames": self.get_replay_frame_count(),
                            "replay_index": self.get_replay_index(),
                        }
                    )
                    time.sleep((1 / 30) / self.get_speed_multiplier())

                    if all(done[a] or trunc[a] for a in done):
                        break

                if restart_requested:
                    current_env.close()
                    with self._env_lock:
                        self._current_env = None
                    current_env = None
                    continue

                if self._stop_event.is_set():
                    break

                current_env.close()
                with self._env_lock:
                    self._current_env = None
                current_env = None

        except Exception as exc:
            print("ERROR: ", exc, "  ", str(exc))
            self._on_error(str(exc))
        finally:
            self._running = False
            self._on_stopped()

    def _make_env(self, configs):
        static_count = sum(1 for cfg in configs if cfg.get("kind") == "static")
        dynamic_count = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")

        return env(
            num_drones_per_swarm=self._num_drones,
            num_static_obstacles=static_count,
            num_dynamic_obstacles=dynamic_count,
            max_steps=self._max_steps,
            lidar=self._lidar,
            render_mode=self._render_mode,
            world_scale=5
        )

    ## Set current env to None close if still existing
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

        old_obstacles = list(getattr(env_instance, "_obstacles", []))
        for obj in old_obstacles:
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
        env_instance.num_static_obstacles = sum(1 for cfg in configs if cfg.get("kind") == "static")
        env_instance.num_dynamic_obstacles = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")

    ################ Model Load logic ################

    def _load_model(self, report_error: bool = True):
        ## No model specified, use heuristic
        if not self._model_path:
            with self._model_lock:
                self._model = None
                self._model_backend = None
            return True, "No model selected."

        ## Attempt to load model from specified path
        model_path = Path(self._model_path)
        if not model_path.is_file():
            with self._model_lock:
                self._model = None
                self._model_backend = None
            message = f"Model load failed: file not found ({model_path})."
            if report_error:
                self._on_error(message)
            return False, message
        
        # Try loading with stable-baselines3 backends
        try:
            sb3 = importlib.import_module("stable_baselines3")
            PPO = getattr(sb3, "PPO")
            SAC = getattr(sb3, "SAC")
        except Exception as exc:
            with self._model_lock:
                self._model = None
                self._model_backend = None
            message = "Model load failed: error importing stable-baselines3."
            if report_error:
                self._on_error(message)
            return False, message

        # Try loading with each backend, return on first success
        load_attempts = (("PPO", PPO), ("SAC", SAC))
        load_failures = []
        for backend_name, backend_cls in load_attempts:
            try:
                loaded_model = backend_cls.load(str(model_path))
                with self._model_lock:
                    self._model = loaded_model
                    self._model_backend = backend_name
                return True, f"Loaded model: {model_path.name} ({backend_name})"
            except Exception as exc:
                load_failures.append(f"{backend_name}: {exc}")
                continue
        # All loads failed
        with self._model_lock:
            self._model = None
            self._model_backend = None
        message = "Failed to load models."
        if report_error:
            self._on_error(message)
        return False, message

    def _heuristic_action(self, obs: np.ndarray) -> np.ndarray:
        to_target = obs[6:9]
        to_centroid = obs[12:15]
        dir_vec = 0.8 * _safe_unit(to_target) + 0.2 * _safe_unit(to_centroid)
        dir_vec = _safe_unit(dir_vec)
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
        act_dict: Dict[str, np.ndarray] = {}
        for agent, obs in obs_dict.items():
            act_dict[agent] = self._predict_action(obs)
        return act_dict

    ################ Model Load logic ################


    ################ Recording and replay logic ################

    def _clear_recording(self):
        with self._record_lock:
            self._recorded_frames = []
            self._replay_index = 0

    def _capture_frame(self, env_instance) -> Dict[str, Any]:
        drone_states: List[Dict[str, Any]] = []
        obstacle_states: List[Dict[str, Any]] = []
        target_states: List[Dict[str, Any]] = []

        for body_id in list(getattr(env_instance, "_drone_ids", [])):
            state = self._capture_body_state(env_instance, body_id)
            if state is not None:
                drone_states.append(state)

        for obstacle in list(getattr(env_instance, "_obstacles", [])):
            body_id = getattr(obstacle, "body_id", None)
            if body_id is None:
                continue
            state = self._capture_body_state(env_instance, body_id)
            if state is not None:
                obstacle_states.append(state)

        seen_targets = set()
        for target in list(getattr(env_instance, "_targets", []) or []):
            body_id = getattr(target, "body_id", None)
            if body_id is None or body_id in seen_targets:
                continue
            seen_targets.add(body_id)
            state = self._capture_body_state(env_instance, body_id)
            if state is not None:
                target_states.append(state)

        return {
            "episode": int(self._episode),
            "step": int(self._step),
            "drones": drone_states,
            "obstacles": obstacle_states,
            "targets": target_states,
        }
    
    def _record_frame(self, env_instance):
        frame = self._capture_frame(env_instance)
        with self._record_lock:
            self._recorded_frames.append(frame)
            if len(self._recorded_frames) > self._max_recorded_frames:
                overflow = len(self._recorded_frames) - self._max_recorded_frames
                self._recorded_frames = self._recorded_frames[overflow:]
            self._replay_index = max(0, len(self._recorded_frames) - 1)

    def get_replay_frame_meta(self, index: int) -> Optional[Dict[str, int]]:
        with self._record_lock:
            if not self._recorded_frames:
                return None
            bounded_index = max(0, min(int(index), len(self._recorded_frames) - 1))
            frame = self._recorded_frames[bounded_index]
            return {
                "episode": int(frame.get("episode", 0)),
                "step": int(frame.get("step", 0)),
            }

    def _capture_body_state(self, env_instance, body_id: int) -> Optional[Dict[str, Any]]:
        p = getattr(env_instance, "_p", None)
        if p is None:
            return None
        try:
            position, orientation = env_instance._call_physics(p.getBasePositionAndOrientation, int(body_id))
            linear_velocity, angular_velocity = env_instance._call_physics(p.getBaseVelocity, int(body_id))
            return {
                "body_id": int(body_id),
                "position": [float(v) for v in position],
                "orientation": [float(v) for v in orientation],
                "linear_velocity": [float(v) for v in linear_velocity],
                "angular_velocity": [float(v) for v in angular_velocity],
            }
        except Exception:
            return None

    def _apply_frame(self, env_instance, frame: Dict[str, Any]):
        p = getattr(env_instance, "_p", None)
        if p is None:
            return

        def restore(state: Dict[str, Any]):
            body_id = int(state.get("body_id", -1))
            if body_id < 0:
                return
            position = state.get("position", [0.0, 0.0, 0.0])
            orientation = state.get("orientation", [0.0, 0.0, 0.0, 1.0])
            linear_velocity = state.get("linear_velocity", [0.0, 0.0, 0.0])
            angular_velocity = state.get("angular_velocity", [0.0, 0.0, 0.0])
            try:
                env_instance._call_physics(
                    p.resetBasePositionAndOrientation,
                    body_id,
                    position,
                    orientation,
                )
                env_instance._call_physics(
                    p.resetBaseVelocity,
                    body_id,
                    linear_velocity,
                    angular_velocity,
                )
            except Exception:
                return

        for state in frame.get("drones", []):
            restore(state)
        for state in frame.get("obstacles", []):
            restore(state)
        for state in frame.get("targets", []):
            restore(state)

    ################ Recording and replay logic ################


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm
# -*- coding: utf-8 -*-

import random
import threading
import time
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from config.defaults import POS_SLICE
from envs.createObjects import Dynamic, EnvObject
from envs.createWorld import SUPPORTED_SHAPES
from interface.core.frame_buffer import FrameBuffer, get_unique_targets
from wrappers.env_3d import DroneSwarm3D as _DroneSwarm3D

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
        self._frame_buffer = frame_buffer   # Buffer to record frames for replay
        self._get_object_configs = object_configs_getter    # object = obstacles
        self._get_drone_spawn_configs = drone_spawn_configs_getter  # Callback to send state updates to the UI
        self._get_target_spawn_configs = target_spawn_configs_getter  
        self._on_state = on_state   # Callback to send state updates to the UI    
        self._on_stopped = on_stopped   # Callback when the simulation loop has fully stopped
        self._on_error = on_error   # Callback to report errors to the UI

        self._num_drones = max(1, min(10, int(num_drones)))
        self._max_steps = max(1, int(max_steps))
        self._lidar = bool(lidar)
        self._render_mode = render_mode
        self._world_scale = max(1.0, float(world_scale))
        self._num_swarms = max(1, min(6, int(num_swarms)))
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

        self._model_path = str(model_path or "").strip()    # Path to the RL model file
        self._model = None  # Loaded RL model instance (e.g. stable_baselines3 model), or None if no model is loaded
        self._model_backend: Optional[str] = None   # Name of the backend used to load the model (e.g. "PPO", "SAC"), or None if no model is loaded
        self._model_lock = threading.Lock()

        self._last_warning: str = ""

        self._load_model(report_error=False)

    # ---- Public  ----

    @property
    def running(self) -> bool:
        """
        Return whether the simulation is currently running.
        """
        return self._running

    @property
    def has_model(self) -> bool:
        """Return True if a model is currently loaded."""
        with self._model_lock:
            return self._model is not None

    def start(self):
        """ 
        Start the simulation.
        If the simulation is already running, this method does nothing.
        """
        if self._running:
            return
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._stop_event.set()  # signal the old thread to stop
            self._sim_thread.join(timeout=5.0)
            if self._sim_thread.is_alive():
                print("[EnvRunner] Warning: old simulation thread did not stop in time")
                return
        self._frame_buffer.clear()
        self._close_current_env()
        self._running = True
        self._stop_event.clear()
        self._restart_event.clear()
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

    def stop(self):
        """ 
        Stop the simulation. 
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

    def close(self):
        """
        Stop the simulation and clean up resources.
        """
        self.stop()
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=5.0)
        self._close_current_env()

    def request_restart(self):
        """
        Request a restart of the simulation. 
        """
        if self._running:
            self._restart_event.set()

    def set_model_path(self, model_path: str) -> Tuple[bool, str]:
        """
        Modifies _model_path to the given path. 
        Try to load the model immediately and return (success, message).
        """
        self._model_path = str(model_path or "").strip()
        return self._load_model(report_error=True)

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

    def get_replay_frame_meta(self, index: int):
        return self._frame_buffer.get_meta(index)

    def seek_replay_frame(self, index: int) -> bool:
        """
        Seek to a specific frame in the replay buffer. Returns True if successful, False if index is out of bounds.
        """
        with self._env_lock:
            env_ref = self._current_env
        if env_ref is None:
            return False
        try:
            if not env_ref._pybullet_client_alive():
                return False
        except Exception:
            return False
        return self._frame_buffer.seek(index, env_ref)
    
    # ---- Simulation loop ----

    def _simulation_loop(self):
        """Main simulation loop running in the background thread."""
        current_env = None
        try:
            while not self._stop_event.is_set(): # Loop over episodes
                configs = self._get_object_configs()
                if current_env is None:     # if no env exists create one
                    current_env = self._make_env(configs)

                with self._env_lock:    # set reference to current env for replay seeking and safe cleanup
                    self._current_env = current_env # Assign the current environment to the shared variable for access by other methods 

                obs_dict, _ = current_env.reset(seed=random.randint(0, 2**31 - 1)) # Reset env with a random seed for variability (seed is overriden by config and does not affect anything but possibly the lidar)
                self._last_warning = "" 
                self._apply_object_configs(current_env, configs)    # Apply obstacle configurations
                self._apply_spawn_configs(  # Apply drone and target spawn configurations
                    current_env,
                    self._get_drone_spawn_configs() if callable(self._get_drone_spawn_configs) else [],
                    self._get_target_spawn_configs() if callable(self._get_target_spawn_configs) else [],
                )
                obs_dict = current_env._build_obs_dict()    # Rebuild obs_dict to reflect any changes from applying configs
                # refresh _prev_dists to match new positions
                for i in range(current_env.total_num_drones):   # Loop over all drones to update their previous distances to targets based on their new positions after applying spawn configs
                    pos = obs_dict[current_env.possible_agents[i]][POS_SLICE]   # Get current position of the drone
                    current_env._prev_dists[i] = float(np.linalg.norm(current_env._targets[i].start_pos - pos))    # Update previous distances to targets based on new drone positions
                # print("Env made")
                self._episode += 1
                self._step = 0
                done = {agent: False for agent in current_env.possible_agents}  # Track done status for each agent; episode ends when all agents are done
                trunc = {agent: False for agent in current_env.possible_agents} # Track truncated status for each agent; episode ends when all agents are truncated
                restart_requested = False
                self._frame_buffer.record(current_env, self._episode, self._step) # Record initial state

                while not self._stop_event.is_set(): # Loop over steps within the episode
                    if self._restart_event.is_set():
                        restart_requested = True
                        self._restart_event.clear()
                        break
                    
                    # Simulation steps
                    # Check alive before step — prevents _ensure_live_aviary() inside
                    # step() from silently rebuilding a connection, which would
                    # make the post-step alive check see a "live" client and not stop.
                    if not current_env._pybullet_client_alive():    # Check if physics client is alive before step and stop if not. 
                        with self._env_lock:
                            self._current_env = None
                        current_env = None
                        self._stop_event.set()
                        break
                    act_dict = self._build_action_dict(obs_dict)
                    obs_dict, _, done, trunc, info_dict = current_env.step(act_dict, allow_rebuild=False)
                    self._step += 1
                    # If step() detected a dead connection, stop before any physics reads.
                    if any(
                        isinstance(info_dict.get(a), dict) and info_dict[a].get("physics_server_recovered")
                        for a in info_dict
                    ):
                        with self._env_lock:
                            self._current_env = None
                        current_env = None
                        self._stop_event.set()
                        break
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
                    time.sleep((1 / 144) / self.get_speed_multiplier())  # play in 144 fps

                    if all(done[a] or trunc[a] for a in done): # Episode ends when all agents are done or truncated
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

    def _make_env(self, configs):
        """
        Create a new environment instance based on the current configuration parameters and the provided object configs.
        """
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
        """
        Close the current environment instance if it exists.
        Uses reference to avoid data race conditions with the simulation thread.
        """
        with self._env_lock:
            env_ref = self._current_env
            self._current_env = None
        if env_ref is not None:
            try:
                env_ref.close()
            except Exception:
                pass

    @staticmethod
    def _apply_body_color(env_instance, body_id: int, rgba) -> None:
        """Apply an RGBA color to all visual links of a physics body."""
        p = getattr(env_instance, "_p", None)   # p is the physics client
        if p is None:
            return
        rgba = [max(0.0, min(1.0, float(c))) for c in rgba] 
        try:
            num_joints = env_instance._call_physics(p.getNumJoints, int(body_id)) # Get the number of joints for the body to determine how many visual links it has. If this fails, assume 0 joints.
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

    def _apply_object_configs(self, env_instance, configs):
        """
        Apply the given object configurations to the environment instance.
        This involves removing existing obstacles and adding new ones based on the configs.
        """
        p = getattr(env_instance, "_p", None)
        if p is None:
            return

        for obj in list(getattr(env_instance, "_obstacles", [])):
            body_id = getattr(obj, "body_id", None)
            if body_id is None:
                continue
            try:
                env_instance._call_physics(p.removeBody, body_id)
            except Exception as exc:
                print(f"[EnvRunner] Failed to remove obstacle body {body_id}: {exc}")
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
            except Exception as exc:
                print(f"[EnvRunner] Failed to create obstacle (shape={shape}): {exc}")
                continue

        env_instance._obstacles = new_obstacles
        env_instance.num_static_obstacles = sum(
            1 for cfg in configs if cfg.get("kind") == "static"
        )
        env_instance.num_dynamic_obstacles = sum(
            1 for cfg in configs if cfg.get("kind") == "dynamic"
        )

    def _apply_spawn_configs(self, env_instance, drone_spawns, target_spawns):
        """
        Apply the given drone and target spawn configurations to the environment instance.
        This involves resetting the positions of existing drones and targets, and applying any specified colors.
        """
        p = getattr(env_instance, "_p", None)   # get the physics client
        if p is None:
            return

        drone_ids = list(getattr(env_instance, "_drone_ids", []))   # get the list of drone IDs
        if drone_ids and drone_spawns:
            new_positions = []
            for i, body_id in enumerate(drone_ids[: len(drone_spawns)]):    # for each drone, apply the corresponding spawn config
                cfg = drone_spawns[i]
                pos = [
                    float(cfg.get("x", 0.0)),
                    float(cfg.get("y", 0.0)),
                    float(cfg.get("z", 0.8)),
                ]
                new_positions.append(pos)
                try:
                    env_instance._call_physics(     # reset drone position and orientation
                        p.resetBasePositionAndOrientation,
                        int(body_id),
                        pos,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    env_instance._call_physics(    # reset drone linear and angular velocity
                        p.resetBaseVelocity,
                        int(body_id),
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    )
                    if "color" in cfg:
                        self._apply_body_color(env_instance, int(body_id), cfg["color"])
                except Exception as exc:
                    print(f"[EnvRunner] Failed to apply drone spawn config {i}: {exc}")
                    continue
            # keep spawn_positions consistent with actual drone positions
            if new_positions:
                env_instance.spawn_positions = np.array(new_positions, dtype=np.float32)

        unique_targets = get_unique_targets(env_instance)

        if unique_targets and target_spawns:    # for each target, apply the corresponding spawn config
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
                    env_instance._call_physics(     # reset target position and orientation
                        p.resetBasePositionAndOrientation,
                        int(body_id),
                        pos,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    if hasattr(target, "start_pos"):
                        target.start_pos = np.array(pos, dtype=np.float32)
                    if "color" in cfg:
                        self._apply_body_color(env_instance, int(body_id), cfg["color"])
                        if hasattr(target, "color"):
                            target.color = cfg["color"]
                except Exception as exc:
                    print(f"[EnvRunner] Failed to apply target spawn config {i}: {exc}")
                    continue

    # ---- Model loading ----

    def _load_model(self, report_error: bool = True) -> Tuple[bool, str]:
        if not self._model_path:
            with self._model_lock:
                self._model = None
                self._model_backend = None
            return False, "No model selected."

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

    def _predict_action(self, obs: np.ndarray) -> np.ndarray:
        with self._model_lock:
            model = self._model
        if model is None:
            raise RuntimeError("No model loaded.")
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 4:
            raise RuntimeError(f"Model output has fewer than 4 values ({action.size}).")
        return action[:4]

    def _build_action_dict(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {agent: self._predict_action(obs) for agent, obs in obs_dict.items()}

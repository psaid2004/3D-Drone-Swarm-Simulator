
import random
import threading
import time

from envs.createObjects import Dynamic, EnvObject
from wrappers.pz_parallel_3d import env as make_env
from envs.createWorld import SUPPORTED_SHAPES 



class EnvironmentViewer:
    def __init__(self, object_configs_getter, on_state, on_stopped, on_error):
        self._get_object_configs = object_configs_getter
        self._on_state = on_state
        self._on_stopped = on_stopped
        self._on_error = on_error

        self._running = False
        self._sim_thread = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()

        self._episode = 0
        self._step = 0
        self._speed_lock = threading.Lock()
        self._speed_multiplier = 1.0

    @property
    def running(self):
        return self._running

    def start(self):
        if self._running:
            return
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

    def request_restart(self):
        if self._running:
            self._restart_event.set()

    def set_speed_multiplier(self, speed_multiplier: float):
        with self._speed_lock:
            self._speed_multiplier = max(0.1, float(speed_multiplier))

    def get_speed_multiplier(self) -> float:
        with self._speed_lock:
            return float(self._speed_multiplier)

    def _simulation_loop(self):
        current_env = None
        try:
            while not self._stop_event.is_set():
                configs = self._get_object_configs()
                current_env = self._make_env(configs)
                obs_dict, _ = current_env.reset(seed=random.randint(0, 2**31 - 1))
                self._apply_object_configs(current_env, configs)

                self._episode += 1
                self._step = 0

                done = {agent: False for agent in current_env.possible_agents}
                trunc = {agent: False for agent in current_env.possible_agents}

                while not self._stop_event.is_set():
                    if self._restart_event.is_set():
                        self._restart_event.clear()
                        break

                    act_dict = {agent: current_env.action_space(agent).sample() for agent in obs_dict}
                    obs_dict, _, done, trunc, _ = current_env.step(act_dict)
                    self._step += 1

                    self._on_state(
                        {
                            "episode": self._episode,
                            "step": self._step,
                            "drones": len(getattr(current_env, "agents", [])),
                            "obstacles": len(getattr(current_env, "_obstacles", [])),
                            "speed_multiplier": self.get_speed_multiplier(),
                        }
                    )
                    time.sleep((1 / 30) / self.get_speed_multiplier())

                    if all(done[a] or trunc[a] for a in done):
                        break

                current_env.close()
                current_env = None

        except Exception as exc:
            self._on_error(str(exc))
        finally:
            if current_env is not None:
                try:
                    current_env.close()
                except Exception:
                    pass
            self._running = False
            self._on_stopped()

    def _make_env(self, configs):
        static_count = sum(1 for cfg in configs if cfg.get("kind") == "static")
        dynamic_count = sum(1 for cfg in configs if cfg.get("kind") == "dynamic")

        return make_env(
            num_drones=8,
            num_static_obstacles=static_count,
            num_dynamic_obstacles=dynamic_count,
            max_steps=5000,
            lidar=False,
            render_mode="human",
        )

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
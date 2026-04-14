import threading
from typing import Any, Dict, List, Optional


def get_unique_targets(env_instance) -> list:
    """Returns targets from env_instance with duplicate body_ids removed."""
    seen: set = set()
    unique = []
    for tgt in list(getattr(env_instance, "_targets", []) or []):
        body_id = getattr(tgt, "body_id", None)
        if body_id is None or body_id in seen:
            continue
        seen.add(body_id)
        unique.append(tgt)
    return unique


class FrameBuffer:
    """Captures and replays physics body states during simulation."""

    def __init__(self, max_frames: int = 999_999):
        self._max_frames = max_frames
        self._lock = threading.Lock()
        self._frames: List[Dict[str, Any]] = []
        self._index: int = 0

    # ---- Public ----

    def clear(self):
        """ Clears all captured frames. """
        with self._lock:
            self._frames = []
            self._index = 0

    def record(self, env_instance, episode: int, step: int):
        """ Records the current state of the environment. """
        frame = self._capture_frame(env_instance, episode, step)
        with self._lock:
            self._frames.append(frame)
            if len(self._frames) > self._max_frames:
                overflow = len(self._frames) - self._max_frames
                self._frames = self._frames[overflow:]
            self._index = max(0, len(self._frames) - 1)

    def get_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def get_index(self) -> int:
        with self._lock:
            return int(self._index)

    def seek(self, index: int, env_instance) -> bool:
        with self._lock:
            if not self._frames:
                return False
            bounded = max(0, min(int(index), len(self._frames) - 1))
            frame = dict(self._frames[bounded])

        if env_instance is None:
            return False
        self._apply_frame(env_instance, frame)

        with self._lock:
            self._index = bounded
        return True

    def get_meta(self, index: int) -> Optional[Dict[str, int]]:
        with self._lock:
            if not self._frames:
                return None
            bounded = max(0, min(int(index), len(self._frames) - 1))
            frame = self._frames[bounded]
            return {
                "episode": int(frame.get("episode", 0)),
                "step": int(frame.get("step", 0)),
            }

    # ---- Frame capture ----

    def _capture_frame(self, env_instance, episode: int, step: int) -> Dict[str, Any]:
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
        for target in get_unique_targets(env_instance):
            body_id = getattr(target, "body_id", None)
            if body_id is None:
                continue
            state = self._capture_body_state(env_instance, body_id)
            if state is not None:
                target_states.append(state)

        return {
            "episode": int(episode),
            "step": int(step),
            "drones": drone_states,
            "obstacles": obstacle_states,
            "targets": target_states,
        }

    def _capture_body_state(self, env_instance, body_id: int) -> Optional[Dict[str, Any]]:
        p = getattr(env_instance, "_p", None)
        if p is None:
            return None
        try:
            position, orientation = env_instance._call_physics(
                p.getBasePositionAndOrientation, int(body_id)
            )
            linear_velocity, angular_velocity = env_instance._call_physics(
                p.getBaseVelocity, int(body_id)
            )
            return {
                "body_id": int(body_id),
                "position": [float(v) for v in position],
                "orientation": [float(v) for v in orientation],
                "linear_velocity": [float(v) for v in linear_velocity],
                "angular_velocity": [float(v) for v in angular_velocity],
            }
        except Exception:
            return None

    # ---- Frame restore ----

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

import threading
from typing import Any, Dict, List, Optional


def get_unique_targets(env_instance) -> list:
    """Returns targets from env_instance with duplicate body_ids removed."""
    seen: set = set()
    unique = []
    for tgt in list(getattr(env_instance, "_targets", []) or []):   # Loops over targets in the environment, handling the case where _targets might be None
        body_id = getattr(tgt, "body_id", None) # Safely gets the body_id of the target, defaulting to None if it doesn't exist
        if body_id is None or body_id in seen:  # If body_id is None or has already been seen, skip this target
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
            self._frames.append(frame)      # Appends the captured frame to the list of frames
            if len(self._frames) > self._max_frames:    # If the number of frames exceeds the maximum allowed, remove the oldest frames to maintain the limit (should never happen)
                overflow = len(self._frames) - self._max_frames # Calculates how many frames need to be removed to get back down to the maximum
                self._frames = self._frames[overflow:]  # Removes the oldest frames from the list, keeping only the most recent ones up to the maximum limit
            self._index = max(0, len(self._frames) - 1) # Updates the current index to point to the most recently recorded frame

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
        """
        Returns metadata for the frame at the given index.
        Metadata = episode and step numbers.  
        (Could be extended in the future to include other info like timestamps, rewards, etc. for detailed statitics)
        """
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
        """
        Captures a frame of the current environment state, including the states of all drones, obstacles, and targets.
        Each body's state includes its position, orientation, linear velocity, and angular velocity.
        """
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
        """
        Captures the state of a single physics body given its body_id, including position, orientation, linear velocity, and angular velocity.
        """
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
        """
        Restores the environment state from a captured frame, applying the saved position, orientation, linear velocity, and angular velocity to each drone, obstacle, and target based on their body_ids.
        """
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

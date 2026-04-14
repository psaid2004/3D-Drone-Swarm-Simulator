from typing import TypedDict


class ObjectConfig(TypedDict, total=False):
    name: str
    kind: str       # "static" | "dynamic"
    shape: str      # "box" | "sphere" | "wall" | "cylinder"; depends on createWorld's supported shapes
    x: float
    y: float
    z: float
    size: float
    yaw: float
    # Dynamic-only fields
    axis: int       # 0=x, 1=y, 2=z
    min: float      # Minimum position along the movement axis (relative to spawn point)
    max: float      
    speed: float
    direction: int  # 1 or -1


class SwarmConfig(TypedDict, total=False):
    name: str
    x: float
    y: float
    z: float
    color_name: str     # Optional color name for the swarm, e.g. "red", "blue", "green"; default color will be assigned if not specified.
    color: list[float]   # RGBA values between 0 and 1


class DroneSpawnConfig(TypedDict, total=False):
    name: str
    swarm_index: int    # Index of the swarm this drone belongs to
    local_x: float
    local_y: float
    local_z: float
    x: float
    y: float
    z: float
    color: list[float]  # RGBA values between 0 and 1


class TargetSpawnConfig(TypedDict, total=False):
    name: str
    x: float
    y: float
    z: float
    color: list[float]  # RGBA values between 0 and 1


class LiveState(TypedDict):
    episode: int
    step: int
    drones: int
    obstacles: int

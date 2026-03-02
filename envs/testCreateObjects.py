import pybullet as p # the physics engine API (loading URDFs, stepping simulation, forces, positions).
import pybullet_data # gives a path to built-in assets
from pathlib import Path
import time
import numpy as np
from createObjects import *


class Swarm:
    def __init__(self, max_drones) :
        self.max_dr = max_drones
        self.drones = []
        
    def add_drone(self, drone) :
        if len(self.drones) < self.max_dr :
            self.drones.append(drone)
            return True
        
        return False
    
class Drone:
    def __init__(self, start_pos, euler_angles):
        self.start_pos = start_pos
        self.rotation = p.getQuaternionFromEuler(euler_angles)   
    
class EnvObject:
    def __init__(self, start_pos, size, orientation):
        self.size = size
        self.start_pos = start_pos
        self.orientation = orientation
        self.SHAPE_CONFIGS = {
            "box": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([0.2, 0.2, 0.2]) * s},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([0.2, 0.2, 0.2]) * s},
                "use_orientation": False,
            },
            "wall": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([0.3, 0.05, 0.2]) * s},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([0.3, 0.05, 0.2]) * s},
                "use_orientation": True,
            },
            "sphere": {
                "geom": p.GEOM_SPHERE,
                "collision_kwargs": lambda s: {"radius": 0.2 * s},
                "visual_kwargs":    lambda s: {"radius": 0.2 * s},
                "use_orientation": False,
            },
            "cylinder": {
                "geom": p.GEOM_CYLINDER,
                "collision_kwargs": lambda s: {"radius": 0.1 * s, "height": 0.5 * s},
                "visual_kwargs":    lambda s: {"radius": 0.1 * s},
                "use_orientation": True,
            },
        }
    
    # kwargs = keyword arguments
    def create(self, shape, color=(1, 0, 0, 1)):
        conf = self.SHAPE_CONFIGS[shape]
        geom = conf["geom"]
        
        collision_kwargs = conf["collision_kwargs"](self.size)
        visual_kwargs = conf["visual_kwargs"](self.size)
        
        collision = p.createCollisionShape(geom, **collision_kwargs)
        visual = p.createVisualShape(geom, rgbaColor=color, **visual_kwargs)
        
        body_kwargs = {
            "baseMass": 0,
            "baseCollisionShapeIndex": collision,
            "baseVisualShapeIndex": visual,
            "basePosition": self.start_pos,
        }
        
        if conf["use_orientation"]:
            body_kwargs["baseOrientation"] = p.getQuaternionFromEuler(self.orientation)

        return p.createMultiBody(**body_kwargs)


class Dynamic(EnvObject):
    def __init__(self, axis, min, max, speed, direction, start_pos, size, orientation):
        self.axis = axis
        self.min = min
        self.max = max
        self.speed = speed
        self.direction = direction
        super().__init__(start_pos, size, orientation)
        
# 1️⃣ Connect to PyBullet
physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)  # Optional: simulation timestep

# 2️⃣ Add the standard data path (so PyBullet can find plane.urdf)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


dynamicObstacles = dict()


start = -2
end = 2
speed = 0.01  # meters per simulation step
direction = 1  # 1 = forward, -1 = backward


pathParams = {"axis" : 0, "min": start, "max": end, "speed":speed, "direction" :direction}

start_position = [2,0,1]
size = 1
orientation = [0,0,0]

cylinder = Dynamic(0, start, end, speed, direction, start_position, size, [0,0,0])

obs_id = cylinder.create("cylinder")
dynamicObstacles[obs_id] = pathParams

static_wall = EnvObject([0,3,3], 1, [0,0,1.57])
static_wall.create("wall")

start = -2
end = 1
speed = 0.01  # meters per simulation step
direction = -1  # 1 = forward, -1 = backward

pathParams = {"axis" : 0, "min": start, "max": end, "speed":speed, "direction" :direction}

wall_st_pos = [-2, -2, 1]

dynamic_wall = Dynamic(0, start, end, speed, direction, wall_st_pos, size, [0.2,0,1.57])
obs_id = dynamic_wall.create("wall")
dynamicObstacles[obs_id] = pathParams


start_position = [-2,2,1]
static_box = EnvObject(start_position, size, [0,0,0])
static_box.create("box")

# 3️⃣ Load a simple ground plane
plane_id = p.loadURDF("plane.urdf")



# Load URDF
drone = Drone([0, 0, 0.5], [0, 0, 0])

# Use CF2X drone URDF from gym-pybullet-drones
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
URDF_PATH  = ASSETS_DIR / "cf2p.urdf"
p.setAdditionalSearchPath(str(ASSETS_DIR))
print(str(URDF_PATH))

drone_id = p.loadURDF(str(URDF_PATH), drone.start_pos, drone.rotation)



Kp = 20.0
Kd = 2.0

target_z = 1.0  # target hover height in meters

# -------------------------------
# 5️⃣ Simulation loop
# -------------------------------
for t in range(8000):

    for obj_id, path in dynamicObstacles.items():
        update_single_axis_obstacle(obj_id, path)

    # Get current drone position and velocity
    pos, ori = p.getBasePositionAndOrientation(drone_id)
    lin_vel, ang_vel = p.getBaseVelocity(drone_id)

    z_error = target_z - pos[2]
    vz = -lin_vel[2]  # velocity in z

    # Compute simple PID force along z-axis
    f_z = Kp * z_error + Kd * vz

    # Apply force at center of mass (z-axis only)
    p.applyExternalForce(drone_id, -1, [0, 0, f_z], [0, 0, 0], p.WORLD_FRAME)

    # Step simulation
    p.stepSimulation()
    time.sleep(1/240)

# 5️⃣ Disconnect when done
p.disconnect()

class Simulator:
    def simulate(self, env, t) :
        pass
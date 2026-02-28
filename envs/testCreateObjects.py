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

    def get_drones(self) :
        return self.drones
    
class Drone:
    def __init__(self, start_pos, euler_angles):
        self.start_pos = start_pos
        self.rotation = p.getQuaternionFromEuler(euler_angles)
    
    def set_id(self, id) :
        self.id = id
            
    def get_start(self) :
        return self.start_pos
    
    def get_ori(self):
        return self.rotation    
    
class EnvObject:
    def __init__(self):
        pass

class Dynamic(EnvObject):
    def __init__(self, axis, min, max, speed, direction):
        self._axis = axis
        self._min = min
        self._max = max
        self._speed = speed
        self._direction = direction

    # axis
    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        self._axis = value

    # min
    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value

    # max
    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value

    # speed
    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    # direction
    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value
        
        
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
obs_id, obs_path = create_dynamic_cylinder(start_position, size, pathParams)
dynamicObstacles[obs_id] = obs_path

create_static_wall([0,3,3],1, [0,0,1.57])


start = -2
end = 1
speed = 0.01  # meters per simulation step
direction = -1  # 1 = forward, -1 = backward

pathParams = {"axis" : 0, "min": start, "max": end, "speed":speed, "direction" :direction}

obs_id, obs_path = create_dynamic_wall([-2, -2, 1], size, pathParams, [0.2,0,1.57])
dynamicObstacles[obs_id] = obs_path


start_position = [-2,2,1]
create_static_box(start_position, size)

# 3️⃣ Load a simple ground plane
plane_id = p.loadURDF("plane.urdf")



# Load URDF
drone_start_pos = [0, 0, 0.5]  # half a meter above ground
drone_start_ori = p.getQuaternionFromEuler([0, 0, 0])

# Use CF2X drone URDF from gym-pybullet-drones
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
URDF_PATH  = ASSETS_DIR / "cf2p.urdf"
p.setAdditionalSearchPath(str(ASSETS_DIR))
print(str(URDF_PATH))

drone_id = p.loadURDF(str(URDF_PATH), drone_start_pos, drone_start_ori)



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
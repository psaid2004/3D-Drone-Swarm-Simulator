import pybullet as p # the physics engine API (loading URDFs, stepping simulation, forces, positions).
import pybullet_data # gives a path to built-in assets
from pathlib import Path
import time
import numpy as np
from createObjects import *




# 1️⃣ Connect to PyBullet
physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)  # Optional: simulation timestep

# 2️⃣ Add the standard data path (so PyBullet can find plane.urdf)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


dynamicObstacles = dict()

x_start = -2
x_end = 2
speed = 0.01  # meters per simulation step
direction = 1  # 1 = forward, -1 = backward

pathParams = [x_start, x_end, speed, direction]

start_position = [2,0,1]
size = 1
orientation = [0,0,0]
obs_id, obs_path = create_dynamic_cylinder(start_position, size, pathParams)
dynamicObstacles[obs_id] = obs_path


x_start = -2
x_end = 1
speed = 0.01  # meters per simulation step
direction = -1  # 1 = forward, -1 = backward

pathParams = [x_start, x_end, speed, direction]

obs_id, obs_path = create_dynamic_wall([-2, -2, 1], size, pathParams, [0,0,1.57])
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

    for id,path in dynamicObstacles.items():
        x_start = path[0]
        x_end = path[1]
        speed = path[2]
        direction = path[3]

        pos, orn = p.getBasePositionAndOrientation(id)
        current_x = pos[0]
        current_y = pos[1]
        current_z = pos[2]

        current_x += speed * direction

        if current_x >= x_end or current_x <= x_start:
            path[3] *= -1

        p.resetBasePositionAndOrientation(
            id,
            [current_x, current_y, current_z],
            orn
        )

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

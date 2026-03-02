import pybullet as p
import pybullet_data
import time
from pathlib import Path

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

class Environment:
    def __init__(self, difficulty, seed=None):
        self.difficulty = difficulty
        self.seed = seed
        
        self.stat_objects = {} # mapping object -> object id
        self.dyn_objects = {} # mapping object -> object id
        self.drones = []
        self.plane_id = None
    
    def spawn_env(self):
        self.plane_id = p.loadURDF("plane.urdf")
        
        
    
# Load URDF

drone = Drone([0, 0, 0.5], [0, 0, 0])

# Use CF2X drone URDF from gym-pybullet-drones
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
URDF_PATH  = ASSETS_DIR / "cf2p.urdf"
p.setAdditionalSearchPath(str(ASSETS_DIR))
print(str(URDF_PATH))

drone_id = p.loadURDF(str(URDF_PATH), drone.start_pos, drone.rotation)


import pybullet as p
import pybullet_data
import time
import numpy as np
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
        
        self.objects = [] # list of tuples of the following format (shape, object, dynamic(bool))
        self.stat_objects = {} # mapping object id -> object
        self.dyn_objects = {}
        self.drones = []
        self.plane_id = None
    
    def spawn_env(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.generate_objects()
        for obj in self.objects :
            shape, obj, dyn = obj
            obj_id = obj.create(shape)
            if dyn :
                self.dyn_objects[obj_id] = obj
            else :
                self.stat_objects[obj_id] = obj
            
    def generate_objects(self):
        rng = np.random.default_rng()
        
        
    
# Load URDF

drone = Drone([0, 0, 0.5], [0, 0, 0])

# Use CF2X drone URDF from gym-pybullet-drones
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
URDF_PATH  = ASSETS_DIR / "cf2p.urdf"
p.setAdditionalSearchPath(str(ASSETS_DIR))
print(str(URDF_PATH))

drone_id = p.loadURDF(str(URDF_PATH), drone.start_pos, drone.rotation)


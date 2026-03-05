import pybullet as p
import pybullet_data
import time
import numpy as np
from createObjects import *
from pathlib import Path

SUPPORTED_SHAPES = ["box", "wall", "sphere", "cylinder"]
WORLD_BOUNDS = {
    "x" : (-3.0, 3.0),
    "y" : (-3.0, 3.0),
    "z" : (0, 5.0)
}

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
        physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
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
        
        n_static = 5
        if self.difficulty == "easy":
            n_static = rng.integers(2, 5)
        
        for _ in range(n_static):
            shape = rng.choice(SUPPORTED_SHAPES)
            pos = [float(rng.uniform(*WORLD_BOUNDS["x"])), float(rng.uniform(*WORLD_BOUNDS["y"])), 1.0]
            size = float(rng.uniform(0.8, 1.2))
            ori = [0,0, float(rng.uniform(-np.pi, np.pi))]
            self.objects.append((shape, EnvObject(pos, size, ori), False))
            
# Load URDF


env = Environment("easy")
env.spawn_env()

drone = Drone([0, 0, 0.5], [0, 0, 0])

# Use CF2X drone URDF from gym-pybullet-drones
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
URDF_PATH  = ASSETS_DIR / "cf2p.urdf"
p.setAdditionalSearchPath(str(ASSETS_DIR))
print(str(URDF_PATH))

drone_id = p.loadURDF(str(URDF_PATH), drone.start_pos, drone.rotation)

while p.isConnected():
    p.stepSimulation()
    time.sleep(1/240)
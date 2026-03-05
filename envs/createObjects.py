import pybullet as p
import pybullet_data
import time
import numpy as np
    

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
    def create(self, shape, physics_call, color=(1, 0, 0, 1)):
        self.physics_call = physics_call
        conf = self.SHAPE_CONFIGS[shape]
        geom = conf["geom"]
        
        collision_kwargs = conf["collision_kwargs"](self.size)
        visual_kwargs = conf["visual_kwargs"](self.size)
        
        collision = physics_call(p.createCollisionShape,geom, **collision_kwargs)
        visual = physics_call(p.createVisualShape,geom, rgbaColor=color, **visual_kwargs)
        
        body_kwargs = {
            "baseMass": 0,
            "baseCollisionShapeIndex": collision,
            "baseVisualShapeIndex": visual,
            "basePosition": self.start_pos,
        }
        
        if conf["use_orientation"]:
            body_kwargs["baseOrientation"] = p.getQuaternionFromEuler(self.orientation)

        body = physics_call(p.createMultiBody, **body_kwargs)
        self.body_id = body  # store ID in object
        return self.body_id

class Dynamic(EnvObject):
    def __init__(self, axis, min, max, speed, direction, start_pos, size, orientation):
        self.axis = axis
        self.min = min
        self.max = max
        self.speed = speed
        self.direction = direction
        super().__init__(start_pos, size, orientation)

    def update_obstacle(self, t):
        # t = current simulation time in seconds
        pos, orn = self.physics_call(p.getBasePositionAndOrientation,self.body_id)

        pos = list(pos)

        # Smooth oscillation along your chosen axis
        amplitude = (self.max - self.min) / 2
        mid = (self.max + self.min) / 2
        pos[self.axis] = mid + amplitude * np.sin(2 * np.pi * self.speed * t)
        self.physics_call(p.resetBasePositionAndOrientation, self.body_id, pos, orn)
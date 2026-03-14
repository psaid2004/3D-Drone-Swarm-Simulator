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
            "x_pos_wall": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([0.05, s, s])},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([0.05, s, s])},
                "use_orientation": False
            },
            "y_pos_wall": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([s, 0.05, s])},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([s, 0.05, s])},
                "use_orientation": False
            },
            "z_pos_wall": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([s, s, 0.05])},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([s, s, 0.05])},
                "use_orientation": False
            }
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

    def isOverlapping(self, obstacle):
        pts = self.physics_call(
            p.getClosestPoints,
            bodyA=self.body_id,
            bodyB=obstacle.body_id,
            distance=0.0
        )
        if len(pts) > 0:
            return True
        else:
            return False

    # Check if obstacle is too close to drones or the targets
    def isValidPosition(self, dronePositions, targets, min_dist):
        if np.any(np.linalg.norm(dronePositions - self.start_pos, axis=1) < min_dist):
            return False

        targets_pos = []
        for target in targets:
            try:
                targets_pos.append(target.start_pos)
            except Exception:
                pass

        targets_pos = np.array(targets_pos, dtype=np.float32)
        if len(targets_pos) > 0:
            if np.any(np.linalg.norm(targets_pos - self.start_pos, axis=-1) < min_dist):
                return False

        return True



class Dynamic(EnvObject):
    def __init__(self, axis, min, max, speed, direction, start_pos, size, orientation):
        self.axis = axis
        self.min = min
        self.max = max
        self.speed = speed
        self.direction = direction
        super().__init__(start_pos, size, orientation)

    def update_obstacle(self, dt):
        pos, orn = self.physics_call(p.getBasePositionAndOrientation, self.body_id)
        pos = list(pos)

        # move along chosen axis
        pos[self.axis] += self.direction * self.speed * dt

        # clamp and flip at boundaries
        if pos[self.axis] <= self.min:
            pos[self.axis] = self.min
            self.direction = 1
        elif pos[self.axis] >= self.max:
            pos[self.axis] = self.max
            self.direction = -1

        self.physics_call(p.resetBasePositionAndOrientation, self.body_id, pos, orn)


    # TODO CHANGE SO IT DETECTS IF OVERLAP WILL HAPPEN DURING ITS PATH
    def isOverlapping(self, obstacle):
        pts = self.physics_call(
            p.getClosestPoints,
            bodyA=self.body_id,
            bodyB=obstacle.body_id,
            distance=0.0
        )
        if len(pts) > 0:
            return True
        else:
            return False


class Target():
    def __init__(self, color, start_pos, radius, physics_call, call_gui):
        self.start_pos = start_pos
        self.radius = radius
        self.physics_call = physics_call
        self.call_gui = call_gui
        self.color = color

    # Check if the target is too close to drones and other targets
    def isValidPosition(self, dronePositions, targets, min_dist):
        if np.any(np.linalg.norm(dronePositions - self.start_pos, axis=1) < min_dist):
            return False

        target_pos = []
        for target in targets:
            try:
                if not np.allclose(target.start_pos, self.start_pos):
                    target_pos.append(target.start_pos)
            except Exception:
                pass

        target_pos = np.array(target_pos, dtype=np.float32)
        if len(target_pos) > 0:
            target_pos = np.array(target_pos, dtype=np.float32).reshape(-1, 3)
            if np.any(np.linalg.norm(target_pos - self.start_pos, axis=-1) < min_dist):
                return False

        return True

    def spawn_target(self):
        vis = self.call_gui(
            p.createVisualShape,
            p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=self.color,
        )


        body = self.call_gui(
                p.createMultiBody,
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(self.start_pos[0]), float(self.start_pos[1]), float(self.start_pos[2])],
            )
        return body

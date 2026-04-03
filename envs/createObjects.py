import pybullet as p
import pybullet_data
import time
import numpy as np

    
class EnvObject:
    def __init__(self, start_pos, size, orientation):
        self.size = size
        self.start_pos = start_pos
        self.orientation = orientation
        self.baseQuat = p.getQuaternionFromEuler(self.orientation)
        self.half_extents = None
        self.radius = None
        self.height = None
        self.shape = None
        self.collision_shape_spec = None
        self.SHAPE_CONFIGS = {
            "box": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([0.2, 0.2, 0.2]) * s},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([0.2, 0.2, 0.2]) * s},
                "use_orientation": False,
                "radius" : 0.2*size
            },
            "wall": {
                "geom": p.GEOM_BOX,
                "collision_kwargs": lambda s: {"halfExtents": np.array([0.3, 0.05, 0.2]) * s},
                "visual_kwargs":    lambda s: {"halfExtents": np.array([0.3, 0.05, 0.2]) * s},
                "use_orientation": True,
                "radius" : 0.3 * size
            },
            "sphere": {
                "geom": p.GEOM_SPHERE,
                "collision_kwargs": lambda s: {"radius": 0.2 * s},
                "visual_kwargs":    lambda s: {"radius": 0.2 * s},
                "use_orientation": False,
                "radius" : 0.2 * size
            },
            "cylinder": {
                "geom": p.GEOM_CYLINDER,
                "collision_kwargs": lambda s: {"radius": 0.1 * s, "height": 0.5 * s},
                "visual_kwargs":    lambda s: {"radius": 0.1 * s, "length": 0.5 * s}, 
                "use_orientation": True,
                "radius": 0.1 * size 
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
    
    
    def set_shape_spec(self, shape):
        self.shape = shape
        conf = self.SHAPE_CONFIGS[shape]
        geom = conf["geom"]
        collision_kwargs = conf["collision_kwargs"](self.size)
        
        self.collision_shape_spec = {
            "geom": geom,
            "collision_kwargs": collision_kwargs
        }
        
        self.half_extents = collision_kwargs.get("halfExtents")
        self.radius = collision_kwargs.get("radius", conf.get("radius"))
        self.height = collision_kwargs.get("height")
    
    
    # kwargs = keyword arguments
    def create(self, shape, physics_call, color=(1, 0, 0, 1)):
        self.physics_call = physics_call
        self.shape = shape
        self.set_shape_spec(shape)
        conf = self.SHAPE_CONFIGS[shape]
        geom = self.collision_shape_spec["geom"]
        collision_kwargs = self.collision_shape_spec["collision_kwargs"]
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
 # and targets?
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

    def is_in_swept_area(self, object, safety_margin = 0.3):
        center, half_extents = object.get_swept_area()
        p = self.start_pos
        for axis in range(3):
            if abs(p[axis] - center[axis]) > half_extents[axis] + safety_margin:
                return False
        return True
    
    # Check if obstacle is too close to drones or the targets
    def isValidPosition(self, dronePositions, targets, min_dist):
        # Check if too close to drones
        if np.any(np.linalg.norm(dronePositions - self.start_pos, axis=1) < min_dist):
            return False
        targets_pos = []
        for target in targets:
            try:
                targets_pos.append(target.start_pos)
            except Exception:
                pass

        targets_pos = np.array(targets_pos, dtype=np.float32)
        # Check if too close to target
        if len(targets_pos) > 0:
            if np.any(np.linalg.norm(targets_pos - self.start_pos, axis=-1) < min_dist):
                return False

        return True
    
    def pose_at_time(self, t: float):
        return self.start_pos.tolist(), self.baseQuat



class Dynamic(EnvObject):
    def __init__(self, axis, min, max, speed, direction, start_pos, size, orientation):
        self.axis = axis
        self.min = min
        self.max = max
        self.speed = speed
        self.direction = direction
        super().__init__(start_pos, size, orientation)

    def update_obstacle(self, dt):
        if self.body_id is None:
            return

        try:
            pos, orn = p.getBasePositionAndOrientation(self.body_id)
        except Exception:
            self.body_id = None
            return

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

        try:
            p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        except Exception:
            self.body_id = None
            return

    def get_swept_area(self):
        center = self.start_pos.copy()
        center[self.axis] = (self.min + self.max) / 2
        object_radius = 0.3 * self.size
        half_extents = [object_radius for _ in range(3)]
        half_extents[self.axis] = (self.max - self.min) / 2
        return center, half_extents

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

    def axis_position_at_time(self, t):
        path_length = self.max - self.min
        x0 = float(self.start_pos[self.axis])
        v = float(self.direction) * float(self.speed) #sigend velocity
        
        unfolded = x0 + v * t # motion formula
        y = (unfolded - self.min) % (2 * path_length)
        if y <= path_length:
            return self.min + y
        else:
            return self.min + (2 * path_length - y) # bounce backwards
    
    def full_position_at_time(self, t):
        pos = self.start_pos.copy()
        pos[self.axis] = self.axis_position_at_time(t)
        return pos
    
    def full_position_at_step(self, step: int, dt: float = 1.0/60.0) -> np.array:
        return self.full_position_at_time(step * dt)
    
    def get_period(self) -> float:
        path_length = self.max - self.min
        return 2.0 * path_length / self.speed
    
    def pose_at_time(self, t: float):
        return self.full_position_at_time(t).tolist(), self.baseQuat
    


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

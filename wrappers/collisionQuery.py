import pybullet as p
import numpy as np
from envs.createObjects import EnvObject, Dynamic

class CollisionQueryWorld:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)

        self.static_bodies = []
        self.dynamic_bodies = {}
        self.temp_bodies = []
    
    def close(self):
        if p.isConnected(self.cid) :
            p.disconnect(self.cid)
    
    def _create_body_from_spec(self, spec, pos, quat):
        shape_id = p.createCollisionShape(
            physicsClientId=self.cid,
            shapeType=spec["geom"],
            **spec["collision_kwargs"]
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,
            basePosition=pos,
            baseOrientation=quat,
            physicsClientId=self.cid
        )
        return body_id

    def add_dynamic_object(self, key, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(obj.collision_shape_spec, pos, obj.baseQuat)
        self.dynamic_bodies[key] = body_id
        return body_id

    def add_static_object(self, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(obj.collision_shape_spec(), pos, obj.baseQuat)
        self.static_bodies.append(body_id)
        return body_id

    
    def add_temp_object(self, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(obj.collision_shape_spec, pos, obj.baseQuat)
        self.temp_bodies.append(body_id)
        return body_id


    def set_body_pose(self, body_id, pos, quat):
        p.resetBasePositionAndOrientation(
            body_id,
            pos,
            quat,
            physicsClientId=self.cid,
        )

    def bodies_collide(self, body_a, body_b, margin=0.0):
        pts = p.getClosestPoints(
            bodyA=body_a,
            bodyB=body_b,
            distance=margin,
            physicsClientId=self.cid,
        )
        return len(pts) > 0
    
    def remove_body(self, body_id):
        try:
            p.removeBody(body_id, physicsClientIds = self.cid)
        except Exception:
            pass
    
    def clear(self):
        self.close()
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        self.static_bodies = []
        self.dynamic_bodies = {}
        self.temp_bodies = []
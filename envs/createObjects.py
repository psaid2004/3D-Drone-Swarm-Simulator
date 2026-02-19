import pybullet as p
import pybullet_data
import time
import numpy as np


#TODO MAKE MORE REALISTIC/COMPLEX SHAPES

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*[Width(X), Depth(Y), Height(Z)]
def create_static_box(startPosition, scale):
    base_size = np.array([0.2,0.2,0.2])
    scaled_size = base_size*scale
    box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents = scaled_size)
    box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = scaled_size, rgbaColor=[1, 0, 0, 1])
    box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision,baseVisualShapeIndex=box_visual, basePosition=startPosition)

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*[Width(X), Depth(Y), Height(Z)]
# orientation = [X/Roll, Y/Pitch, Z/Yaw] how the object in orientated when its spawned (tilted= [0.2, 0,0] , 90 degrees to the right = [0,0,1.57])
def create_static_wall(startPosition, scale, orientation =[0,0,0]):
    base_size = np.array([0.3,0.05,0.2])
    scaled_size = base_size*scale
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents = scaled_size)
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = scaled_size, rgbaColor=[1, 0, 0, 1])
    wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual, basePosition=startPosition, baseOrientation=p.getQuaternionFromEuler(orientation))

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*radius
def create_static_sphere(startPosition, scale):
    base_radius = 0.2
    scaled_radius = base_radius*scale
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=scaled_radius)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=scaled_radius, rgbaColor=[1, 0, 0, 1])
    sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision,baseVisualShapeIndex=sphere_visual, basePosition=startPosition)

def create_static_cylinder(startPosition, scale, orientation=[0,0,0]):
    base_radius = 0.1
    base_height = 0.5
    scaled_radius = base_radius*scale
    scaled_height = base_height*scale
    cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=scaled_radius, height=scaled_height)
    cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=scaled_radius, length=scaled_height, rgbaColor=[1, 0, 0, 1])
    cylinder_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cylinder_collision,baseVisualShapeIndex=cylinder_visual, basePosition=startPosition, baseOrientation=p.getQuaternionFromEuler(orientation))


# --------------------------------------------- Dynamic Obstacles ---------------------------------


# All functions take pathParameters which is a
# path = {
#     "axis": 0,          # 0 = X, 1 = Y, 2 = Z
#     "min": -2.0,
#     "max":  2.0,
#     "speed": 0.01,
#     "direction": 1
# }

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*[Width(X), Depth(Y), Height(Z)]
def create_dynamic_box(startPosition, scale, pathParameters):
    base_size = np.array([0.2,0.2,0.2])
    scaled_size = base_size*scale
    box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents = scaled_size)
    box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = scaled_size, rgbaColor=[1, 0, 0, 1])
    box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision,baseVisualShapeIndex=box_visual, basePosition=startPosition)
    return box_id,pathParameters

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*[Width(X), Depth(Y), Height(Z)]
# orientation = [X/Roll, Y/Pitch, Z/Yaw] how the object in orientated when its spawned (tilted= [0.2, 0,0] , 90 degrees to the right = [0,0,1.57])
def create_dynamic_wall(startPosition, scale, pathParameters, orientation =[0,0,0]):
    base_size = np.array([0.3,0.05,0.2])
    scaled_size = base_size*scale
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents = scaled_size)
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = scaled_size, rgbaColor=[1, 0, 0, 1])
    wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual, basePosition=startPosition, baseOrientation=p.getQuaternionFromEuler(orientation))
    return wall_id, pathParameters

# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*radius
def create_dynamic_sphere(startPosition, scale, pathParameters):
    base_radius = 0.2
    scaled_radius = base_radius*scale
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=scaled_radius)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=scaled_radius, rgbaColor=[1, 0, 0, 1])
    sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision,baseVisualShapeIndex=sphere_visual, basePosition=startPosition)
    return sphere_id, pathParameters


# startPosition = [X, Y, Z]
# scale =number x, that will be used to x*radius and x*height
# orientation = [X/Roll, Y/Pitch, Z/Yaw] how the object in orientated when its spawned (tilted= [0.2, 0,0] , 90 degrees to the right = [0,0,1.57])
def create_dynamic_cylinder(startPosition, scale, pathParameters, orientation=[0,0,0]):
    base_radius = 0.1
    base_height = 0.5
    scaled_radius = base_radius*scale
    scaled_height = base_height*scale
    cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=scaled_radius, height=scaled_height)
    cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=scaled_radius, length=scaled_height, rgbaColor=[1, 0, 0, 1])
    cylinder_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cylinder_collision,baseVisualShapeIndex=cylinder_visual, basePosition=startPosition, baseOrientation=p.getQuaternionFromEuler(orientation))
    return cylinder_id, pathParameters

def update_single_axis_obstacle(obj_id, path):
    pos, orn = p.getBasePositionAndOrientation(obj_id)
    pos = list(pos)

    axis = path["axis"]
    pos[axis] += path["speed"] * path["direction"]

    if pos[axis] > path["max"]:
        pos[axis] = path["max"]
        path["direction"] *= -1
    elif pos[axis] < path["min"]:
        pos[axis] = path["min"]
        path["direction"] *= -1

    p.resetBasePositionAndOrientation(obj_id, pos, orn)

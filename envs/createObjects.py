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

import pybullet as p
import pybullet_data
import time
import numpy as np

# All functions take pathParameters which is a [x_start, x_end, speed, direction = 1/forward or -1/backward]


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
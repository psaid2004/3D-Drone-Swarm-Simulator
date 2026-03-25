import random

SHAPES = {
    "box": {"default_size": 0.5, "size_range": (0.3, 4.0)},
    "sphere": {"default_size": 0.5, "size_range": (0.3, 4.0)},
    "cylinder": {"default_size": 0.5, "size_range": (0.3, 4.0)},
    "capsule": {"default_size": 0.5, "size_range": (0.3, 4.0)},
}

MOVEMENT_AXES = {
    0: "X",
    1: "Y", 
    2: "Z"
}

def generate_random_object(obj_id):
    """Generate a random obstacle object with randomized properties"""
    
    kind = random.choice(["dynamic", "static"])
    
    shape = random.choice(list(SHAPES.keys()))
    
    x = random.uniform(-5.0, 5.0)
    y = random.uniform(-5.0, 5.0)
    z = random.uniform(-0.0, 5.0)
    
    size_range = SHAPES[shape]["size_range"]
    size = random.uniform(size_range[0], size_range[1])
    
    yaw = random.uniform(0.0, 360.0)
    
    if kind == "dynamic":
        axis = random.randint(0, 2)
        if axis == 0:  
            min_pos = max(-10.0, x - random.uniform(1.0, 8.0))
            max_pos = min(10.0, x + random.uniform(1.0, 8.0))
            if min_pos >= max_pos:
                min_pos = max(-10.0, x - 2.0)
                max_pos = min(10.0, x + 2.0)
        elif axis == 1:
            min_pos = max(-10.0, y - random.uniform(1.0, 8.0))
            max_pos = min(10.0, y + random.uniform(1.0, 8.0))
            if min_pos >= max_pos:
                min_pos = max(-10.0, y - 2.0)
                max_pos = min(10.0, y + 2.0)
        else: 
            min_pos = max(-10.0, z - random.uniform(1.0, 8.0))
            max_pos = min(10.0, z + random.uniform(1.0, 8.0))
            if min_pos >= max_pos:
                min_pos = max(-10.0, z - 2.0)
                max_pos = min(10.0, z + 2.0)
        
        speed = random.uniform(0.1, 0.8)
        
        direction = random.choice([-1, 1])
    else:
        axis = 0
        min_pos = -2.0
        max_pos = 2.0
        speed = 0.0
        direction = 0
    
    return {
        "name": f"Object {obj_id + 1}",
        "kind": kind,
        "shape": shape,
        "x": round(x),
        "y": round(y),
        "z": round(z),
        "size": round(size),
        "yaw": round(yaw),
        "axis": axis,
        "min": min_pos,
        "max": max_pos,
        "speed": speed,
        "direction": direction
    }

def generate_random_objects(num_objects=2, config=None):
    objects = []
    
    if config and config.get('num_obstacles', 0) > 0:
        num_objects = config.get('num_obstacles', 5)
    
    for i in range(num_objects):
        obj = generate_random_object(i)
        objects.append(obj)
    
    return objects
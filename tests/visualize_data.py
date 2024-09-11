import numpy as np
from matplotlib import pyplot as plt

def generate_right_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    x = start_x + radius * (1 - np.cos(t)) * strength
    y = start_y + radius * np.sin(t)
    return x, y

def generate_left_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    x = start_x - radius * (1 - np.cos(t)) * strength
    y = start_y + radius * np.sin(t)
    return x, y

def rotate_points(x, y, rotation_angle):
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle
    return x_rot, y_rot

def generate_one_turn(robot_x, robot_y, how, inertia_angle, radius_min=2, radius_max=15, angle_min=70, angle_max=110, strength_min=0.5, strength_max=2) -> tuple:
    # Turn parameters
    radius = np.random.uniform(radius_min, radius_max)  # Rayon de courbure
    angle = np.random.uniform(angle_min, angle_max)  # Angle de la courbe
    strength = np.random.uniform(strength_min, strength_max)  # Force de la courbe

    if how == 'right':
        x, y = generate_right_turn(start_x = 0, start_y = 0, radius =  radius, angle = angle, strength = strength)
    elif how == 'left':
        x, y = generate_left_turn(start_x = 0, start_y = 0, radius = radius, angle = angle, strength = strength)

    # Rotate turn to align it with the robot's direction
    rotation_angle = inertia_angle - np.pi / 2
    x_rot, y_rot = rotate_points(x, y, rotation_angle)

    # Shift turn towards the robot
    x_rot += robot_x
    y_rot += robot_y
    return x_rot, y_rot, radius, angle

# N Robots visualisation
num_robots = 10
robot_spawn = 10
bounds = 20
robot_positions = np.random.uniform(-robot_spawn, robot_spawn, (num_robots, 2))
inertia_angles = np.random.uniform(-np.pi, np.pi, num_robots)
inertia_length = 2
scatter = False

plt.figure(figsize=(10, 10))

for i in range(num_robots):
    # Robot position
    robot_x, robot_y = robot_positions[i]
    inertia_angle = inertia_angles[i]
    
    x_rot, y_rot,radius, angle = generate_one_turn(robot_x, robot_y, 'right', inertia_angle)
    # If the turn is out of bounds
    while np.any(np.abs(x_rot) > bounds * 0.9) or np.any(np.abs(y_rot) > bounds * 0.9):
        x_rot, y_rot, radius, angle = generate_one_turn(robot_x, robot_y, 'right', inertia_angle)

    if scatter:
        indices = np.linspace(0, len(x_rot) - 1, 5, dtype=int)
        x_rot = x_rot[indices]
        y_rot = y_rot[indices]
        plt.scatter(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')
    else:
        plt.plot(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')

    plt.plot(robot_x, robot_y, 'go', markersize=10)
    plt.arrow(robot_x, robot_y, inertia_length * np.cos(inertia_angle), inertia_length * np.sin(inertia_angle),
              head_width=0.5, head_length=0.5, fc='blue', ec='blue')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
# Draw a red square around the bounds
plt.plot([-bounds, bounds, bounds, -bounds, -bounds], [-bounds, -bounds, bounds, bounds, -bounds], 'r')
plt.axis([-bounds - 5, bounds + 5, -bounds - 5, bounds + 5])
plt.show()

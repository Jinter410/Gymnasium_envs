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

# N Robots visualisation
num_robots = 10
robot_positions = np.random.uniform(-10, 10, (num_robots, 2))
inertia_angles = np.random.uniform(0, 2 * np.pi, num_robots)
inertia_length = 2
scatter = False

plt.figure(figsize=(10, 10))

for i in range(num_robots):
    # Robot position
    robot_x, robot_y = robot_positions[i]
    inertia_angle = inertia_angles[i]
    
    # Turn parameters
    radius = np.random.uniform(5, 15)  # Rayon de courbure
    angle = np.random.uniform(70, 110)  # Angle de la courbe
    strength = np.random.uniform(0.5, 2)  # Force de la courbe

    x, y = generate_left_turn(0, 0, radius, angle, strength)

    # Rotate turn to align it with the robot's direction
    rotation_angle = inertia_angle - np.pi / 2
    x_rot, y_rot = rotate_points(x, y, rotation_angle)

    # Shift turn towards the robot
    x_rot += robot_x
    y_rot += robot_y

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
plt.axis('equal')
plt.show()

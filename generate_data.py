import gymnasium as gym
import numpy as np
import Gymnasium_envs
import pygame

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

def main(n_rays=180, n_crowd=4, interceptor_percentage = 0.5, max_steps=100, render_mode="human"):
    pygame.init()
    env_name = "Navigation-v0"
    if env_name == "Navigation-v0":
        env = gym.make(env_name, n_rays=n_rays, n_crowd=n_crowd, max_steps=max_steps, render_mode=render_mode)
    else:
        env = gym.make(env_name, n_rays=n_rays, n_crowd=n_crowd, interceptor_percentage=interceptor_percentage, max_steps=max_steps, render_mode=render_mode)
    observation = env.reset()
    print(observation)
    quit()
    done = False
    truncated = False
    clock = pygame.time.Clock()

    while not (done or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = env.action_space.sample()

        observation, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    for i in range(100):
        main(n_rays= 40, n_crowd=10, interceptor_percentage=1, max_steps=700, render_mode="human")

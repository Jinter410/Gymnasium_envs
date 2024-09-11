from typing import Tuple
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import Gymnasium_envs
import pygame
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

INSTRUCTIONS = {
    "left": [
        "Turn left.",
        "Rotate left.",
        "Take a left turn.",
        "Move leftward.",
        "Steer to the left.",
        "Swing left.",
        "Adjust course to the left.",
        "Head to the left.",
        "Shift to the left.",
        "Angle left."
    ],
    "right": [
        "Turn right.",
        "Rotate right.",
        "Take a right turn.",
        "Move rightward.",
        "Steer to the right.",
        "Swing right.",
        "Adjust course to the right.",
        "Head to the right.",
        "Shift to the right.",
        "Angle right."
    ]
}

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

def get_embeddings(model, tokenizer, sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Moyenne des embeddings de tokens pour chaque phrase
    return embeddings.cpu().numpy()

def generate(how, model, tokenizer, disc_output = 5, n_rays=40, n_crowd=4, interceptor_percentage = 0.5, max_steps = 100, n_data =100, render_mode=None) -> Tuple[np.ndarray, np.ndarray]:
    pygame.init()
    env_name = "Navigation-v0"
    if env_name == "Navigation-v0":
        env = gym.make(env_name, n_rays=n_rays, max_steps=max_steps, render_mode=render_mode)
    else:
        env = gym.make(env_name, n_rays=n_rays, n_crowd=n_crowd, interceptor_percentage=interceptor_percentage, max_steps=max_steps, render_mode=render_mode)

    sentences = INSTRUCTIONS[how]
    embeddings = get_embeddings(model, tokenizer, sentences)

    observation_size = env.observation_space.shape[0] + embeddings.shape[1]
    rot_size = disc_output * 2
    X = np.zeros((n_data, observation_size))
    y = np.zeros((n_data, rot_size))

    n_steps = np.random.randint(2, 5)
    for _ in tqdm(range(n_data)):
        emb_i = np.random.choice(embeddings.shape[0], 1)
        r_emb = embeddings[emb_i]
        observation = env.reset()
        for i in range(n_steps):
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            
        inertia_angle = observation[1] + np.pi
        robot_x = env.get_wrapper_attr('agent_pos')[0]
        robot_y = env.get_wrapper_attr('agent_pos')[1]
        x_rot, y_rot, radius, angle = generate_one_turn(robot_x, robot_y, how, inertia_angle)
        # If the turn is out of bounds
        while np.any(np.abs(x_rot) > env.get_wrapper_attr('WIDTH') - 2 * env.get_wrapper_attr('PHS')) or np.any(np.abs(y_rot) > env.get_wrapper_attr('HEIGHT') - 2 * env.get_wrapper_attr('PHS')):
            x_rot, y_rot, radius, angle = generate_one_turn(robot_x, robot_y, how, inertia_angle)
        
        x_rot -= robot_x
        y_rot -= robot_y
        # Scattering
        indices = np.linspace(0, len(x_rot) - 1, disc_output, dtype=int)
        x_rot = x_rot[indices]
        y_rot = y_rot[indices]
        # plt.plot(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')
        # plt.plot(robot_x, robot_y, 'go', markersize=10)
        # plt.arrow(robot_x, robot_y, 2 * np.cos(inertia_angle), 2 * np.sin(inertia_angle),
        #           head_width=0.5, head_length=0.5, fc='blue', ec='blue')
        # plt.axis([-env.WIDTH, env.WIDTH, -env.HEIGHT, env.HEIGHT])
        # plt.show()
        # quit()
        X[_] = np.concat([observation.flatten(), r_emb.flatten()])
        y[_] = np.concat([x_rot, y_rot])
        n_steps = np.random.randint(2, 5)
        
    env.close()
    return X, y

if __name__ == "__main__":
    model_name = "thenlper/gte-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    X_left,y_left = generate("left", model, tokenizer, disc_output = 5, n_rays= 40, max_steps=10, n_data=20000)
    X_right,y_right = generate("right", model, tokenizer, disc_output = 5, n_rays= 40, max_steps=10, n_data=20000)
    X = np.concat([X_left, X_right])
    y = np.concat([y_left, y_right])
    np.save("./data/X.npy", X)
    np.save("./data/y.npy", y)
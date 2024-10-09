from typing import Tuple
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import Gymnasium_envs
import pygame
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils import generate_one, get_embeddings

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
    ],
    "forward": [
        "Move forward.",
        "Go straight.",
        "Proceed straight ahead.",
        "Advance forward.",
        "Head straight.",
        "Continue forward.",
        "Move ahead.",
        "Keep going straight.",
        "Walk straight ahead.",
        "Progress forward."
    ]
}



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
            
        inertia_angle = observation[1]
        robot_x = env.get_wrapper_attr('agent_pos')[0]
        robot_y = env.get_wrapper_attr('agent_pos')[1]
        x_rot, y_rot, radius, angle = generate_one(robot_x, robot_y, how, inertia_angle)
        # If the turn is out of bounds
        half_width = env.get_wrapper_attr('WIDTH') / 2
        half_height = env.get_wrapper_attr('HEIGHT') / 2
        phs = env.get_wrapper_attr('PHS')

        while np.any(x_rot < -half_width + phs) or np.any(x_rot > half_width - phs) or \
            np.any(y_rot < -half_height + phs) or np.any(y_rot > half_height - phs):
            x_rot, y_rot, radius, angle = generate_one(robot_x, robot_y, how, inertia_angle)
        
        # Scattering
        indices = np.linspace(0, len(x_rot) - 1, disc_output, dtype=int)
        x_rot = x_rot[indices]
        y_rot = y_rot[indices]

        #############################
        # plt.plot(x_rot, y_rot, label=f'Robot {i+1}: rayon={radius:.2f}, angle={angle:.2f}°')
        # plt.plot(robot_x, robot_y, 'go', markersize=10)
        # plt.arrow(robot_x, robot_y, 2 * np.cos(inertia_angle), 2 * np.sin(inertia_angle),
        #         head_width=0.5, head_length=0.5, fc='blue', ec='blue')

        # # Annoter chaque point avec son numéro
        # for idx, (x, y) in enumerate(zip(x_rot, y_rot)):
        #     plt.text(x, y, str(idx + 1), fontsize=12, color='red', ha='center', va='center')

        # plt.axis([-env.WIDTH, env.WIDTH, -env.HEIGHT, env.HEIGHT])
        # # # Invert Y axis to match pygame's coordinate system
        # # plt.gca().invert_yaxis()
        # plt.show()
        # quit()
        #############################
        
        X[_] = np.concatenate([observation.flatten(), r_emb.flatten()])
        zipped_points = np.array([coord for pair in zip(x_rot, y_rot) for coord in pair])
        y[_] = zipped_points  # Utiliser les coordonnées zippées pour y
        n_steps = np.random.randint(2, 5)
        
    env.close()
    return X, y

if __name__ == "__main__":
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    X_left,y_left = generate("left", model, tokenizer, disc_output = 5, n_rays= 40, max_steps=10, n_data=10000)
    X_right,y_right = generate("right", model, tokenizer, disc_output = 5, n_rays= 40, max_steps=10, n_data=10000)
    X_forward,y_forward = generate("forward", model, tokenizer, disc_output = 5, n_rays= 40, max_steps=10, n_data=10000)
    X = np.concat([X_left, X_right, X_forward])
    y = np.concat([y_left, y_right, y_forward])
    np.save("./data/X.npy", X)
    np.save("./data/y.npy", y)
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Gymnasium_envs
from transformers import AutoTokenizer, AutoModel
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
import pygame
from typing import Tuple
from tqdm import tqdm

from models import MLP

def c2p(cart):
        r = np.linalg.norm(cart)
        theta = np.arctan2(cart[1], cart[0])
        return np.array([r, theta])

# Fonction pour charger le modèle à partir d'un checkpoint
def load_model(checkpoint_path, input_size, hidden_size, output_size):
    model = MLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def get_embeddings(model, tokenizer, sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def generate_turn_points(observation, embedding, model):
    # Concaténer les observations et l'embedding
    input_data = np.concatenate([observation.flatten(), embedding.flatten()])
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Ajouter une dimension pour le batch
    
    # Prédire les points de sortie
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).numpy()
    
    return output

# Fonction principale
def main(checkpoint_path, env_name="Navigation-v0", n_rays=40, max_steps=100):

    model_name = "thenlper/gte-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModel.from_pretrained(model_name)
    pygame.init()
    
     # Initialiser l'environnement
    env = gym.make(env_name, n_rays=n_rays, max_steps=max_steps, render_mode=None)
    
    # Charger le modèle MLP depuis le checkpoint
    input_size = env.observation_space.shape[0] + 384 
    fc_size = 128
    output_size = 10  # 5 points x et 5 points y
    
    mlp_model = load_model(checkpoint_path, input_size, fc_size, output_size)
    
    # Boucle principale
    done = False
    # observation, info = env.reset()
    # Load model
    kwargs = {'n_rays': 40, 'max_steps': 150}

    kwargs['render_mode'] = 'human'
    model_path = "./navigation/results/sac_Nav40Rays/models/rl_model_999984_steps.zip"
    env_path = "./navigation/results/sac_Nav40Rays/models/rl_model_vecnormalize_999984_steps.pkl"
    env = VecNormalize.load(env_path, make_vec_env(env_name, n_envs=1, env_kwargs=kwargs))
    model = SAC.load(model_path, device='cpu')
    observation = env.reset()
    
    objectives = []
    while 1:
        action, _states = model.predict(observation, deterministic=True)
        # After receiving the observation from env.step()
        observation, reward, done, info = env.step(action)
        if done:
            env.envs[0].unwrapped.set_coordinate_list([])

        # Unnormalize the observation
        unnormalized_obs = env.unnormalize_obs(observation)
        
        if len(objectives) > 0:
            curr_objective = objectives[0].copy()

            if (np.linalg.norm(env.envs[0].get_wrapper_attr('agent_pos') - curr_objective) < 0.4):
                objectives = objectives[1:]
                
            curr_objective -= env.envs[0].get_wrapper_attr('agent_pos')

            curr_objective_polar = c2p(curr_objective)

            # Inject the unnormalized waypoint into the unnormalized observation
            unnormalized_obs[0][2:4] = curr_objective_polar

            # Re-normalize the observation
            observation = env.normalize_obs(unnormalized_obs)
            # time.sleep(1)
            
        env.render()  # Affiche l'environnement

        # Vérifier si l'utilisateur appuie sur "C"
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                instruction = input("Enter something to do :")
                # Obtenir l'embedding de l'instruction
                embedding = get_embeddings(text_model, tokenizer, [instruction])[0]
                
                # Générer les points de sortie avec le modèle MLP
                output = generate_turn_points(observation, embedding, mlp_model)
                x_points = output[::2]
                y_points = output[1::2]
                x_robot, y_robot = env.envs[0].get_wrapper_attr('agent_pos')
                
                x_points += x_robot
                y_points += y_robot
                coordinates = list(zip(x_points, y_points))
                objectives = np.array(coordinates[1:])
                # Remove points that are outside the map
                objectives = objectives[objectives[:, 0] > -10]
                objectives = objectives[objectives[:, 0] < 10]

                # Set the coordinate list in the environment
                env.envs[0].unwrapped.set_coordinate_list(coordinates)
                env.render()

                ############################
                # x_points_plot = x_robot + x_points
                # y_points_plot = y_robot + y_points
                # # Obtenir l'inertie (angle) du robot
                # inertia_angle = observation[0][1] 

                # # Plot simple avec matplotlib
                # plt.figure(figsize=(8, 8))
                # plt.plot(x_points_plot, y_points_plot, 'ro-', label="Trajectoire prédite")
                # plt.plot(x_robot, y_robot, 'go', markersize=10, label="Position du Robot")

                # # Ajouter des numéros aux points
                # for i, (x, y) in enumerate(zip(x_points_plot, y_points_plot)):
                #     plt.text(x, y, f'{i+1}', fontsize=12, ha='right')

                # # Dessiner une flèche pour l'inertie
                # arrow_length = 2
                # plt.arrow(x_robot, y_robot, arrow_length * np.cos(inertia_angle), arrow_length * np.sin(inertia_angle),
                #         head_width=0.5, head_length=0.5, fc='blue', ec='blue', label="Inertie")

                # # Configurer le plot
                # plt.xlabel('Position X')
                # plt.ylabel('Position Y')
                # plt.title('Trajectoire du Robot avec Inertie')
                # plt.grid(True)
                # plt.legend()
                # plt.xlim([-20, 20])
                # plt.ylim([-20, 20])
                # # Invert Y axis to match pygame's coordinate system
                # plt.gca().invert_yaxis()
                # plt.show()
                ############################

                
                

    env.close()

# Exemple d'appel à la fonction principale
if __name__ == '__main__':
    checkpoint_path = './models/128_neur/model_epoch_40.pth'  # Remplacer par le chemin de votre checkpoint
    main(checkpoint_path)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Gymnasium_envs
from transformers import AutoTokenizer, AutoModel
import gymnasium as gym
import pygame
from typing import Tuple
from tqdm import tqdm

from models import MLP

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
    input_data = np.concatenate([observation, embedding], axis=-1)
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
    env = gym.make(env_name, n_rays=n_rays, max_steps=max_steps, render_mode='human')
    
    # Charger le modèle MLP depuis le checkpoint
    input_size = env.observation_space.shape[0] + 384 
    hidden_size = 64
    output_size = 10  # 5 points x et 5 points y
    
    mlp_model = load_model(checkpoint_path, input_size, hidden_size, output_size)
    
    # Boucle principale
    done = False
    observation, info = env.reset()
    
    while not done:
        action = env.action_space.sample()  # Action aléatoire (peut être remplacé par un modèle de contrôle)
        observation, reward, done, truncated, info = env.step(action)
        
        env.render()  # Affiche l'environnement
        
        # Vérifier si l'utilisateur appuie sur "C"
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                print("En pause, entrez une instruction :")
                instruction = input("Instruction : ")
                
                # Obtenir l'embedding de l'instruction
                embedding = get_embeddings(text_model, tokenizer, [instruction])[0]
                
                # Générer les points de sortie avec le modèle MLP
                output = generate_turn_points(observation, embedding, mlp_model)
                x_points = output[::2]
                y_points = output[1::2]
                coordinates = list(zip(x_points, y_points))

                # x_robot, y_robot = env.get_wrapper_attr('agent_pos')
                # x_points += x_robot
                # y_points += y_robot
                
                env.set_coordinate_list(coordinates)
                env.render()
                

    env.close()

# Exemple d'appel à la fonction principale
if __name__ == '__main__':
    checkpoint_path = './models/run1/model_epoch_20.pth'  # Remplacer par le chemin de votre checkpoint
    main(checkpoint_path)
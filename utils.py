import numpy as np
import pygame
import torch


def manual_action() -> tuple:
    keys = pygame.key.get_pressed()
    if not np.any(keys):
        action = (0, 0)
    else:
        linear_velocity = 1
        angle = None
        
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            angle = -np.pi / 4 
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            angle = -3 * np.pi / 4  
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            angle = np.pi / 4  
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            angle = 3 * np.pi / 4  
        elif keys[pygame.K_UP]:
            angle = -np.pi / 2 
        elif keys[pygame.K_DOWN]:
            angle = np.pi / 2  
        elif keys[pygame.K_LEFT]:
            angle = np.pi  
        elif keys[pygame.K_RIGHT]:
            angle = 0  
            
        if angle is not None:
            action = (linear_velocity, angle)
    return action

def generate_right_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    # x = start_x + radius * (1 - np.cos(t)) * strength # NOTE : Original right turn with pyplot's Y axis
    x = start_x - radius * (1 - np.cos(t)) * strength # NOTE : Right turn with inverted Y axis for pygame
    y = start_y + radius * np.sin(t)
    return x, y

def generate_left_turn(start_x, start_y, radius, angle, strength, steps=100):
    t = np.linspace(0, np.radians(angle), steps)
    # x = start_x - radius * (1 - np.cos(t)) * strength # NOTE : Original left turn with pyplot's Y axis
    x = start_x + radius * (1 - np.cos(t)) * strength # NOTE : Left turn with inverted Y axis for pygame
    y = start_y + radius * np.sin(t)
    return x, y

def rotate_points(x, y, rotation_angle):
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle
    return x_rot, y_rot

def generate_one_turn(robot_x, robot_y, how, inertia_angle, radius_min=2, radius_max=15, angle_min=70, angle_max=110, strength_min=0.5, strength_max=2, shift = False) -> tuple:
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

    if shift:
        x_rot += robot_x
        y_rot += robot_y

    return x_rot, y_rot, radius, angle

def get_embeddings(model, tokenizer, sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Moyenne des embeddings de tokens pour chaque phrase
    return embeddings.cpu().numpy()
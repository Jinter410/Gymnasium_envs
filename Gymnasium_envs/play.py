import gymnasium as gym
import numpy as np
import Gymnasium_envs
import pygame

def main(manual_control=False, n_rays=180, n_crowd=4, interceptor_percentage = 0.5, max_steps=100, render_mode="human"):
    pygame.init()
    env_name = "Navigation-v0"
    if env_name == "Navigation-v0":
        env = gym.make(env_name, n_rays=n_rays, max_steps=max_steps, render_mode=render_mode)
    else:
        env = gym.make(env_name, n_rays=n_rays, n_crowd=n_crowd, interceptor_percentage=interceptor_percentage, max_steps=max_steps, render_mode=render_mode)
    observation = env.reset()
    done = False
    truncated = False
    clock = pygame.time.Clock()

    while not (done or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if manual_control:
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
        else:
            action = env.action_space.sample()

        observation, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    for i in range(100):
        main(manual_control=False, n_rays= 40, n_crowd=10, interceptor_percentage=1, max_steps=700, render_mode="human")

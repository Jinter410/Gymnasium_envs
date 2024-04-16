import gymnasium as gym
import pygame
import sys
sys.path.append("C:/Users/jorco/Desktop/These/Code/Simulation")
from Gymnasium_envs.gym_StaticCrowdEnv import StaticCrowdEnv

def main(manual_control=False, n_rays=180, n_crowd=4, render_mode="human"):
    pygame.init()
    env = gym.make("StaticCrowd-v0", n_rays=n_rays, n_crowd=n_crowd, render_mode=render_mode)
    observation = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if manual_control:
            keys = pygame.key.get_pressed()
            linear_velocity = 0
            angular_velocity = 0
            
            if keys[pygame.K_UP]:
                angular_velocity = -1 
            if keys[pygame.K_DOWN]:
                angular_velocity = 1
            
            if keys[pygame.K_LEFT]:
                linear_velocity = -1
            if keys[pygame.K_RIGHT]:
                linear_velocity = 1

            action = (linear_velocity, angular_velocity)
        else:
            action = env.action_space.sample()

        observation, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(manual_control=True, n_rays= 40, n_crowd=40)

import gymnasium as gym
import Gymnasium_envs
import pygame

def main(manual_control=False, n_rays=180, n_crowd=4, interceptor_percentage = 0.5, max_steps=100, render_mode="human"):
    pygame.init()
    env = gym.make("ConstantVelocity-v0", n_rays=n_rays, n_crowd=n_crowd, interceptor_percentage=interceptor_percentage, max_steps=max_steps, render_mode=render_mode)
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

if __name__ == "__main__":
    for i in range(100):
        main(manual_control=False, n_rays= 40, n_crowd=10, interceptor_percentage=1, max_steps=70, render_mode="human")

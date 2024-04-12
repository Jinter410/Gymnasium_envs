import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class Ray:
    def __init__(self, angle):
        self.angle = angle
        self.cos = np.cos(angle)
        self.sin = np.sin(angle)

class StaticCrowdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_rays: int,
        n_crowd: int,
        width: int = 20,
        height: int = 20,
        interceptor_percentage: float = 0.5,
        max_steps: int = 100,
        render_mode: str = None,
    ):
        # Environment constants
        self.MAX_EPISODE_STEPS = max_steps
        self.N_CROWD = n_crowd
        self.INTERCEPTOR_PERCENTAGE = interceptor_percentage
        self.N_RAYS = n_rays
        self.RAYS = [Ray(angle) for angle in np.linspace(0, 2 * np.pi, n_rays, endpoint=False)]
        self.WIDTH = width
        self.HEIGHT = height
        self.W_BORDER = self.WIDTH / 2
        self.H_BORDER = self.HEIGHT / 2
        self.MAX_LINEAR_VEL = 3.0 # m/s
        self.MAX_ANGULAR_VEL = 1.5 # rad/s
        self.PHS = 0.4
        self.PRS = 1.4
        self.SCS = 1.9
        # Reward constants
        self.COLLISION_REWARD = -10
        self.Cc = 2 * self.PHS * \
            np.log(-self.COLLISION_REWARD / self.MAX_EPISODE_STEPS + 1)
        self.Cg = -(1 - np.exp(self.Cc / self.SCS)) /\
            np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2)
        self.TASK_COMPLETION_REWARD = -self.COLLISION_REWARD / 2
        # Action space (linear and angular velocity)
        action_bound = np.array([self.MAX_LINEAR_VEL, self.MAX_ANGULAR_VEL])
        self.action_space = spaces.Box(
            low=-action_bound, high=action_bound, shape=action_bound.shape
        )
        # Observation space
        agent_bounds = np.array([self.WIDTH, self.HEIGHT, self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL])
        ray_bounds = np.array([np.sqrt(self.WIDTH**2 + self.HEIGHT**2)] * self.N_RAYS)
        self.observation_space = spaces.Box(
            low=np.concatenate([-agent_bounds, np.full(self.N_RAYS, 0)]),
            high=np.concatenate([agent_bounds, ray_bounds]),
            dtype=np.float32
        )
        # Plotting variables
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        # Ray distances
        angles = np.array([self.RAYS[i].cos, self.RAYS[i].sin] for i in range(self.N_RAYS))
        default_distances = np.min([self.W_BORDER, self.H_BORDER] / np.abs(angles), axis=-1) # Distances to borders
        
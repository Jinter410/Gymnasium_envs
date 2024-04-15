import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

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
        self.RAY_ANGLES = np.linspace(0, 2 * np.pi, n_rays, endpoint=False) + 1e-6 # To avoid div by zero
        self.RAY_COS = np.cos(self.RAY_ANGLES)
        self.RAY_SIN = np.sin(self.RAY_ANGLES)
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

    def reset(self, seed=None, options=None):
        # Seeding
        super().reset(seed=seed)
        self._steps = 0

        # Agent state
        self.agent_pos = np.zeros(2)
        self.agent_vel = np.zeros(2)

        # Goal state
        self.goal_pos = np.random.uniform(
           [-self.W_BORDER + self.PHS, -self.H_BORDER + self.PHS],
           [self.W_BORDER - self.PHS, self.H_BORDER - self.PHS]
        )

        # Crowd state
        self.crowd_poss = np.zeros((self.N_CROWD, 2))
        collision = True 
        while collision:
            self.crowd_poss = np.random.uniform(
                [-self.W_BORDER, -self.H_BORDER],
                [self.W_BORDER, self.H_BORDER],
                (self.N_CROWD, 2)
            )
            # Check for agent, goal and crowd collisions
            collision = np.any(np.linalg.norm(self.crowd_poss - self.agent_pos, axis=1) < self.PRS * 2) or \
                        np.any(np.linalg.norm(self.crowd_poss - self.goal_pos, axis=1) < self.PRS * 2) or \
                        np.any(np.linalg.norm(self.crowd_poss[:, None] - self.crowd_poss[None, :], axis=-1)[np.triu_indices(self.N_CROWD, k=1)] < self.PHS * 2)
        
        observation = self._get_obs()
        # info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, {}
    
    def _get_obs(self):
        print(self.RAY_COS)
        # Vectorized ray distances
        default_distances = np.min([self.W_BORDER / np.abs(self.RAY_COS), self.H_BORDER / np.abs(self.RAY_SIN)], axis=0)
        
        return self.crowd_poss
        # TODO
        D_ortho = np.abs(np.outer(positions_x, sin_angles) - np.outer(positions_y, cos_angles)) 
        
        # Calcul vectorisé des intersections avec les cercles
        intersections = D_ortho <= rayons[:, None]
        
        # Calcul vectorisé de la distance le long du rayon jusqu'à la projection orthogonale
        D_along_ray = np.outer(positions_x, cos_angles) + np.outer(positions_y, sin_angles)
        
        # Calcul vectorisé de la distance du centre du cercle à l'intersection avec le rayon
        D_to_intersection = np.sqrt(np.maximum(rayons[:, None]**2 - D_ortho**2, 0))
        
        # Calcul vectorisé de la distance totale du rayon à l'intersection
        distances = np.where(intersections, D_along_ray - D_to_intersection, np.inf)
        
        # Trouver la distance minimale pour chaque rayon
        min_distances = np.min(np.where(distances > 0, distances, np.inf), axis=0)
        
        # Comparer avec la distance par défaut
        final_distances = np.minimum(min_distances, default_distances)
        
        return final_distances
        

if __name__ == "__main__":
    env = StaticCrowdEnv(n_rays=360, n_crowd=30)
    print(env.reset())
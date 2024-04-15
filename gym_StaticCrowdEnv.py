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
        max_steps: int = 10000,
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
        self.PHS = 0.4
        self.PRS = 1.4
        self.SCS = 1.9
        # Physics
        self._dt = 0.1
        self.MAX_LINEAR_VEL = 3.0 # m/s
        self.MAX_ANGULAR_VEL = 1.5 # rad/s
        self.MAX_GOAL_VEL = 0.5 # m/s
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

        # Episode variables
        self._steps = 0
        self._goal_reached = False
        self._is_collided = False

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
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _get_obs(self):
        # Vectorized ray distances
        default_distances = np.min([
            (self.W_BORDER - np.where(self.RAY_COS > 0, self.agent_pos[0], -self.agent_pos[0])) / np.abs(self.RAY_COS),
            (self.H_BORDER - np.where(self.RAY_SIN > 0, self.agent_pos[1], -self.agent_pos[1])) / np.abs(self.RAY_SIN)
        ], axis=0)
        x_crowd_rel, y_crowd_rel = self.crowd_poss[:, 0] - self.agent_pos[0], self.crowd_poss[:, 1] - self.agent_pos[1]
        orthog_dist = np.abs(np.outer(x_crowd_rel, self.RAY_SIN) - np.outer(y_crowd_rel, self.RAY_COS)) # Orthogonal distances from obstacles to rays
        intersections_mask = orthog_dist <= self.PHS # Mask for intersections
        along_dist = np.outer(x_crowd_rel, self.RAY_COS) + np.outer(y_crowd_rel, self.RAY_SIN) # Distance along ray to orthogonal projection
        orthog_to_intersect_dist = np.sqrt(np.maximum(self.PHS**2 - orthog_dist**2, 0)) # Distance from orthogonal projection to intersection
        intersect_distances = np.where(intersections_mask, along_dist - orthog_to_intersect_dist, np.inf) # Distances from ray to intersection if existing
        min_intersect_distances = np.min(np.where(intersect_distances > 0, intersect_distances, np.inf), axis=0) # Minimum distance for each ray to have the closest intersection
        ray_distances = np.minimum(min_intersect_distances, default_distances) # If no intersection, rays collide with border
        
        # Agent state
        agent_state = np.concatenate([self.agent_pos, self.agent_vel])
        # Goal relative position
        goal_rel_pos = self.goal_pos - self.agent_pos
        # Store ray distances
        self.ray_distances = ray_distances

        return np.concatenate([ray_distances, agent_state, goal_rel_pos])
        
    def step(self, action):
        # Update agent state
        self.agent_vel = np.clip(action, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        self.agent_pos += self.agent_vel * self._dt
        
        terminated = self._terminate()
        reward = self._get_reward()
        info = self._get_info()
        observation = self._get_obs()
        self._steps += 1

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info
    
    def _get_reward(self):
        if self._goal_reached:
            return self.TASK_COMPLETION_REWARD
        if self._is_collided:
            return self.COLLISION_REWARD
        
        dg = np.linalg.norm(self.agent_pos - self.goal_pos)
        # Goal distance reward
        Rg = -self.Cg * dg
        # Crowd distance reward
        dist_crowd = np.linalg.norm(self.agent_pos - self.crowd_poss, axis=-1)
        Rc = np.sum(
            (1 - np.exp(self.Cc / dist_crowd)) *\
            (dist_crowd < [self.SCS + self.PHS] * self.N_CROWD)
        )
        # Walls distance reward
        dist_walls = np.array([
            self.W_BORDER - abs(self.agent_pos[0]),
            self.H_BORDER - abs(self.agent_pos[1])
        ])
        Rw = np.sum(
            (1 - np.exp(self.Cc / dist_walls)) * (dist_walls < self.PHS * 2)
        )
        return Rg + Rc + Rw
    
    def _terminate(self):
        # Check for collisions with crowd
        if np.any(np.linalg.norm(self.agent_pos - self.crowd_poss, axis=1) < self.PHS * 2):
            self._is_collided = True

        # Check for collisions with walls
        if np.any(np.abs(self.agent_pos) > np.array([self.W_BORDER, self.H_BORDER]) - self.PHS):
            self._is_collided = True

        # Check for goal reached
        if (np.linalg.norm(self.agent_pos - self.goal_pos) < self.PHS) and \
            (np.linalg.norm(self.agent_vel) < self.MAX_GOAL_VEL):
            self._goal_reached = True
        return self._goal_reached or self._is_collided or self._steps >= self.MAX_EPISODE_STEPS
    
    def _get_info(self):
        return {
            "goal_reached": self._goal_reached, 
            "collision": self._is_collided, 
            "steps": self._steps, 
            "dist_to_goal": np.linalg.norm(self.agent_pos - self.goal_pos)
        }
    
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.WIDTH * 50, self.HEIGHT * 50))
            self.clock = pygame.time.Clock()

        self.window.fill((245,245,245))

        # Agent
        agent_color = (0, 255, 0)
        agent_center = (
            int((self.agent_pos[0] + self.W_BORDER) * 50),
            int((self.agent_pos[1] + self.H_BORDER) * 50)
        )
        agent_radius = self.PHS * 50
        pygame.draw.circle(self.window, agent_color, agent_center, agent_radius)

        # Goal
        goal_color = (0, 0, 255)
        goal_pos_x = int((self.goal_pos[0] + self.W_BORDER) * 50)
        goal_pos_y = int((self.goal_pos[1] + self.H_BORDER) * 50)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y - 10), (goal_pos_x + 10, goal_pos_y + 10), 2)
        pygame.draw.line(self.window, goal_color, (goal_pos_x - 10, goal_pos_y + 10), (goal_pos_x + 10, goal_pos_y - 10), 2)

        # Crowd
        crowd_color = (255, 0, 0)  # Red
        for pos in self.crowd_poss:
            crowd_center = (
            int((pos[0] + self.W_BORDER) * 50),
            int((pos[1] + self.H_BORDER) * 50)
            )
            # Physical space
            crowd_phs = int(self.PHS * 50)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_phs)

            # Personal space
            crowd_prs = int(self.PRS * 50)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_prs, 2)

            # Draw dotted circle
            crowd_scs = int(self.SCS * 50)
            pygame.draw.circle(self.window, crowd_color, crowd_center, crowd_scs, 1)

        
        # Wall borders
        wall_color = (0, 0, 0)
        pygame.draw.rect(self.window, wall_color, (self.PHS * 50, self.PHS * 50, (self.WIDTH - 2 * self.PHS) * 50, (self.HEIGHT - 2 * self.PHS) * 50), 1)

        # Rays
        ray_color = (128, 128, 128)  # Gray
        for angle, distance in zip(self.RAY_ANGLES, self.ray_distances):
            end_x = agent_center[0] + distance * 50 * np.cos(angle)
            end_y = agent_center[1] + distance * 50 * np.sin(angle)
            pygame.draw.line(self.window, ray_color, agent_center, (int(end_x), int(end_y)), 1)

        pygame.display.flip()  # Update the full display surface to the screen
        self.clock.tick(60)  # Limit frames per second
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
if __name__ == "__main__":
    env = StaticCrowdEnv(n_rays=180, n_crowd=4, render_mode="human")
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
    env.close()

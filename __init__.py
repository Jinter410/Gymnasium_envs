from gymnasium.envs.registration import register
from .gym_StaticCrowdEnv import StaticCrowdEnv

register(
    id="StaticCrowd-v0",
    entry_point="Gymnasium_envs.gym_StaticCrowdEnv:StaticCrowdEnv",
    max_episode_steps=150,
)

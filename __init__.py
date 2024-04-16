from gymnasium.envs.registration import register

register(
     id="StaticCrowd-v0",
     entry_point="StaticCrowd.envs:StaticCrowdEnv",
     max_episode_steps=300,
)
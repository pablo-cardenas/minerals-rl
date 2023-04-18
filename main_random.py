from itertools import count
from environment import MineralEnv

import gymnasium as gym


env = MineralEnv(num_queues=(3, 3), num_trucks=5, num_blocks=100, render_mode="human")

for episode in count():
    observation, info = env.reset(seed=42)

    for t in count():
        # Select an action
        #action = env.action_space.sample()
        action = int(input())

        # Perform action in the enviroment
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        

        if terminated or truncated:
            break
        observation = next_observation

    print(f"{episode=}, {t+1=}")

env.close()

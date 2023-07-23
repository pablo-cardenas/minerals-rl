from itertools import count
from env import MineralEnv

import gymnasium as gym


env = MineralEnv(render_mode="human")
#env = MineralEnv()

for episode in count():
    observation, info = env.reset(seed=42)

    for t in count():
        # Select an action
        action = env.action_space.sample()
        #action = int(input("Ingrese acción: "))

        # Perform action in the enviroment
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break
        observation = next_observation

    print(f"{episode=}, {t+1=}")

env.close()

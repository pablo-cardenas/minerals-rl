import gymnasium as gym
from model import QLearning
import matplotlib.pyplot as plt
import numpy as np
from environment import MineralEnv
from pprint import pprint
from itertools import count


env = MineralEnv(num_queues=(2, 2), num_trucks=2, num_blocks=5)
#env = gym.make('Blackjack-v1', render_mode="rgb_array")


print('##########')
print('# Empezo #')
print('##########')
model = QLearning(env)
model.learn(alpha=0.1, gamma=1, epsilon=0.1,total_timesteps=10000)

for episode in count():
    observation, info = env.reset(seed=42)
    print(f"{observation=}")

    for t in count():
        # Select an action
        action, _ = model.predict(observation)

        # Perform action in the enviroment
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        print(f"{action=}, {observation=}")

        if terminated or truncated:
            break
        observation = next_observation

    print(f"{episode=}, {t+1=}")

env.close()

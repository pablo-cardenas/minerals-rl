import math
import os
import random
from collections import deque
from collections import namedtuple
from itertools import count
from environment import MineralEnv
import matplotlib.pyplot as plt


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    """My implementation of DQN


    :param n_observations: TODO
    :param n_actions: TODO
    """

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()

        self._n_observations = n_observations
        self._n_actions = n_actions

        self._linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_relu_stack(x)


num_queues = (3,3)
num_trucks = 5
num_blocks = 100
env = MineralEnv(num_queues=num_queues, num_trucks=num_trucks, num_blocks=num_blocks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env.metadata["render_fps"] = 120

policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
if os.path.exists("weight2.pt"):
    target_net.load_state_dict(torch.load("weight2.pt"))
    print("loaded model")

policy_net.load_state_dict(target_net.state_dict())
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(policy_net.parameters(), lr=0.001, amsgrad=True)

Transition = namedtuple(
    "Transition",
    ("observation", "action", "reward", "next_observation", "terminated"),
)

if os.path.exists("memory2.pt"):
    replay_memory_batch = torch.load("memory2.pt")
    replay_memory = deque(
        [Transition(*x) for x in zip(*replay_memory_batch)], maxlen=100000
    )
    print("loaded replay memory")
else:
    replay_memory = deque(maxlen=100000)


try:
    Gs = []
    for i_episode in count():
        G = 0
        observation, info = env.reset()
        print(target_net(torch.tensor(observation, dtype=torch.float32, device=device)).detach())
        print(policy_net(torch.tensor(observation, dtype=torch.float32, device=device)).detach())
        for t in count():
            # Select an action
            epsilon = 0.05 + (0.999 - 0.05) * math.exp(
                -len(replay_memory) / 1000
            )
            action = (
                env.action_space.sample()
                if random.random() < epsilon
                else policy_net(torch.tensor(observation, dtype=torch.float32, device=device))
                .argmax()
                .item()
            )

            # Perform action in the enviroment
            next_observation, reward, terminated, truncated, info = env.step(
                action
            )

            transition = Transition(
                observation=torch.tensor(
                    observation, device=device, dtype=torch.float32
                ),
                action=torch.tensor(action, device=device, dtype=torch.int64),
                reward=torch.tensor(
                    reward, device=device, dtype=torch.float32
                ),
                next_observation=torch.tensor(
                    next_observation, device=device, dtype=torch.float32
                ),
                terminated=torch.tensor(
                    terminated, device=device, dtype=torch.bool
                ),
            )
            replay_memory.append(transition)

            if len(replay_memory) > 128:
                batch_sample = random.sample(replay_memory, 128)
                batch = Transition(
                    *[torch.stack(x) for x in zip(*batch_sample)]
                )

                loss = criterion(
                    policy_net(batch.observation)
                    .gather(1, batch.action.unsqueeze(1))
                    .squeeze(1),
                    (batch.reward - 0 * torch.abs(batch.observation[:, 0]))
                    + ~batch.terminated
                    * 0.99
                    * target_net(batch.next_observation).max(1)[0].detach(),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            policy_net_state_dict = policy_net.state_dict()
            target_net_state_dict = target_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] += 0.005 * (
                    policy_net_state_dict[key] - target_net_state_dict[key]
                )

            target_net.load_state_dict(target_net_state_dict)

            G = reward - 0 * abs(observation[0]) + 0.999 * G
            observation = next_observation
            if terminated or truncated:
                break

        print(f"{i_episode=} {t=} {len(replay_memory)=} {G=}")
        Gs.append(G)

except KeyboardInterrupt:
    pass
finally:
    print('hola')
    plt.plot(Gs)
    plt.savefig('G.png')
    torch.save(target_net.state_dict(), "weight2.pt")
    replay_memory_batch = Transition(
        *(torch.stack(x) for x in zip(*replay_memory))
    )
    torch.save(replay_memory_batch, "memory2.pt")

env.close()

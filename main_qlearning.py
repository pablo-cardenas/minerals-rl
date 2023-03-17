import gym
from model import QLearning
import matplotlib.pyplot as plt
import numpy as np
from enviroment import MineralEnv
from pprint import pprint


env = MineralEnv(num_queues=(2, 2), num_trucks=2, num_blocks=5)
#env = gym.make('Blackjack-v1', render_mode="rgb_array")


print('##########')
print('# Empezo #')
print('##########')
model = QLearning(env)
model.learn(alpha=0.1, gamma=1, epsilon=0.1,total_timesteps=10000)

x = np.zeros((32, 11))
pprint(model.q)
for i in range(32):
    for j in range(11):
        #action, _state = model.predict((i, j, 1))
        action = max(model.q[(i,j,1)][0], model.q[(i,j,1)][1])
        x[i, j] = action
plt.imshow(x, origin='lower')
plt.show()
exit()

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, terminate, truncate, info = env.step(action)
    env.render()
    if terminate or truncate:
        break

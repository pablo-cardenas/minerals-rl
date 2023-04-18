import gymnasium as gym
from collections import defaultdict
import numpy as np
from numpy.random import default_rng

def epsilon_greedy(q_s, epsilon, rng):
    i_max = np.argmax(q_s)
    
    random = rng.uniform()
    if random > epsilon:
        return i_max
    else:
        return rng.integers(len(q_s))

def greedy(q_s):
    return  np.argmax(q_s)
    

class QLearning:
    def __init__(self, env):
        self.env = env

    def learn(self, alpha, gamma, epsilon, total_timesteps):
        rng = default_rng()
        self.q = defaultdict(lambda : [0, 0])

        for _ in range(total_timesteps):
            print('# Episodio Terminado #')
            obs, info = self.env.reset()
            action = epsilon_greedy(self.q[obs], epsilon, rng)
            print('action = ', action)
            #print(obs, action)

            while True:
                obs1, rew, term, trunc, info = self.env.step(action)
                if term or trunc:
                    self.q[obs][action] = self.q[obs][action] + alpha * (rew - self.q[obs][action])
                    break

                action1 = epsilon_greedy(self.q[obs1], epsilon, rng)
                print('action = ', action1)
                self.q[obs][action] = self.q[obs][action] + alpha * (rew + gamma*self.q[obs1][action1] - self.q[obs][action])

                obs = obs1
                action = action1


    def predict(self, obs):
        return greedy(self.q[obs]), obs

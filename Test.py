import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten

from collections import deque

import random

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
n_states = 4

D = deque()

q_matrix = np.zeros([n_states, n_actions])

# Observe initial state s

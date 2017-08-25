import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten

from collections import deque

import random

# Initialize environment
env = gym.make('CartPole-v0')

# Neural Network
model = Sequential()
model.add(Dense(32, input_shape=(4,), activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')

# Parameters
n_actions = env.action_space.n
n_states = 4

epsilon = 0.7  # Probability of choosing a random action

generations = 200

# Initialize replay memory D
D = deque()

# Initialize action-value function Q with random weights
q_matrix = np.zeros([n_states, n_actions])

# Observe initial state
state = env.reset()

for i in range(generations):
    done = False
    print(state)
    while not done:
        if random.random() <= epsilon:
            action = random.randint(0, 1)
        else:
            pass

        done = True

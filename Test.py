import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten

from collections import deque

import random

# Initialize environment
env = gym.make('CartPole-v0')

# Parameters
n_actions = env.action_space.n
n_states = 4
mb_size = 5

# Neural Network
model = Sequential()
model.add(Dense(32, input_shape=(4,), kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(n_actions, kernel_initializer='uniform', activation='linear'))
model.compile(optimizer='adam', loss='mse')

epsilon = 0.7  # Probability of choosing a random action
gamma = 0.9  #

generations = 200

# Initialize replay memory D
D = deque()

# Initialize action-value function Q with random weights
q_matrix = np.zeros([n_states, n_actions])

# TODO: Consider adding dimension to input and state

# Observe initial state
state = env.reset()
state = np.expand_dims(state, axis=0)

for i in range(generations):
    done = False

    tot_reward = 0

    while not done:
        env.render()
        # Select an action
        if random.random() <= epsilon:  # With probability epsilon select a random action
            action = random.randint(0, 1)
        else:  # Otherwise select a = argmax(Q(s,a'))
            Q = model.predict(state)
            action = np.argmax(Q)

        # Carry out action and observe reward and new state
        state_new, reward, done, info = env.step(action)
        state_new = np.expand_dims(state_new, axis=0)

        # Store experience <s, a, r, s', done> in replay memory D
        D.append((state, action, reward, state_new, done))
        tot_reward += reward
        state = state_new

    print(tot_reward)

    env.reset()

    # Sample random transitions <ss, aa, rr, ss'> from replay memory D
    mb_size = round(tot_reward)
    minibatch = random.sample(D, mb_size)

    inputs = np.zeros((mb_size, state.shape[1]))
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(0, mb_size):
        mb_state = minibatch[i][0]
        mb_action = minibatch[i][1]
        mb_reward = minibatch[i][2]
        mb_state_new = minibatch[i][3]
        mb_done = minibatch[i][4]

        targets[i] = model.predict(mb_state)
        Q_sa = model.predict(mb_state)

        # Calculate target for each minibatch transition
        if mb_done:  # if ss' is terminal state then tt = rr
            targets[i][mb_action] = mb_reward
        else:  # Otherwise tt = rr + ymax(Q(ss', aa,)
            targets[i][mb_action] = mb_reward + gamma * np.argmax(Q_sa)

        # Train the Q network using (tt - Q(ss, aa))^2 as loss
        model.train_on_batch(inputs, targets)

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from collections import deque

import random

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
n_states = 4

run_time = 200

epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward
mb_size = 50  # Minibatch size

model = Sequential()

model.add(Dense(10, input_shape=(2,) + env.observation_space.shape,
                kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(n_actions, kernel_initializer='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam')

D = deque()

if __name__ == '__main__':
    q_matrix = np.zeros([n_states, n_actions])

    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)  # Maybe make 4 obs

    done = False

    for i in range(run_time):
        env.render()

        if np.random.rand() <= epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]  # TODO: Test without size
        else:
            print(obs.shape, obs)
            Q = model.predict(state)  # TODO: Test if obs is correct or state
            print(Q.shape, np.argmax(Q), '\n')
            action = np.argmax(Q)  # https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb

        observation_new, reward, done, info = env.step(action)
        obs_new = np.expand_dims(observation_new, axis=0)

        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)

        D.append((state, action, reward, obs_new, done))  # TODO: Check if obs should be state

        state = state_new

        if done:
            env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)

    minibatch = random.sample(D, mb_size)
    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        print(inputs.ndim)
        # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        print(inputs.shape)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)
        print(targets.ndim)
        model.train_on_batch(inputs, targets)

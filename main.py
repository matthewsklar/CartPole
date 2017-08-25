import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten

from collections import deque

import random

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
n_states = 4

run_time = 200

epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward
mb_size = 1  # Minibatch size

model = Sequential()

model.add(Dense(20, input_shape=(2,) + env.observation_space.shape,
                kernel_initializer='uniform', activation='relu'))
model.add(Flatten())
model.add(Dense(18, init='uniform', activation='relu'))
model.add(Dense(n_actions, kernel_initializer='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam')

D = deque()

if __name__ == '__main__':
    q_matrix = np.zeros([n_states, n_actions])

    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)  # Maybe make 4 obs

    done = False

    for j in range(200):
        t = 0
        while not done:
            env.render()

            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]  # TODO: Test without size
            else:
                Q = model.predict(state)
                action = np.argmax(Q)  # https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb

            observation_new, reward, done, info = env.step(action)
            obs_new = np.expand_dims(observation_new, axis=0)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)

            D.append((state, action, reward, state_new, done))

            state = state_new

            t += reward

        env.reset()
        print(D)
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
        minibatch = random.sample(D, mb_size)
        inputs_shape = (mb_size,) + state.shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, env.action_space.n))
        print(t)

        for i in range(0, mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]

            # Build Bellman equation for the Q function
            inputs[i:i+1] = np.expand_dims(state, axis=0)

            targets[i] = model.predict(state)
            Q_sa = model.predict(state_new)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * np.max(Q_sa)

            model.train_on_batch(inputs, targets)

    for i in range(200):
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)

        done = False
        net_reward = 0.0

        while not done:
            #env.render()

            Q = model.predict(state)
            action = np.argmax(Q)

            observation, reward, done, info = env.step(action)
            obs = np.expand_dims(observation, axis=0)
            state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
            net_reward += reward

        print('Game ended! Total reward: {}'.format(net_reward))

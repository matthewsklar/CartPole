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
mb_size = 50

# Neural Network
# Initialize action-value function Q with random weights
model = Sequential()
model.add(Dense(32, input_shape=(2,) + env.observation_space.shape, kernel_initializer='uniform', activation='relu'))
model.add(Flatten())
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(n_actions, kernel_initializer='uniform', activation='linear'))
model.compile(optimizer='adam', loss='mse')

epsilon = 0.7  # Probability of choosing a random action
gamma = 0.9  #

generations = 5000

# Initialize replay memory D
D = deque()

max_reward = 0
best_gen = 0
all_reward = 0

if __name__ == '__main__':
    for i in range(generations):
        # Observe initial state
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)

        done = False

        new_reward = 0

        for t in range(500):
            # Select an action
            if random.random() <= epsilon:  # With probability epsilon select a random action
                action = random.randint(0, 1)
            else:  # Otherwise select a = argmax(Q(s,a'))
                Q = model.predict(state)
                action = np.argmax(Q)

            # Carry out action and observe reward and new state
            observation_new, reward, done, _ = env.step(action)
            obs_new = np.expand_dims(observation_new, axis=0)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)

            # Store experience <s, a, r, s', done> in replay memory D
            D.append((state, action, reward, state_new, done))
            new_reward += reward
            state = state_new

            if done:
                env.reset()
                obs = np.expand_dims(observation, axis=0)
                state = np.stack((obs, obs), axis=1)

        env.reset()

        # Sample random transitions <ss, aa, rr, ss'> from replay memory D
        mb_size = 50

        minibatch = random.sample(D, mb_size)

        inputs = np.zeros(((mb_size,) + state.shape[1:]))
        targets = np.zeros((mb_size, env.action_space.n))

        for j in range(0, mb_size):
            mb_state = minibatch[j][0]
            mb_action = minibatch[j][1]
            mb_reward = minibatch[j][2]
            mb_state_new = minibatch[j][3]
            mb_done = minibatch[j][4]

            inputs[j:j+1] = np.expand_dims(mb_state, axis=0)

            targets[j] = model.predict(mb_state)
            Q_sa = model.predict(mb_state_new)

            # Calculate target for each minibatch transition
            if mb_done:  # if ss' is terminal state then tt = rr
                targets[j][mb_action] = mb_reward
            else:  # Otherwise tt = rr + ymax(Q(ss', aa,)
                targets[j][mb_action] = mb_reward + gamma * np.max(Q_sa)

            # Train the Q network using (tt - Q(ss, aa))^2 as loss
            model.train_on_batch(inputs, targets)

        observation_final = env.reset()
        obs_final = np.expand_dims(observation_final, axis=0)
        state_final = np.stack((obs_final, obs_final), axis=1)

        done = False

        tot_reward = 0

        while not done:
            env.render()

            # Select an action
            Q = model.predict(state_final)
            action = np.argmax(Q)

            # Carry out action and observe reward and new state
            observation_final, reward, done, info = env.step(action)
            obs_final = np.expand_dims(observation_final, axis=0)
            state_final = np.append(np.expand_dims(obs_final, axis=0), state_final[:, :1, :], axis=1)

            tot_reward += reward

        if tot_reward > max_reward:
            max_reward = tot_reward
            best_gen = i

        all_reward += tot_reward
        average_reward = all_reward / (i+1)

        print('Generation {}; Reward: {}; Max Reward: {} on gen {}; Avg Reward: {}'.format(
            i, tot_reward, max_reward, best_gen, average_reward))

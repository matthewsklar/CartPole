import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
n_states = 4

run_time = 200

epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward
mb_size = 50  # Minibatch size

model = Sequential()
model.add(Dense(20, input_shape=(4,), kernel_initializer='uniform', activation='relu'))
model.add(Dense(env.action_space.n, kernel_initializer='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

if __name__ == '__main__':
    q_matrix = np.zeros([n_states, n_actions])

    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)

    done = False

    for i in range(run_time):
        env.render()

        if np.random.rand() <= 1:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            action = None  # https://github.com/llSourcell/deep_q_learning/blob/master/03_PlayingAgent.ipynb

        observation_new, reward, done, info = env.step(action)
        obs_new = np.expand_dims(observation_new, axis=0)

        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)

        state = state_new

        if done:
            env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)


import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
n_states = 4

run_time = 200

epsilon = 0.7  # Probability of doing a random move
gamma = 0.9  # Discounted future reward
mb_size = 50  # Minibatch size

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

        if done:
            env.reset()

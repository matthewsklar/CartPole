import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')


n_actions = env.action_space.n
n_states = 4

if __name__ == '__main__':
    q_matrix = np.zeros([n_states, n_actions])

    print(q_matrix)
import tensorflow as tf
import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    n_episode = 500

    for i in range(n_episode):
        state = env.reset()

        done = False

        while not done:
            env.render()
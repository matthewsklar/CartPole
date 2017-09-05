import gym
import random
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


class DQNAgent:
    def __init__(self, learning_rate=0.001, gamma=0.95):
        """
        Initialize DQNAgent

        Args:
            learning_rate: A float representing the learning rate for the network
            gamma: A float representing the discount factor for Bellman's Function
        """
        # The amount of states in the environment
        self.n_states = env.observation_space.shape

        # The amount of possible actions the Agent can apply to the environment
        self.n_actions = env.action_space.n

        # Memory of (state, action, reward, new_state, done)
        self.memory = deque(maxlen=2000)

        # How much new information overrides old information
        self.learning_rate = learning_rate

        # Gamma (discount factor) determines the importance of estimated future rewards
        self.gamma = gamma

        # The chance of the agent picking a random action
        self.epsilon = 1.0

        self.epsilon_min = 0.01

        # Information about the neural network (model, loss, optimizer)
        self.model = self.create_neural_network()

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_states[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, r_state, r_action, r_reward, r_new_state, r_done):
        """
        Stores the (state, action, reward, new_state, done) in the memory

        Args:
            r_state: A numpy array representing the original state of the environment
            r_action: A numpy int64 representing the action the agent sends to the environment
            r_reward: A float representing the reward the agent receives from the action
            r_new_state: A numpy array representing the state of the environment after the action
            r_done: A boolean representing if the environment finished a game after the action
        """
        self.memory.append((r_state, r_action, r_reward, r_new_state, r_done))

    def act(self, a_state):
        """
        Determine which action the agent should send to the environment

        Args:
            a_state: A numpy array representing the state of the environment

        Returns:
            An integer of either 0 or 1
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            action_values = self.model.predict(a_state)

            return np.argmax(action_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for mb_state, mb_action, mb_reward, mb_state_new, mb_done in minibatch:
            target = mb_reward

            if not mb_done:
                # Bellman's Function
                target = mb_reward + self.gamma * np.amax(self.model.predict(mb_state_new)[0])
                # print(target)
            q = self.model.predict(mb_state)
            q[0][mb_action] = target

            self.model.fit(mb_state, q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.995


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = DQNAgent()

    for e in range(500):
        # The initial observed state of the environment
        state = env.reset()
        state = np.expand_dims(state, axis=0)

        for t in range(500):
            env.render()

            action = agent.act(state)

            state_new, reward, done, _ = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)

            agent.remember(state, action, reward, state_new, done)

            state = state_new

            if done:
                print('Episode {}, reward: {}'.format(e, t))
                break

        agent.replay(np.amin((len(agent.memory), 50)))  # TODO: Increase

import gym
import random

import tensorflow as tf
import numpy as np

from collections import deque


class DQNAgent:
    def __init__(self, learning_rate=0.001, gamma=0.95):
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

        # A placeholder for the networks input
        self.input = tf.placeholder(tf.float32, [None, self.n_states[0]])

        # A placeholder for the networks expected output
        self.targets = tf.placeholder(tf.float32, [None, self.n_actions])

        # Information about the neural network (model, loss, optimizer)
        self.neural_network = self.create_neural_network()

        # Model of neural network
        self.model = self.neural_network[0]

        # Loss of the neural network
        self.loss = self.neural_network[1]

        # Optimizer to train the neural network
        self.optimizer = self.neural_network[2]

        print('State Shape: %d\tAction Shape: %d' % (self.n_states[0], self.n_actions))

    def create_neural_network(self):
        """
        Build a neural network with specified amount of dense layers with specified inputs and outputs.
        The input and hidden layers use the relu activation function and the output layer uses a linear activation
        function.

        :return:
        """
        n_nodes_hl1 = 24
        n_nodes_hl2 = 24

        # TODO: Maybe flatten to prevent bias from having 2 rows (2, 4) -> (1, 8)
        hidden_layer_1 = {
            'weights': tf.Variable(tf.random_normal([self.n_states[0], n_nodes_hl1])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
        }

        hidden_layer_2 = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
        }

        output_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl2, self.n_actions])),
            'biases': tf.Variable(tf.random_normal([self.n_actions]))
        }

        calculate_layer = lambda i, l: tf.add(tf.matmul(i, l['weights']), l['biases'])

        layer = tf.nn.relu(calculate_layer(self.input, hidden_layer_1))
        layer = tf.nn.relu(calculate_layer(layer, hidden_layer_2))
        output = calculate_layer(layer, output_layer)

        q_vals = tf.reduce_sum(tf.matmul(output, self.targets), 1)

        loss = tf.losses.mean_squared_error(output, self.targets)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return output, loss, optimizer

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
            action_values = sess.run(self.model, feed_dict={self.input: a_state})

            return np.argmax(action_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for mb_state, mb_action, mb_reward, mb_state_new, mb_done in minibatch:
            target = mb_reward

            if not mb_done:
                # print(sess.run(self.model, feed_dict={self.input: mb_state_new}))
                # Bellman's Function
                target = mb_reward + self.gamma * np.argmax(sess.run(self.model, feed_dict={self.input: mb_state_new}))
                # print(target)
            q = sess.run(self.model, feed_dict={self.input: mb_state})
            q[0][mb_action] = target

            _, loss = sess.run([self.loss, self.optimizer], feed_dict={self.input: mb_state, self.targets: q})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.995

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = DQNAgent()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(3000):
            # The initial observed state of the environment
            observation = env.reset()
            # state = np.expand_dims(observation, axis=0)
            state = np.reshape(observation, [1, 4])

            for t in range(500):
                # env.render()

                action = agent.act(state)

                observation_new, reward, done, _ = env.step(action)
                state_new = np.reshape(observation_new, [1, 4])

                agent.remember(state, action, reward, state_new, done)

                state = state_new

                if done:
                    print('Episode {}, reward: {}'.format(e, t))
                    break

            agent.replay(np.amin((len(agent.memory), 50)))  # TODO: Increase

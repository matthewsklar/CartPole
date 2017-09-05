import gym
import random

import tensorflow as tf
import numpy as np

from collections import deque


class Model:
    def create_model(self, n_states, n_actions, input, targets, learning_rate):
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
            'weights': tf.Variable(tf.random_normal([n_states[0], n_nodes_hl1])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
        }

        hidden_layer_2 = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
        }

        output_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_actions])),
            'biases': tf.Variable(tf.random_normal([n_actions]))
        }

        calculate_layer = lambda i, l: tf.add(tf.matmul(i, l['weights']), l['biases'])

        layer = tf.nn.relu(calculate_layer(input, hidden_layer_1))
        layer = tf.nn.relu(calculate_layer(layer, hidden_layer_2))
        output = calculate_layer(layer, output_layer)

        q_vals = tf.reduce_sum(tf.matmul(output, targets), 1)

        loss = tf.losses.mean_squared_error(output, targets)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return output, loss, optimizer


class DQNAgent:
    """Networks intelligent agent.

    Observes the environment and acts on it with the intention of achieving a goal.

    Attributes:
        n_states: A tuple holding data about the amount of states in the environment.
        n_actions: An integer representing the amount of actions the agent can apply to the environment.
        memory: A collections.deque object that stores in memory (state, action, reward, new_state, done).
        learning_rate: A float representing the learning rate (how much new information overrides old information).
        gamma: A float representing the discount factor (the importance of estimated future rewards).
        epsilon: A float with the starting chance of the agent picking a random action (exploration over exploitation).
        epsilon_min: A float with the minimum epsilon value.
        epsilon_decay: A float with the amount that the epsilon value decreases every iteration if above epsilon_min.
        input: A tensor holding a placeholder for the networks input.
        targets: A tensor holding a placeholder for the networks expected output.
        neural_network: A tuple holding information about the neural network (model, loss, optimizer).
        model: A tensor holding the model of the neural network.
        loss: A tensor holding the loss of the output vs expected output.
        optimizer: A TensorFlow operation that trains the model.
    """

    def __init__(self, learning_rate=0.001, gamma=0.95):
        """Initialize DQNAgent.

        Args:
            learning_rate: A float representing the learning rate for the network.
            gamma: A float representing the discount factor for Bellman's Function.
        """
        self.n_states = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.input = tf.placeholder(tf.float32, [None, self.n_states[0]])
        self.targets = tf.placeholder(tf.float32, [None, self.n_actions])
        self.neural_network = Model().create_model(self.n_states, self.n_actions, self.input, self.targets,
                                                   self.learning_rate)
        self.model = self.neural_network[0]
        self.loss = self.neural_network[1]
        self.optimizer = self.neural_network[2]

        print('State Shape: %d\tAction Shape: %d' % (self.n_states[0], self.n_actions))

    def remember(self, r_state, r_action, r_reward, r_new_state, r_done):
        """Stores the memory.

        Memory contains (state, action, reward, new_state, done).

        Args:
            r_state: A numpy array representing the original state of the environment.
            r_action: A numpy int64 representing the action the agent sends to the environment.
            r_reward: A float representing the reward the agent receives from the action.
            r_new_state: A numpy array representing the state of the environment after the action.
            r_done: A boolean representing if the environment finished a game after the action.
        """
        self.memory.append((r_state, r_action, r_reward, r_new_state, r_done))

    def act(self, a_state):
        """Selects the next actions.

        Determine which action the agent should send to the environment based on the state. There is an epsilon chance
        that it explores by selecting a random action.

        Args:
            a_state: A numpy array representing the state of the environment.

        Returns:
            An integer of either 0 or 1.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            action_values = sess.run(self.model, feed_dict={self.input: a_state})

            return np.argmax(action_values)

    def replay(self, batch_size):
        """Train the network in batches and update epsilon.

        Replay the network using memory to determine the expected output using the Bellman Equation taking into account
        future expected rewards. Train the network using by running the loss function in the optimizer and letting
        TensorFlow do it's magic. Train the network the specified amount of times. After, reduce the epsilon value if it
        is greater than the minimum epsilon value to reduce exploration and increase exploitation.

        Args:
            batch_size: The number of samples to get from memory and propagate through the network to train it.
        """
        minibatch = random.sample(self.memory, batch_size)

        for mb_state, mb_action, mb_reward, mb_state_new, mb_done in minibatch:
            target = mb_reward

            if not mb_done:
                # Bellman's Function
                target = mb_reward + self.gamma * np.amax(sess.run(self.model, feed_dict={self.input: mb_state_new}))

            q = sess.run(self.model, feed_dict={self.input: mb_state})
            q[0][mb_action] = target

            _, loss = sess.run([self.loss, self.optimizer], feed_dict={self.input: mb_state, self.targets: q})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = DQNAgent()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(500):
            # The initial observed state of the environment
            state = env.reset()
            state = np.expand_dims(state, axis=0)

            for t in range(500):
                # env.render()

                action = agent.act(state)

                state_new, reward, done, _ = env.step(action)
                state_new = np.expand_dims(state_new, axis=0)

                agent.remember(state, action, reward, state_new, done)

                state = state_new

                if done:
                    print('Episode {}, reward: {}, epsilon {}'.format(e, t, agent.epsilon))
                    break

            agent.replay(np.amin((len(agent.memory), 50)))  # TODO: Increase

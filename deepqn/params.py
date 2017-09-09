# The simulation the environment should run
ENVIRONMENT = 'CartPole-v0'

# The amount of times to run the simulation
EPISODES = 1000

# The maximum amount of time each instance of the environment can run
STEPS = 200

# The maximum amount of steps of data to store
MEMORY_CAPACITY = 2000

# A float storing the chance of the agent picking a random action (exploration over exploitation).
EPSILON = 1.0

# A float containing the minimum epsilon value.
EPSILON_MIN = 0.01

# A float with the amount that epsilon decreases every iteration if above minimum epsilon.
EPSILON_DECAY = 0.98

# A float representing the discount factor (the importance of estimated future rewards).
GAMMA = 0.999

# The batch size for training the network
BATCH_SIZE = 100

# A float representing the learning rate (how much new information overrides old information).
LEARNING_RATE = 0.001

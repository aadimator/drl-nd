
# Deep Reinforcement Learning Nanodegree
## P2 - Continuous Control Report
This report outlines my implementation for Udacity's Deep Reinforcement Learning Nanodegree's second project on the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 

![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Implementation
The algorithm used is **Deep Deterministic Policy Gradients (DDPG)**.

The final hyper-parameters used were as follows (n_episodes=1000, max_t=1000).
```python
SEED = 13                    # random seed for python, numpy & torch
episodes = 1000             # max episodes to run
max_t = 1000                # max steps in episode
solved_threshold = 30       # finish training when avg. score in 100 episodes crosses this threshold

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM   = 10        # number of learning passes
GRAD_CLIPPING = 1.0     # Gradient Clipping

# Ornstein-Uhlenbeck noise parameters
OU_SIGMA  = 0.1
OU_THETA  = 0.15
EPSILON       = 1.0     # for epsilon in the noise process (act step)
EPSILON_DECAY = 1e-6

```

#### Deep Deterministic Policy Gradient (DDPG)
This algorithm is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning for continuous tasks.

#### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

You can find the actor-critic logic implemented in the file **`ddpg_agent.py`**. The actor-critic models can be found via their respective **`Actor()`** and **`Critic()`** classes in **`models.py`**.

In the algorithm, local and target networks are implemented separately for both the actor and the critic.

```python
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

#### Exploration vs Exploitation
One challenge is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the rewards observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the **exploration-exploitation dilemma**.

For this project, we'll use the **Ornstein-Uhlenbeck process**, as suggested in the previously mentioned [paper by Google DeepMind](https://arxiv.org/pdf/1509.02971.pdf) (see bottom of page 4). The Ornstein-Uhlenbeck process adds a certain amount of noise to the action values at each timestep. This noise is correlated to previous noise, and therefore tends to stay in the same direction for longer durations without canceling itself out. This allows the arm to maintain velocity and explore the action space with more continuity.

You can find the Ornstein-Uhlenbeck process implemented in the **`OUNoise`** class in **`ddpg_agent.py`**.

In total, there are five hyperparameters related to this noise process.

The Ornstein-Uhlenbeck process itself has three hyperparameters that determine the noise characteristics and magnitude:
- mu: the long-running mean
- theta: the speed of mean reversion
- sigma: the volatility parameter

The final noise parameters were set as follows:

```python
OU_SIGMA = 0.1          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process
```

#### Experience Replay
Experience replay allows the RL agent to learn from past experience.

DDPG also utilizes a replay buffer to gather experiences from each agent. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. Each agent samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agents have multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

#### Neural Network
As implemented in the file **`model.py`**, both **Actor** and **Critic** (and local & target for each) consist of three (3) fully-connected (**Linear**) layers. The **input to fc1 is state_size**, while the **output of fc3 is action_size**. There are **400 and 300 hidden units** in fc1 and fc2, respectively, and **batch normalization (BatchNorm1d) **is applied to fc1. **ReLU activation is applied to fc1 and fc2**, while **tanh is applied to fc3**.

##### &nbsp;

**NOTE**: The files **`ddpg_agent.py`** and **`model.py`** were taken *almost verbatim* from the **Deep Deterministic Policy Gradients (DDPG)** Coding Exercise in **3. Policy-Based Methods, Lesson 5. Actor-Critic Methods.** Specificially, from **DDPG.ipynb** running the **'Pendulum-v0'** gym environment.

##### &nbsp;

## Plot of Rewards

The best result (DDPG) was an agent being able to solve the environment in ***296 episodes!***.

![Best Agent](https://i.imgur.com/ZWb77a1.png)

##### &nbsp;

## Ideas for Future Work
1. Do **hyper-parameter tuning** on the current DDPG model.
2. Try **Trust Region Policy Optimization (TRPO)** and **Truncated Natural Policy Gradient (TNPG)** as these two algorithms have been shown to achieve better performance.
3. Try the (very!) recent **Distributed Distributional Deterministic Policy Gradients (D4PG)** algorithm as another method for adapting DDPG for continuous control. 
4. Try the **(Optional) Challenge: Crawl**.


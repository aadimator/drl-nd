import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 0.005
        self.alpha = 0.1
        self.gamma = 1.0
        
    def epsilon_greedy_probs(self, Q_s):
#         self.eps =  max(self.eps * 0.99999, 0.0005)
        policy_s = np.ones(self.nA) * self.eps / self.nA
        policy_s[np.argmax(Q_s)] = 1 - self.eps + (self.eps / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(np.arange(self.nA), p=self.epsilon_greedy_probs(self.Q[state]))

    
    def update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])
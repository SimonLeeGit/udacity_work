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
        self.epsilon = 0.005
        self.gamma = 1.0
        self.alpha = 0.3

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.epsilon_greedy_policy(self.Q[state], self.nA, self.epsilon)
        return np.random.choice(np.arange(self.nA), p=policy_s)

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
        policy_s = self.epsilon_greedy_policy(self.Q[next_state], self.nA, self.epsilon)
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.dot(self.Q[next_state], policy_s) - self.Q[state][action])
        
    def epsilon_greedy_policy(self, Q_s, nA, epsilon=1.0):
        policy_s = np.ones(nA) * epsilon / nA
        a = np.argmax(Q_s)
        policy_s[a] = 1 - epsilon + (epsilon / nA)
        return policy_s
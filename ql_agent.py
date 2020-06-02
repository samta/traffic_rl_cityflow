from agent import Agent

from epsilon_greedy import EpsilonGreedy


class QLAgent(Agent):

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        super(QLAgent, self).__init__(state_space, action_space)
        self.state = starting_state
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        print ('self.state:', self.state)
        self.q_table = {self.state: [0 for _ in range(action_space)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self, state):
        print('self.q_table:', self.q_table)
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        print('self.action:', self.action)
        return self.action

    def learn(self, new_state, reward, done=False):
        if new_state not in self.q_table:
            self.q_table[new_state] = [0 for _ in range(self.action_space)]

        s = self.state
        s1 = new_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward

import abc


class Agent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    @abc.abstractmethod
    def new_episode(self):
        pass

    @abc.abstractmethod
    def observe(self, observation):
        ''' To override '''
        pass

    @abc.abstractmethod
    def act(self):
        ''' To override '''
        pass

    @abc.abstractmethod
    def learn(self, action, reward, done):
        ''' To override '''
        pass

from random import randint

class RoverLearner:
    def __init__(self, state_size, action_size, state):
        self.state_size = state_size
        self.action_size = action_size
        self.state = state
        self.action = 0

    def setState(self, state, reward, terminal):
        self.state = state
        self.action = randint(0, self.action_size - 1)
        if reward == 1:
            print "REWARD!!!"

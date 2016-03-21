from RoverLearning import RoverLearner
from ValueFunction import threeColorValueFunction as valueFunction
import numpy as np
import scipy.io as sio
from time import sleep
import cv2

import theano
from deep_q_rl import DeepQLearner

class roverSim():
    def __init__(self):
        self.quit = False

        self.input_width = 3
        self.input_height = 1
        self.n_actions = 2
        self.discount = 0.9
        self.learn_rate = 0.001
        self.batch_size = 100
        self.rng = np.random
        self.replay_size = 1000
        self.max_iter = 10000
        self.epsilon = 0.2

        self.state = [0, 0, 0]
        self.prev_state = []
        self.actions = ["left", "right"]

        self.leftTransitions = {
            "000":"001",
            "001":"002",
            "002":"102",
            "102":"103",
            "103":"203",
            "203":"200",
            "200":"210",
            "210":"310",
            "310":"320",
            "320":"020",
            "020":"030",
            "030":"000"
        }

        self.rightTransitions = {
            "001":"000",
            "002":"001",
            "102":"002",
            "103":"102",
            "203":"103",
            "200":"203",
            "210":"200",
            "310":"210",
            "320":"310",
            "020":"320",
            "030":"020",
            "000":"030"
        }

        self.D = (
            np.zeros((self.replay_size, 1, self.input_height, self.input_width), dtype=theano.config.floatX),
            np.zeros((self.replay_size, 1), dtype='int32'),
            np.zeros((self.replay_size, 1), dtype=theano.config.floatX),
            np.zeros((self.replay_size, 1, self.input_height, self.input_width), dtype=theano.config.floatX),
            np.zeros((self.replay_size, 1), dtype='int32')
        )

        for step in range(self.replay_size):
            sequence = self.getAction()
            for entry in range(len(self.D)):
                self.D[entry][step] = sequence[entry]
            if sequence[4] == 0:
                self.state = sequence[3]
            elif sequence[4] == 1:
                self.state = [0, 0, 0]

        print "hi"
        self.agent = DeepQLearner(
            self.input_width,
            self.input_height,
            self.n_actions,
            self.discount,
            self.learn_rate,
            self.batch_size,
            self.rng
        )

        print "hi"

        self.terminate()

        self.display = np.zeros((400, 400, 3), np.uint8)

        self.terminal = False
        self.run()

    def serialize(self, state):
        string = ""
        for x in state:
            string += str(x)
        return string

    def deserialize(self, string):
        state = []
        for x in string:
            state.append(int(x))
        return state

    def run(self):
        for i in xrange(10000):
            print i
            self.refreshVideo()

        terminal = False
        steps = 0
        while not terminal:
            steps += 1
            action = self.agent.choose_action(self.state, 0)
            self.takeAction(action)
            reward, terminal = valueFunction(self.state)
        print steps
        

    def refreshVideo(self):
        action = self.agent.choose_action(self.state, self.epsilon)
        self.updateDisplay()
        self.prev_state = self.state
        if self.terminal:
            self.terminate()
        else:
            self.takeAction(action)
        if not cmp(self.state, self.prev_state) == 0:
            reward, terminal = valueFunction(self.state)
            sequence = [self.prev_state, action, reward, self.state, terminal]
            for entry in range(len(self.D)):
                np.delete(self.D[entry], 0, 0)
                np.append(self.D[entry], sequence[entry])
            batch_index = np.random.permutation(self.batch_size)
            loss = self.agent.train(self.D[0][batch_index], self.D[1][batch_index], self.D[2][batch_index], self.D[3][batch_index], self.D[4][batch_index])
            self.terminal = terminal

    def takeAction(self, action):
        action = self.actions[action]
        if action == "left":
            self.state = self.deserialize(self.leftTransitions[self.serialize(self.state)])
        elif action == "right":
            self.state = self.deserialize(self.rightTransitions[self.serialize(self.state)])
        print self.state

    def getAction(self):
        action = np.random.randint(2)
        state_prime = None
        if action == 0:
            state_prime = self.deserialize(self.leftTransitions[self.serialize(self.state)])
        else:
            state_prime = self.deserialize(self.rightTransitions[self.serialize(self.state)])
        reward, terminal = valueFunction(state_prime)
        return [self.state, action, reward, state_prime, terminal]

    def terminate(self):
        self.state = [0, 0, 0]
        print "TERMINATED"

    def updateDisplay(self):
        self.display[:,:] = (0,0,0)
        self._updateDisplay(self.state[0], (203, 192, 255)) #pink
        self._updateDisplay(self.state[1], (0, 165, 255)) #orange
        self._updateDisplay(self.state[2], (0, 255, 0)) #green

        #cv2.imshow("window", self.display)
        #cv2.waitKey(1)

    def _updateDisplay(self, idx, color):
        if idx == 1:
            self.display[:,0:0.333*400] = color
        elif idx == 2:
            self.display[:,0.333*400:0.666*400] = color
        elif idx == 3:
            self.display[:,0.666*400:400] = color


if __name__ == "__main__":
    roverSim()

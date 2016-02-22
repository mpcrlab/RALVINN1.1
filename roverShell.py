########################################################
# ------------------------------------------------------#
#
# Machine Perception and Cognitive Robotics Laboratory
# Center for Complex Systems and Brain Sciences
#           Florida Atlantic University
#
# ------------------------------------------------------#
########################################################
# ------------------------------------------------------#
#
# Distributed ALVINN, See:
# Pomerleau, Dean A. Alvinn:
# An autonomous land vehicle in a neural network.
# No. AIP-77. Carnegie-Mellon Univ Pittsburgh Pa
# Artificial Intelligence And Psychology Project, 1989.
#
# ------------------------------------------------------#
########################################################

import pygame
from pygame.locals import *
from time import sleep
from datetime import date
from random import choice
from string import ascii_lowercase, ascii_uppercase
import threading
import cStringIO
import numpy as np
from scipy.misc import imresize
from scipy import ndimage as ndi
from af import *

from rover import Rover20


class roverShell(Rover20):
    def __init__(self):
        Rover20.__init__(self)
        self.quit = False
        self.lock = threading.Lock()

        self.treads = [0, 0]
        self.nn_treads = [0, 0]
        self.currentImage = None
        self.peripherals = {'lights': False, 'stealth': False,
                            'detect': True, 'camera': 0}

        self.action_choice = 1
        self.action_labels = ['forward', 'backward', 'left', 'right']
        self.action_vectors_motor = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
        self.action_vectors_neuro = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        self.n1 = 32 * 24 * 3
        # Number of neurons on the network
        self.number_of_neurons = 5
        # Number of actions available, like forward, back , left and right.
        self.number_of_actions = 4

        self.network_weight_one = 0.0001 * np.random.random((self.n1 + 1, self.number_of_neurons))
        self.network_weight_two = 0.01 * np.random.random((self.number_of_neurons + 1, self.number_of_actions))

        self.dw1 = np.zeros(self.network_weight_one.shape)
        self.dw2 = np.zeros(self.network_weight_two.shape)

        # learning rate of network
        self.network_learning_rate_one = 0.001
        # learning rate of network
        self.network_learning_rate_two = 0.01
        # Network Momemtum Value
        self.M = .5

    # main loop
    def processVideo(self, jpegbytes, timestamp_10msec):
        self.lock.acquire()
        if self.peripherals['detect']:
            self.processImage(jpegbytes)
            self.currentImage = jpegbytes
        else:
            self.currentImage = jpegbytes
        self.lock.release()
        self.setTreads(self.treads[0], self.treads[1])
        self.setperipherals()
        if self.quit:
            self.close()

    # openCV operations
    def processImage(self, jpegbytes):
        pass
    # camera features
    def setperipherals(self):
        if self.peripherals['lights']:
            self.turnLightsOn()
        else:
            self.turnLightsOff()

        if self.peripherals['stealth']:
            self.turnStealthOn()
        else:
            self.turnStealthOff()

        if self.peripherals['camera'] in (-1, 0, 1):
            self.moveCameraVertical(self.peripherals['camera'])
        else:
            self.peripherals['camera'] = 0

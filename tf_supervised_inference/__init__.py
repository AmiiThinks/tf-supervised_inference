from .distributions import *
from .data import *


class LinearModel(object):
    def __init__(self, weights):
        self.weights = [weights]

    def predict(self, phi):
        return phi @ self.weights[0]

class LinearModel(object):
    def __init__(self, weights):
        self.weights = [weights]

    def __call__(self, phi):
        return phi @ self.weights[0]

    def predict(self, phi):
        return self(phi)

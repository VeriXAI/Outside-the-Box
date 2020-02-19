import numpy as np


class DummyLayer(object):
    def __init__(self, n):
        self.output_shape = [0, n]


class ManualModel(object):
    def __init__(self, weight_matrizes, activations):
        self.weight_matrizes = weight_matrizes
        self.activations = activations
        self.layers = ManualModel.get_layers(weight_matrizes)  # needed for the interface

    def predict(self, inputs, layer=None):
        last_layer = len(self.layers) - 1 if layer is None else layer
        outputs = []
        for input in inputs:
            v = input
            for l in range(last_layer):
                v = self.weight_matrizes[l].dot(v)
                v = np.squeeze(np.asarray(v))   # convert matrix to vector
                activation = self.activations[l]
                if activation == "relu":
                    for j, vj in enumerate(v):
                        if vj < 0:
                            v[j] = 0.0
                elif activation == "":
                    pass
                else:
                    raise(ValueError("Unknown activation function {} detected.".format(activation)))
            outputs.append(v)  # TODO convert to categorical?
        return outputs

    def fit(self, x, y, epochs, batch_size):  # needed for the interface
        pass

    def save(self, model_path):  # needed for the interface
        pass

    def is_manual_model(self):  # needed for the interface
        return True

    @staticmethod
    def get_layers(weight_matrices):
        layers = [DummyLayer(weight_matrices[0].shape[1])]
        layers.extend([DummyLayer(weights.shape[0]) for weights in weight_matrices])
        return layers

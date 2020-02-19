from .Trainer import Trainer
from utils import *


class StandardTrainer(Trainer):
    def __init__(self):
        pass

    def __str__(self):
        return "StandardTrainer"

    def train(self, model, data_train: DataSpec, data_test: DataSpec, epochs: int, batch_size: int):
        history = model.fit(data_train.x(), np.array(data_train.ground_truths()), epochs=epochs, batch_size=batch_size)

        if VERBOSE_MODEL_TRAINING:
            print("score:", model.evaluate(data_test.x(), data_test.y(), batch_size=batch_size))
            print(model.summary())

        return history

        # input_prediction = x_test_final[0:1, :]
        # prediction = model.predict(input_prediction, 1, 1)
        # predicted_class = np.argmax(prediction)
        # print("testing prediction:\n input: ", input_prediction,
        #       "\n output: ", prediction,
        #       "\n class (in [0,", n_classes, "]): ", predicted_class)

        # Testing
        # layer_outs = get_watched_values(model, x_train[0:1, :])
        # print(layer_outs)

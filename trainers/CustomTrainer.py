from tensorflow_core.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.keras.metrics import Mean, SparseCategoricalAccuracy
import tensorflow_core as tf

from .Trainer import Trainer
from utils import *


class CustomTrainer(Trainer):
    """
    This trainer follows https://www.tensorflow.org/beta/tutorials/quickstart/advanced .
    """
    def __init__(self):
        # loss function for training
        self.loss_object = SparseCategoricalCrossentropy()
        # optimizer for training
        self.optimizer = Adam()
        # metrics to measure the loss and the accuracy of the model
        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = Mean(name='test_loss')
        self.test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')

    def __str__(self):
        return "CustomTrainer"

    @tf.function
    def train_step(self, inputs, classes, model):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = self.loss_object(classes, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(classes, predictions)

    @tf.function
    def test_step(self, inputs, classes, model):
        predictions = model(inputs)
        loss = self.loss_object(classes, predictions)

        self.test_loss(loss)
        self.test_accuracy(classes, predictions)

    def print_summary(self, epoch):
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              self.train_loss.result(),
                              self.train_accuracy.result() * 100,
                              self.test_loss.result(),
                              self.test_accuracy.result() * 100))

    def train_epoch(self, model, train_ds, test_ds, epoch, data_train: DataSpec, data_test: DataSpec):
        for train_inputs, train_classes in train_ds:
            self.train_step(train_inputs, train_classes, model)

        for test_inputs, test_classes in test_ds:
            self.test_step(test_inputs, test_classes, model)

        if VERBOSE_MODEL_TRAINING:
            self.print_summary(epoch)

        # never terminate earlier
        return False

    def train(self, model, data_train: DataSpec, data_test: DataSpec, epochs, batch_size):
        x_train = data_train.x()
        y_train = data_train.y()
        x_test = data_test.x()
        y_test = data_test.y()

        y_train = to_classes(y_train)
        y_test = to_classes(y_test)

        train_ds = to_dataset(x_train, y_train, batch_size)
        test_ds = to_dataset(x_test, y_test, batch_size)

        for epoch in range(epochs):
            stop = self.train_epoch(model, train_ds, test_ds, epoch, data_train, data_test)
            if stop:
                break

        # TODO return the history
        return None

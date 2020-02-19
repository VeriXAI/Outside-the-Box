from os.path import isfile
from tensorflow_core.python.keras.models import load_model as tf_load_model
from time import time

from utils import *
from models.ModelLoader import get_model_loader


def get_model(model_name: str, data_train: DataSpec, data_test: DataSpec, n_classes, model_trainer, n_epochs,
              batch_size, statistics: Statistics, model_path):
    model_path, model_constructor = get_model_loader(model_name, model_path)

    model_path = "../" + model_path  # go up one folder because run scripts are started from the folder "run/"
    loaded = False
    if isfile(model_path):
        try:
            model, history = load_model(model_path)
            loaded = True
        except:
            pass

    if not loaded:
        print("Could not load model", model_path, "- creating and training new model")
        image_shape = get_image_shape(data_train.x())

        # construct raw model
        model = model_constructor(weights=None, classes=n_classes, input_shape=image_shape)

        # train model
        time_training_model = time()
        history = model_trainer.train(model, data_train, data_test, epochs=n_epochs, batch_size=batch_size)
        statistics.time_training_model = time() - time_training_model

        # store model
        store_model(model_path, model, history)

    return model, history


def load_model(model_path):
    model = tf_load_model(model_path)
    print("Loaded model from", model_path)
    history = None  # TODO store/load history?
    return model, history


def store_model(model_path, model, history):
    print("Storing model to", model_path)
    model.save(model_path)
    # TODO store/load history?

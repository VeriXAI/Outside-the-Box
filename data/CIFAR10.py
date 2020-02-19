import numpy as np
from os import listdir
import pickle
from pickle import load, dump
import tensorflow_core
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.applications.resnet import preprocess_input
from utils import DataSpec, load_data, filter_labels


def load_CIFAR_10(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
                  data_test_monitor: DataSpec, data_run: DataSpec):
    # raise(NotImplementedError("This method was abandoned. Please fix it first before using it."))

    cifar10_dataset_folder_path = "../data/cifar-10-python/cifar-10-batches-py"
    # preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
    n_batches = 5
    x_train = []
    y_train = []
    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        x_train.extend(features)
        y_train.extend(labels)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    data_train_model.set_data(x=x_train, y=y_train)
    data_train_monitor.set_data(x=x_train, y=y_train)
    with open(cifar10_dataset_folder_path + "/test_batch", mode='rb') as file:
        test = pickle.load(file, encoding='latin1')
    x_test = np.array(test['data'].reshape((len(test['data']), 3, 32, 32)).transpose(0, 2, 3, 1))
    y_test = np.array(test['labels'])
    data_test_model.set_data(x=x_test, y=y_test)
    data_test_monitor.set_data(x=x_test, y=y_test)
    data_run.set_data(x=x_test, y=y_test)
    pixel_depth = 255.0
    all_classes_network, all_classes_rest = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth)
    # labels
    labels_all = ['label' + str(i) for i in range(10)]

    labels_network = filter_labels(labels_all, all_classes_network)
    labels_rest = filter_labels(labels_all, all_classes_rest)

    return all_classes_network, labels_network, all_classes_rest, labels_rest


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + "/data_batch_" + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(normalize, one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + "/test_batch", mode='rb') as file:
        batch = load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         "preprocess_testing.p")


# extract features from each photo in the directory
def extract_features(data_path):
    model = network()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(data_path):
        filename = data_path + "/" + name
        image1 = name.split(".")
        image_id = image1[0]
        if not image1[1] == 'jpg':
            continue
        image = tensorflow_core.python.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        image = tensorflow_core.python.keras.preprocessing.image.img_to_array(image)  # this is (224, 224, 3)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])  # this is (1, 224, 224, 3)
        image = preprocess_input(image)
        feature = model.predict(image)
        features[image_id] = feature
        print(">%s" % name)
    return features

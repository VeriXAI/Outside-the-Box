from mpl_toolkits.mplot3d import Axes3D

from data import *
from run.experiment_helper import *


def run_explanation_alpha_thresholding():
    n_classes = 3
    model_name, data_name, stored_network_name, total_classes = instance_F_MNIST()
    total_classes = 4

    # load instance
    data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, _ = \
        load_instance(n_classes, total_classes, stored_network_name)
    get_data_loader(data_name)(data_train_model=data_train_model, data_test_model=data_test_model,
                               data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor,
                               data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model, n_classes=None,
                         model_trainer=None, n_epochs=None, batch_size=None, statistics=Statistics(),
                         model_path=model_path)

    # compute results
    probabilities = model.predict_proba(data_run.x())
    ground_truths = data_run.ground_truths()
    x = 0
    y = 1
    z = 2
    nov = 3
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []
    for p, gt in zip(probabilities, ground_truths):
        if gt == 0:
            x1.append(p[x])
            y1.append(p[y])
            z1.append(p[z])
        elif gt == 1:
            x2.append(p[x])
            y2.append(p[y])
            z2.append(p[z])
        elif gt == 2:
            x3.append(p[x])
            y3.append(p[y])
            z3.append(p[z])
        elif gt == 3:
            x4.append(p[x])
            y4.append(p[y])
            z4.append(p[z])
        else:
            assert False

    # create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, color=[0, 0, 1], marker="+", label="c{:}".format(x))
    ax.scatter(x2, y2, z2, color=[0, 0, 0], marker="+", label="c{:}".format(y))
    ax.scatter(x3, y3, z3, color=[0.5, 1, 0.5], marker="+", label="c{:}".format(z))
    ax.scatter(x4, y4, z4, color=[1, 0, 0], marker=".", alpha=0.1, label="c{:}".format(nov))
    ax.set_xlabel("x{:d}".format(x))
    ax.set_ylabel("x{:d}".format(y))
    ax.set_zlabel("x{:d}".format(z))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    # ax.legend()
    title = "Alpha threshold"
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)

    plt.show()


if __name__ == "__main__":
    run_explanation_alpha_thresholding()

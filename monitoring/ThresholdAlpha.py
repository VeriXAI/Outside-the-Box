from utils import *
from data import *
from trainers import *
from monitoring import *


def test_alpha(model, data, history, alpha):
    n = float(number_of_model_classes(model))
    b = n / (n - 1)
    a = -b
    confidences = model.predict_proba(data.x())
    predictions = [np.argmax(x) for x in confidences]
    history.set_ground_truths(data.ground_truths())
    history.set_predictions(predictions)
    results = []
    for c_prediction, confidence_vector in zip(predictions, confidences):
        result = MonitorResult()
        # map range [1/n, 1] (possible probabilities of chosen class) to range [100, 0] (confidence range)
        # we use a linear function ax + b with a, b defined above
        confidence = a * confidence_vector[c_prediction] + b
        result.add_confidence(confidence)
        results.append(result)
    history.set_monitor_results(0, results)
    history.update_statistics(0, confidence_threshold=alpha)


def run_script():
    # options
    alpha = .99
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data_name = "F_MNIST"
    data_train_model = DataSpec(classes=classes)
    data_test_model = DataSpec(classes=classes)
    data_train_monitor = DataSpec(classes=classes)
    data_test_monitor = DataSpec(classes=classes)
    data_run = DataSpec(randomize=False, classes=[x for x in range(0, 10)])
    model_name = "F_MNIST"
    model_path = "F_MNIST_MNIST_10-model.h5"
    n_epochs = 0
    batch_size = 128
    labels_network, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    model_trainer = StandardTrainer()
    statistics = Statistics()

    model, history_model = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                                     n_classes=len(labels_network), model_trainer=model_trainer, n_epochs=n_epochs,
                                     batch_size=batch_size, statistics=statistics, model_path=model_path)
    history = History()

    test_alpha(model, data_run, history, alpha)

    # plot statistics after timers
    tn = history.true_negatives()
    tp = history.true_positives()
    fn = history.false_negatives()
    fp = history.false_positives()
    fig, ax = initialize_single_plot("Performance of threshold test confidence >= {:f}".format(alpha))
    plot_pie_chart_single(ax=ax, tp=tp, tn=tn, fp=fp, fn=fn, n_run=data_run.n)
    plt.show()


if __name__ == "__main__":
    run_script()

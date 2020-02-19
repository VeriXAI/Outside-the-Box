from monitoring import *
from run.Runner import run
from utils import *
from data import *
from trainers import *


def evaluate_all(model_name, model_path, data_name, data_train_model, data_test_model, data_train_monitor,
                 data_test_monitor, data_run, monitor_manager: MonitorManager, alphas=None,
                 model_trainer=StandardTrainer(), seed=0, n_epochs=-1, batch_size=-1):
    # set random seed
    set_random_seed(seed)

    # construct statistics wrapper
    statistics = Statistics()

    # load data
    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    # load network model or create and train it
    model, history_model = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                                     n_classes=len(labels_network), model_trainer=model_trainer, n_epochs=n_epochs,
                                     batch_size=batch_size, statistics=statistics, model_path=model_path)

    print(("Data: classes {} with {:d} inputs (monitor training), classes {} with {:d} inputs (monitor test), " +
          "classes {} with {:d} inputs (monitor run)").format(
        classes2string(data_train_monitor.classes), data_train_monitor.n,
        classes2string(data_test_monitor.classes), data_test_monitor.n,
        classes2string(data_run.classes), data_run.n))

    # normalize and initialize monitors
    monitor_manager.normalize_and_initialize(model, len(labels_rest))

    # train monitors
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=statistics)

    # run monitors & collect novelties
    history_run = monitor_manager.run(model=model, data=data_run, statistics=statistics)
    novelty_wrapper_run = history_run.novelties(data_run, all_classes_network, all_classes_rest)

    if alphas is None:
        return history_run, novelty_wrapper_run, statistics

    # run alpha threshold
    histories_alpha_thresholding = []
    novelty_wrappers_alpha_thresholding = []
    for alpha in alphas:
        history_alpha_thresholding = History()
        test_alpha(model, data_run, history_alpha_thresholding, alpha)
        novelty_wrapper_alpha_thresholding =\
            history_alpha_thresholding.novelties(data_run, all_classes_network, all_classes_rest)
        histories_alpha_thresholding.append(history_alpha_thresholding)
        novelty_wrappers_alpha_thresholding.append(novelty_wrapper_alpha_thresholding)

    return history_run, histories_alpha_thresholding, novelty_wrapper_run, novelty_wrappers_alpha_thresholding,\
           statistics


def evaluate_combination(seed, data_name, data_train_model, data_test_model, data_train_monitor, data_test_monitor,
                         data_run, model_trainer, model_name, model_path, n_epochs, batch_size,
                         monitor_manager: MonitorManager, alpha, confidence_thresholds=None, skip_image_plotting=False):

    # convex-set monitoring
    model, history_abstraction, classes_network, labels_network, classes_rest, labels_rest, statistics = run(
        seed, data_name, data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run,
        model_trainer, model_name, model_path, n_epochs, batch_size, monitor_manager, confidence_thresholds,
        skip_image_plotting, show_statistics=False)

    # alpha-threshold monitoring
    history_alpha_thresholding = History()
    test_alpha(model, data_run, history_alpha_thresholding, alpha)

    # combinations
    history_combined = CombinedHistory([history_abstraction, history_alpha_thresholding])
    history_conditional_abs_at = ConditionalHistory([history_abstraction], [history_alpha_thresholding], alpha)
    history_conditional_at_abs = ConditionalHistory([history_alpha_thresholding], [history_abstraction], alpha)
    n_monitors = len(monitor_manager.monitors()) + 1

    # bin plots
    plot_false_decisions(monitors=monitor_manager.monitors(), history=history_abstraction,
                         confidence_thresholds=confidence_thresholds)
    plot_false_decisions(monitors=[0], history=history_alpha_thresholding, confidence_thresholds=confidence_thresholds,
                         name="alpha threshold")
    for i in range(1, n_monitors + 1):
        # plot_false_decisions(monitors=[0], history=history_combined, confidence_thresholds=confidence_thresholds,
        #                      n_min_acceptance=i)
        plot_false_decisions(monitors=[0], history=history_combined, confidence_thresholds=confidence_thresholds,
                             n_min_acceptance=-i)
    plot_false_decisions(monitors=[0], history=history_conditional_abs_at, confidence_thresholds=confidence_thresholds,
                         name="abstraction then alpha threshold")
    plot_false_decisions(monitors=[0], history=history_conditional_at_abs, confidence_thresholds=confidence_thresholds,
                         name="alpha threshold then abstraction")

    # pie plots
    # for monitor in monitor_manager.monitors():
    #     m_id = monitor.id()
    #     pie_plot(data_run, m_id, history_abstraction, alpha=confidence_thresholds[0])
    #     pie_plot(data_run, m_id, history_abstraction, alpha=confidence_thresholds[-1])
    # pie_plot(data_run, 0, history_alpha_thresholding, alpha=confidence_thresholds[0])
    # pie_plot(data_run, 0, history_alpha_thresholding, alpha=confidence_thresholds[-1])

    # novelty bin plots
    novelty_wrapper_abstraction = history_abstraction.novelties(data_run, classes_network, classes_rest)
    plot_novelty_detection(monitor_manager.monitors(), novelty_wrapper_abstraction, confidence_thresholds)
    novelty_wrapper_alpha_thresholding = history_alpha_thresholding.novelties(data_run, classes_network, classes_rest)
    plot_novelty_detection([0], novelty_wrapper_alpha_thresholding, confidence_thresholds, name="alpha threshold")
    novelty_wrapper_combined = history_combined.novelties(data_run, classes_network, classes_rest)
    for i in range(1, n_monitors + 1):
        # plot_novelty_detection([0], novelty_wrapper_combined, confidence_thresholds, n_min_acceptance=i)
        plot_novelty_detection([0], novelty_wrapper_combined, confidence_thresholds, n_min_acceptance=-i)
    novelty_wrapper_conditional1 = history_conditional_abs_at.novelties(data_run, classes_network, classes_rest)
    plot_novelty_detection([0], novelty_wrapper_conditional1, confidence_thresholds,
                           name="abstraction then alpha threshold")
    novelty_wrapper_conditional2 = history_conditional_at_abs.novelties(data_run, classes_network, classes_rest)
    plot_novelty_detection([0], novelty_wrapper_conditional2, confidence_thresholds,
                           name="alpha threshold then abstraction")

    # comparison plots
    confidence_threshold1_default = 0.0
    confidence_threshold2_default = alpha
    while True:
        answer = input("Show comparison plots [y, n]? ")
        if answer == "n":
            break
        confidence_threshold1 = input("confidence threshold 1 [empty string for default {:f}]? ".format(
            confidence_threshold1_default))
        confidence_threshold2 = input("confidence threshold 1 [empty string for default {:f}]? ".format(
            confidence_threshold2_default))
        if confidence_threshold1 == "":
            confidence_threshold1 = confidence_threshold1_default
        else:
            confidence_threshold1 = float(confidence_threshold1)
        if confidence_threshold2 == "":
            confidence_threshold2 = confidence_threshold2_default
        else:
            confidence_threshold2 = float(confidence_threshold2)
        plot_decisions_of_two_approaches(monitor_manager.monitors()[0], history_abstraction, confidence_threshold1,
                                         0, history_alpha_thresholding, confidence_threshold2, classes_network,
                                         classes_rest)

    # ROC curve
    for monitor in monitor_manager.monitors():
        ROC_plot(monitor.id(), history_abstraction)
    ROC_plot(0, history_alpha_thresholding, name="alpha threshold")
    for i in range(1, n_monitors + 1):
        # ROC_plot(0, history_combined, n_min_acceptance=i)
        ROC_plot(0, history_combined, n_min_acceptance=-i)
    ROC_plot(0, history_conditional_abs_at, name="abstraction then alpha threshold")
    ROC_plot(0, history_conditional_at_abs, name="alpha threshold then abstraction")

    # print("\nDone! In order to keep the plots alive, this program does not terminate until they are closed.")
    # plt.show()
    answer = input("Save all plots [y, n]? ")
    if answer == "y":
        save_all_figures()


def pie_plot(data_run, monitor_id, history_alpha_thresholding, alpha):
    history_alpha_thresholding.update_statistics(monitor_id, alpha)
    tn = history_alpha_thresholding.true_negatives()
    tp = history_alpha_thresholding.true_positives()
    fn = history_alpha_thresholding.false_negatives()
    fp = history_alpha_thresholding.false_positives()
    fig, ax = initialize_single_plot("Performance of monitor {:d} with confidence >= {:f}".format(monitor_id, alpha))
    plot_pie_chart_single(ax=ax, tp=tp, tn=tn, fp=fp, fn=fn, n_run=data_run.n)


def ROC_plot(monitor_id, history, n_min_acceptance=None, name=None):
    fp_list = []
    tp_list = []
    for alpha in range(0, 100, 1):
        history.update_statistics(monitor_id, float(alpha) / 100.0, n_min_acceptance=n_min_acceptance)
        fp_list.append(history.false_positive_rate())
        tp_list.append(history.true_positive_rate())

    fig, ax = plt.subplots(1, 1)
    if name is None:
        if n_min_acceptance is None:
            name = "{:d}".format(monitor_id)
        else:
            if n_min_acceptance >= 0:
                name = "acceptance {:d}".format(n_min_acceptance)
            else:
                name = "rejection {:d}".format(-n_min_acceptance)
    title = "ROC curve (monitor {})".format(name)
    fig.suptitle(title)
    fig.canvas.set_window_title(title)
    # plot ROC curve
    ax.cla()
    ax.scatter(fp_list, tp_list, marker='^', c="r")
    ax.plot([0, 1], [0, 1], label="baseline", c="k", linestyle=":")
    ax.set_title('ROC curve')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()
    plt.draw()
    plt.pause(0.0001)

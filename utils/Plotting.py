import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# import pandas as pd
# import seaborn as sns


from . import *


def initialize_subplots(n_subplots, title):
    n_cols = math.ceil(math.sqrt(n_subplots))
    n_rows = math.ceil(n_subplots / n_cols)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    row = 0
    col = -1
    fig.canvas.set_window_title(title)
    fig.suptitle(title)
    return fig, ax, n_cols, row, col


def initialize_single_plot(title):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    return fig, ax


def plot_histograms(monitor_manager, data_train_monitor: DataSpec, layer2all_trained_values):
    if layer2all_trained_values is None:
        return

    y = data_train_monitor.y()
    n_plots = 0
    for class_index, box_family in enumerate(monitor_manager.monitors[0].abstraction.abstractions):
        if n_plots >= N_HISTOGRAM_PLOTS_UPPER_BOUND:
            if n_plots == 0:
                print("Skipping histogram plots as requested in the options!".format(N_HISTOGRAM_PLOTS_UPPER_BOUND))
            else:
                print("Skipping the remaining histogram plots!".format(N_HISTOGRAM_PLOTS_UPPER_BOUND))
            break
        boxes = box_family.boxes
        for box in boxes:
            if box.isempty():
                continue
            if n_plots >= N_HISTOGRAM_PLOTS_UPPER_BOUND:
                break
            n_plots += 1
            for layer_index, all_trained_values in layer2all_trained_values.items():
                # plot 1D-histograms of trained values and compare them with the boxes
                fig, ax, n_cols, row, col = initialize_subplots(len(all_trained_values[0]), "Histograms")
                for dim in range(len(all_trained_values[0])):
                    if col < n_cols - 1:
                        col += 1
                    else:
                        row += 1
                        col = 0
                    dimension = []
                    for ind, trained_value in enumerate(all_trained_values):
                        if categorical2number(y[ind]) == class_index:
                            dimension.append(trained_value[dim])
                    ax[row][col].hist(dimension, color='steelblue',
                                      bins=int(np.sqrt(len(all_trained_values[0]))),
                                      edgecolor='black', linewidth=1)
                    ax[row][col].plot([box.low[dim], box.high[dim]], [1, 1], color='red', linewidth=5)
                plt.draw()
                plt.pause(0.0001)


def plot_model_history(history):
    if history is None:
        return

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    fig, ax, _, _, _ = initialize_subplots(2, "History")
    ax = ax[0]
    for l in loss_list:
        ax[0].plot(epochs, history.history[l],
                   label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        ax[0].plot(epochs, history.history[l], 'g',
                   label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Accuracy
    for l in acc_list:
        ax[1].plot(epochs, history.history[l],
                   label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        ax[1].plot(epochs, history.history[l], 'g',
                   label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.draw()
    plt.pause(0.0001)


def plot_images(images, labels, classes, iswarning: bool, monitor_id: int):
    if iswarning:
        n = len(images)
        if n > N_PRINT_WARNINGS:
            print("printing only the first {:d} out of {:d} warnings for monitor {:d}".format(
                N_PRINT_WARNINGS, n, monitor_id))
            n = N_PRINT_WARNINGS
        title_list = ["Warnings of monitor {:d}".format(monitor_id)]
        images_list = [images]
        n_list = [n]
    else:
        title_list = ["Novelties detected by monitor {:d}".format(monitor_id),
                      "Novelties not detected by monitor {:d}".format(monitor_id)]
        images_list = [images["detected"], images["undetected"]]
        n_list = []
        for images in images_list:
            n = len(images)
            if n > N_PRINT_NOVELTIES:
                print("printing only the first {:d} out of {:d} novelties for monitor {:d}".format(
                    N_PRINT_NOVELTIES, n, monitor_id))
                n = N_PRINT_NOVELTIES
            n_list.append(n)

    plotted_once = False
    for title, images, n in zip(title_list, images_list, n_list):
        if n == 0:
            continue
        plotted_once = True
        colors = get_rgb_colors(max(classes) + 1)
        fig, ax, n_cols, row, col = initialize_subplots(n, title)

        for i, image in enumerate(images):
            if iswarning and i >= N_PRINT_WARNINGS:
                break
            if not iswarning and i >= N_PRINT_NOVELTIES:
                break

            if col < n_cols - 1:
                col += 1
            else:
                row += 1
                col = 0
            ax[row][col].axis('off')
            normalized_image = np.clip(image.original_input, 0, 1)
            if len(normalized_image.shape) > 2 and normalized_image.shape[2] == 1:
                normalized_image = normalized_image.reshape((28, 28))
            ax[row][col].imshow(normalized_image)

            # add ground-truth class
            ax[row][col].scatter(-10, -5, color=colors[image.c_ground_truth])
            ax[row][col].annotate(labels[image.c_ground_truth] + " (GT)", (-8, -5))
            # add predicted class
            ax[row][col].scatter(-10, -10, color=colors[image.c_predicted])
            ax[row][col].annotate(labels[image.c_predicted] + " (NN)", (-8, -10))

    if plotted_once:
        plt.draw()
        plt.pause(0.0001)


def plot_monitor_training(monitor, history, iterations, scores, best_scores, fp_list, fn_list, tp_list,
                          class2inertias, score_name, category_title):
    ax = PLOT_MONITOR_TRAINING_AXIS()
    ax.cla()
    for layer in monitor.layers():
        plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=category_title, ax=ax)

    fig, ax = PLOT_MONITOR_RATES_AXIS()
    fig.canvas.set_window_title("Monitor-training history")

    # plot rates & score
    ax[0].cla()
    ax[0].scatter(iterations, fp_list, marker='^', c="r")
    ax[0].plot(iterations, fp_list, label="false positive rate", c="r", linestyle=":")
    ax[0].scatter(iterations, fn_list, marker='x', c="b")
    ax[0].plot(iterations, fn_list, label="false negative rate", c="b", linestyle="--")
    ax[0].plot(iterations, scores, label=score_name, c="g")
    ax[0].plot(iterations, best_scores, label="best score", c="orange")
    ax[0].set_title('False rates & score of the monitor')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Rates/Score')
    ax[0].legend()

    # adding another y axis does not work correctly
    # ax2 = ax[0].twinx()
    # ax2.cla()
    # ax2.plot(iterations, n_boxes_list, c="y", linestyle="--")
    # ax2.set_ylabel("# boxes", c="y")
    # ax2.legend()

    # plot ROC curve
    ax[1].cla()
    ax[1].scatter(fp_list, tp_list, marker='^', c="r")
    ax[1].plot([0, 1], [0, 1], label="baseline", c="k", linestyle=":")
    ax[1].set_title('ROC curve')
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].legend()

    # plot clustering inertias
    ax[2].cla()
    for class_index, inertias in class2inertias.items():
        ax[2].plot(iterations, inertias, label="clustering inertia class {}".format(class_index), linestyle=":")
    ax[2].set_title('Clustering inertias')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Inertia')
    ax[2].legend()

    plt.draw()
    plt.pause(0.0001)


def plot_2d_projection(history, monitor, layer, category_title, ax=None, known_classes=None, novelty_marker="$N$",
                       dimensions=None):
    if ax is None:
        ax = plt.figure().add_subplot()
    m_id = 0 if monitor is None else monitor.id()
    title = "Projected data & abstractions ({}) (monitor {:d}, layer {:d})".format(category_title, m_id, layer)
    ax.figure.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    if dimensions is None:
        dimensions = monitor.dimensions(layer)
    x = dimensions[0]
    y = dimensions[1]
    ax.set_xlabel("x{:d}".format(x), size=16)
    ax.set_ylabel("x{:d}".format(y), size=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    # create mapping 'class -> values'
    class2values = dict()
    for cj, vj in zip(history.ground_truths, history.layer2values[layer]):
        if cj in class2values.keys():
            xs, ys = class2values[cj]
        else:
            xs = []
            ys = []
            class2values[cj] = (xs, ys)
        xs.append(vj[x])
        ys.append(vj[y])

    if known_classes is None:
        n_classes = max(class2values.keys()) + 1
    else:
        n_classes = len(known_classes)
    colors = get_rgb_colors(n_classes)
    markers = get_markers(n_classes)

    # plot abstraction
    if monitor is not None:
        for i, ai in enumerate(monitor.abstraction(layer).abstractions()):
            if ai.isempty():
                continue
            ai.plot(dims=dimensions, ax=ax, color=colors[i])

    # scatter plot
    novelties = []
    for cj, (xs, ys) in class2values.items():
        if known_classes is None or cj in known_classes:
            color = [colors[cj]]
            marker = markers[cj]
        else:
            novelties.append((xs, ys))
            continue
        ax.scatter(xs, ys, alpha=0.5, label="c" + str(cj), c=color, marker=marker)
    # plot novelties last
    for xs, ys in novelties:
        ax.scatter(xs, ys, alpha=1.0, label="novelty", c=["k"], marker=novelty_marker, zorder=3)

    # ax.legend()
    plt.draw()
    plt.pause(0.0001)


def plot_zero_point(ax, color, epsilon, epsilon_relative):
    if epsilon > 0 and not epsilon_relative:
        print("Epsilon with zero filtering is ignored in plotting.")
    ax.scatter([0], [0], alpha=1.0, c=[color], marker="$+$")
    plt.draw()
    plt.pause(0.0001)


def plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim):
    if epsilon > 0:
        print("Epsilon with zero filtering is ignored in plotting.")
    if is_x_dim:
        points = [[p1, 0], [p2, 0]]
    else:
        points = [[0, p1], [0, p2]]
    polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none")
    ax.add_patch(polygon)


def plot_pie_chart_single(ax, tp, tn, fp, fn, n_run):
    # pie chart, where the slices will be ordered and plotted counter-clockwise
    if n_run >= 0:
        sizes = [ratio(tn, n_run), ratio(tp, n_run), ratio(fp, n_run), ratio(fn, n_run)]
    else:
        sizes = [1, 1, 1, 1]

    colors = ["w", "w", "w", "w"]
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=0, colors=colors,
                                      wedgeprops={"edgecolor": [0, .4, .6]},
                                      pctdistance=1.2, labeldistance=1.5)
    plt.setp(autotexts, size=16)
    patterns = [".", "o", "*", "O"]
    for i in range(len(wedges)):
        wedges[i].set_hatch(patterns[i % len(patterns)])
        if i in [2, 3]:
            wedges[i].set_ec([1, 0, 0])

    """nicer labels but they don't work well"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(sizes[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    """

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if n_run < 0:
        # plot only the legend
        ax.cla()
        plt.axis('off')
        labels = 'true negatives', 'true positives', 'false positives', 'false negatives'
        ax.legend(wedges, labels, loc="center", handleheight=3)


def _get_binary_pie(t):
    n = t[0] + t[1]
    return [ratio(t[0], n), ratio(t[1], n)]


def plot_novelty_detection(monitors, novelty_wrapper, confidence_thresholds, n_min_acceptance=None, name=None):
    x, xticks = get_xticks_bars(confidence_thresholds)

    if name is None and n_min_acceptance is not None:
        if n_min_acceptance >= 0:
            name = "acceptance {:d}".format(n_min_acceptance)
        else:
            name = "rejection {:d}".format(-n_min_acceptance)
    for monitor in monitors:
        m_id = monitor if isinstance(monitor, int) else monitor.id()
        y = []
        for confidence_threshold in confidence_thresholds:
            novelties = novelty_wrapper.evaluate_detection(m_id, confidence_threshold,
                                                           n_min_acceptance=n_min_acceptance)
            n = len(novelties["detected"])
            d = n + len(novelties["undetected"])
            y.append(ratio(n, d))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(x, y, color=[0, .4, .6], edgecolor='white', width=0.5)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Novelties detected [%]")
        ax.set_ylim([0, 100])
        ax.xaxis.set_ticks(xticks)
        if name is None:
            final_name = "{:d}".format(m_id)
        else:
            final_name = name
        title = "Novelty detection (monitor {})".format(final_name)
        fig.suptitle(title)
        ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


def plot_novelty_detection_given_all_lists(core_statistics_list_of_lists: list, n_ticks, name=""):
    n_monitors = len(core_statistics_list_of_lists)
    n_bars = len(core_statistics_list_of_lists[0])
    for core_statistics_list in core_statistics_list_of_lists:
        assert len(core_statistics_list) == n_bars, "Incompatible list lengths found!"
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, len(core_statistics_list_of_lists[0]) + 2)], n=n_ticks, to_float=False)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 1.0 / float(n_monitors + 1)
    for b in range(n_bars):
        for i, core_statistics_list in enumerate(core_statistics_list_of_lists):
            cs = core_statistics_list[b]
            d = cs.novelties_detected + cs.novelties_undetected
            nd = ratio(cs.novelties_detected, d)
            nu = ratio(cs.novelties_undetected, d)
            x_adapted = x[b] + i * width
            ax.bar(x_adapted, nd, color=[0, .4, .6], edgecolor='white', width=width)
            sums = nd
            ax.bar(x_adapted, nu, bottom=sums, color=[1, 0.6, 0.2], edgecolor='white', width=width)
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    title = "Novelty detection {}".format(name)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


# 'monitors' can either be a list of numbers or a list of Monitor objects
def plot_false_decisions(monitors, history: History, confidence_thresholds, n_min_acceptance=None, name=None,
                         title=None):
    d = len(history.ground_truths)
    x, xticks = get_xticks_bars(confidence_thresholds)

    if name is None and n_min_acceptance is not None:
        if n_min_acceptance >= 0:
            name = "acceptance {:d}".format(n_min_acceptance)
        else:
            name = "rejection {:d}".format(-n_min_acceptance)
    for monitor in monitors:
        m_id = monitor if isinstance(monitor, int) else monitor.id()
        if name is None:
            final_name = "{:d}".format(m_id)
        else:
            final_name = name
        y_fn = []
        y_fp = []
        y_tp = []
        y_tn = []
        for confidence_threshold in confidence_thresholds:
            history.update_statistics(m_id, confidence_threshold=confidence_threshold,
                                      n_min_acceptance=n_min_acceptance)
            fn = history.false_negatives()
            fp = history.false_positives()
            tp = history.true_positives()
            tn = history.true_negatives()
            y_fn.append(ratio(fn, d))
            y_fp.append(ratio(fp, d))
            y_tp.append(ratio(tp, d))
            y_tn.append(ratio(tn, d))

        _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, final_name, title=title)

    plt.draw()
    plt.pause(0.0001)


def _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, name, name2="", title=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.5
    blue = [0, .4, .6]
    yellow = [1, 0.65, 0.25]
    red = [1, 0, 0]
    ax.bar(x, y_tp, color=blue, edgecolor="white", width=width)
    sums = y_tp
    ax.bar(x, y_fn, bottom=sums, color=yellow, edgecolor="white", hatch="x", width=width)
    sums = [_x + _y for _x, _y in zip(sums, y_fn)]
    ax.bar(x, y_fp, bottom=sums, color=red, edgecolor='white', hatch=".", width=width)
    # sums = [_x + _y for _x, _y in zip(sums, y_tp)]
    # ax.bar(x, y_tn, bottom=sums, color=[0, 0.9, 0.1], edgecolor='white', width=width)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("True positives [blue] / false negatives [orange] / false positives [red]")
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    if title is None:
        title = "Decision performance (monitor {}) {}".format(name, name2)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)


def plot_false_decisions_given_list(core_statistics_list: list, n_ticks, name="", name2=""):
    d = core_statistics_list[0].get_n()
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, len(core_statistics_list) + 2)], n=n_ticks, to_float=False)
    y_fn = []
    y_fp = []
    y_tp = []
    y_tn = []
    for cs in core_statistics_list:
        y_fn.append(ratio(cs.fn, d))
        y_fp.append(ratio(cs.fp, d))
        y_tp.append(ratio(cs.tp, d))
        y_tn.append(ratio(cs.tn, d))

    _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, name=name, name2=name2)

    plt.draw()
    plt.pause(0.0001)


def plot_false_decisions_given_all_lists(core_statistics_list_of_lists: list, n_ticks, n_bars=None, name=""):
    n_monitors = len(core_statistics_list_of_lists)
    n_bars_reference = len(core_statistics_list_of_lists[0])
    if n_bars is None:
        n_bars = n_bars_reference

    for core_statistics_list in core_statistics_list_of_lists:
        assert len(core_statistics_list) == n_bars_reference, "Incompatible list lengths found!"
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, n_bars + 2)], n=n_ticks, to_float=False)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 1.0 / float(n_monitors + 1)
    blue = [0, .4, .6]
    yellow = [1, 0.65, 0.25]
    red = [1, 0, 0]
    for b in range(n_bars):
        for i, core_statistics_list in enumerate(core_statistics_list_of_lists):
            cs = core_statistics_list[b]
            d = cs.get_n()
            y_fn = ratio(cs.fn, d)
            y_fp = ratio(cs.fp, d)
            y_tp = ratio(cs.tp, d)
            y_tn = ratio(cs.tn, d)
            x_adapted = x[b] + i * width
            ax.bar(x_adapted, y_tp, color=blue, edgecolor="white", width=width)
            sums = y_tp
            ax.bar(x_adapted, y_fn, bottom=sums, color=yellow, edgecolor="white", hatch="x", width=width)
            sums += y_fn
            ax.bar(x_adapted, y_fp, bottom=sums, color=red, edgecolor='white', hatch=".", width=width)
            # sums = [_x + _y for _x, _y in zip(sums, y_tp)]
            # ax.bar(x_adapted, y_tn, bottom=sums, color=[0, 0.9, 0.1], edgecolor='white', width=width)
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    title = "Decision performance {}".format(name)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


def plot_false_decisions_legend():
    fig = plt.figure()
    ax = fig.add_subplot()
    title = "Legend"
    ax.figure.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    labels = 'false positives', 'false negatives', 'true positives'
    width = 0.5

    blue = [0, .4, .6]
    yellow = [1, 0.65, 0.25]
    red = [1, 0, 0]
    res1 = ax.bar([1], [1], color=red, edgecolor='white', hatch=".", width=width)
    res2 = ax.bar([1], [2], bottom=[1], color=yellow, edgecolor="white", hatch="x", width=width)
    res3 = ax.bar([1], [3], bottom=[2], color=blue, edgecolor="white", width=width)
    ax.cla()
    plt.axis('off')
    ax.legend((res1[0], res2[0], res3[0]), labels, loc="center", handleheight=3)


def get_xticks_bars(confidence_thresholds, n=10, to_float=True):
    # x ticks
    step = int(len(confidence_thresholds) / n)
    x = []
    for confidence_threshold in confidence_thresholds:
        xi = float_printer(confidence_threshold) if to_float else confidence_threshold
        x.append(xi)
    xticks = [x[i * step] for i in range(n)]
    return x, xticks


def plot_decisions_of_two_approaches(monitor1, history1: History, confidence_threshold1: float,
                                     monitor2, history2: History, confidence_threshold2: float,
                                     classes_network: list, classes_rest: list):
    # collect data
    m_id1 = monitor1 if isinstance(monitor1, int) else monitor1.id()
    m_id2 = monitor2 if isinstance(monitor2, int) else monitor2.id()
    ground_truths = history1.ground_truths
    predictions = history1.predictions
    results1 = history1.monitor2results[m_id1]
    results2 = history2.monitor2results[m_id2]
    class2category2numbers = dict()
    category2correctness2numbers = {"a1 a2": {True: 0, False: 0}, "a1 r2": {True: 0, False: 0},
                                    "r1 a2": {True: 0, False: 0}, "r1 r2": {True: 0, False: 0}}
    known_category2correctness2numbers = {"a1 a2": {True: 0, False: 0}, "a1 r2": {True: 0, False: 0},
                                          "r1 a2": {True: 0, False: 0}, "r1 r2": {True: 0, False: 0}}
    novel_category2numbers = {"a1 a2": 0, "a1 r2": 0, "r1 a2": 0, "r1 r2": 0}
    all_classes = sorted(set(classes_network + classes_rest))
    n_classes = len(all_classes)
    for class_id in all_classes:
        class2category2numbers[class_id] = {"a1 a2": 0, "a1 r2": 0, "r1 a2": 0, "r1 r2": 0}
    for gt, pd, r1, r2 in zip(ground_truths, predictions, results1, results2)\
            :  # type: int, int, MonitorResult, MonitorResult
        if r1.accepts(confidence_threshold1):
            if r2.accepts(confidence_threshold2):
                category = "a1 a2"
            else:
                category = "a1 r2"
        else:
            if r2.accepts(confidence_threshold2):
                category = "r1 a2"
            else:
                category = "r1 r2"
        class2category2numbers[gt][category] += 1
        category2correctness2numbers[category][gt == pd] += 1
        if gt in classes_network:
            known_category2correctness2numbers[category][gt == pd] += 1
        else:
            novel_category2numbers[category] += 1

    # create figures
    fig, ax, n_cols, row, col = initialize_subplots(4, "Comparison with confidences {:f} and {:f}".format(
        confidence_threshold1, confidence_threshold2))
    x = ["a1 a2", "a1 r2", "r1 a2", "r1 r2"]

    # plot comparison by classes
    sums = [0 for _ in range(4)]
    colors = get_rgb_colors(n_classes)
    for i, class_id in enumerate(all_classes):
        current = [class2category2numbers[class_id][x[i]] for i in range(4)]
        is_known_class = class_id in classes_network
        label = "class {:d} ({})".format(class_id, "o" if is_known_class else "n")
        ax[0][0].bar(x, current, bottom=sums, color=colors[i], edgecolor='white', width=0.5, label=label)
        for i in range(4):
            sums[i] += current[i]
    ax[0][0].set_xlabel("By classes")
    ax[0][0].legend()

    # plot comparison by correct/incorrect
    current = [category2correctness2numbers[x[i]][True] for i in range(4)]
    ax[0][1].bar(x, current, color="b", edgecolor='white', width=0.5, label="correct")
    current2 = [category2correctness2numbers[x[i]][False] for i in range(4)]
    ax[0][1].bar(x, current2, bottom=current, color="r", edgecolor='white', width=0.5, label="incorrect")
    ax[0][1].set_xlabel("All classes")
    ax[0][1].legend()

    # plot comparison by correct/incorrect for known classes
    current = [known_category2correctness2numbers[x[i]][True] for i in range(4)]
    ax[1][0].bar(x, current, color="b", edgecolor='white', width=0.5, label="correct")
    current2 = [known_category2correctness2numbers[x[i]][False] for i in range(4)]
    ax[1][0].bar(x, current2, bottom=current, color="r", edgecolor='white', width=0.5, label="incorrect")
    ax[1][0].set_xlabel("Known classes")
    ax[1][0].legend()

    # plot comparison by correct/incorrect for novel classes
    current = [novel_category2numbers[x[i]] for i in range(4)]
    ax[1][1].bar(x, current, color="r", edgecolor='white', width=0.5)
    ax[1][1].set_xlabel("Novel classes")

    plt.draw()
    plt.pause(0.0001)


# taken from https://stackoverflow.com/a/26369255
def save_all_figures(figs=None, extension="pdf", close=False):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:  # type
        fig.savefig("../{}.{}".format(fig._suptitle._text, extension))
        if close:
            plt.close(fig)

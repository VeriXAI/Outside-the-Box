from . import *


def print_statistics(statistics: Statistics, monitor_manager, n_train_model: int, n_train_monitor: int,
                     n_test_monitor: int, n_run: int, epochs: int, novelty_wrapper: NoveltyWrapper, history: History,
                     confidence_thresholds: list):
    print("\n--- final statistics ---\n")

    # general statistics
    print("overall statistics")
    if statistics.time_training_model > 0:
        print("{} seconds for training model on {:d} samples with {:d} epochs".format(
            float_printer(statistics.time_training_model), n_train_model, epochs))
    print("{} seconds for extracting {:d} values during monitor training".format(
        float_printer(statistics.time_training_monitor_value_extraction), n_train_monitor))
    print("{} seconds for clustering during monitor training".format(
        float_printer(statistics.time_training_monitor_clustering)))
    print("{} seconds for monitor training on {:d} samples".format(
        float_printer(statistics.time_training_monitor_tweaking), n_train_monitor))
    print("{} seconds for extracting {:d} values during running the monitored session".format(
        float_printer(statistics.time_running_monitor_value_extraction), n_run))
    print("{} seconds for running the monitored session on {:d} samples".format(
        float_printer(statistics.time_running_monitor_classification), n_run))
    correct_classifications, incorrect_classifications = history.classification_statistics()
    n = correct_classifications
    d = correct_classifications + incorrect_classifications
    print("success rate of the unmonitored classifier: {:d} / {:d} = {:.2f} % ({:d} misclassifications)".format(
        n, d, ratio(n, d), incorrect_classifications))

    # individual statistics for each monitor
    monitors = monitor_manager.monitors()
    for i, monitor in enumerate(monitors):
        m_id = monitor.id()
        print("\nprinting statistics for monitor {:d} with abstraction structure {}".format(m_id, monitor.short_str()))
        print("{} seconds for training the monitor on {:d} samples".format(
            float_printer(statistics.time_tweaking_each_monitor[m_id]), n_train_monitor))
        print("{} seconds for running the monitor on {:d} samples".format(
            float_printer(statistics.time_running_each_monitor[m_id]), n_run))
        for confidence_threshold in confidence_thresholds:
            print("\n confidence threshold {:f} ".format(confidence_threshold))
            # recompute statistics for the monitor and confidence threshold
            history.update_statistics(m_id, confidence_threshold=confidence_threshold)
            tn = history.true_negatives()
            tp = history.true_positives()
            fn = history.false_negatives()
            fp = history.false_positives()
            zero_filtered = history.zero_filtered()
            tn_string = str(tn)
            tp_string = str(tp)
            fp_string = str(fp)
            fn_string = str(fn)
            zero_filtered_string = str(zero_filtered)
            tn_string, tp_string, fp_string, fn_string, zero_filtered_string =\
                extend([tn_string, tp_string, fp_string, fn_string, zero_filtered_string])
            print(tn_string, "samples were classified   correctly and accepted by the monitor (+) (",
                  float_printer(ratio(tn, n_run)), "%)")
            print(fp_string, "samples were classified   correctly but rejected by the monitor (-) (",
                  float_printer(ratio(fp, n_run)), "%)")
            print(tp_string, "samples were classified incorrectly and rejected by the monitor (+) (",
                  float_printer(ratio(tp, n_run)), "%)")
            print(fn_string, "samples were classified incorrectly but accepted by the monitor (-) (",
                  float_printer(ratio(fn, n_run)), "%)")
            print(zero_filtered_string, "samples were rejected by default because of a mismatch in the " +
                  "zero-dimension pattern ({}%)".format(float_printer(ratio(zero_filtered, n_run))))
            n = tp
            d = tp + fn
            print("detection rate of the monitor:", n, "/", d, "=", float_printer(ratio(n, d)),
                  "% (of the incorrectly classified samples)")
            n = fp
            d = fp + tn
            print("false-warning rate of the monitor:", n, "/", d, "=", float_printer(ratio(n, d)),
                  "% (of the correctly classified samples)")
            novelties = novelty_wrapper.evaluate_detection(m_id, confidence_threshold)
            n = len(novelties["detected"])
            d = n + len(novelties["undetected"])
            print("novelty detection: {:d} / {:d}".format(n, d))
    print()

    # plot statistics after timers
    if PLOT_MONITOR_PERFORMANCE:
        factors = [1, -1] if monitor_manager._alpha_thresholding else [1]
        for i, monitor in enumerate(monitors):
            for factor in factors:
                m_id = monitor.id() * factor
                for confidence_threshold in confidence_thresholds:
                    # recompute statistics for the given monitor
                    history.update_statistics(m_id, confidence_threshold=confidence_threshold)
                    tn = history.true_negatives()
                    tp = history.true_positives()
                    fn = history.false_negatives()
                    fp = history.false_positives()
                    fig, ax = initialize_single_plot("Performance of monitor {:d} & confidence >= {:f}".format(
                        m_id, confidence_threshold))
                    plot_pie_chart_single(ax=ax, tp=tp, tn=tn, fp=fp, fn=fn, n_run=n_run)
        fig, ax = initialize_single_plot("Performance of monitor (legend)")
        plot_pie_chart_single(ax=ax, tp=0, tn=0, fp=0, fn=0, n_run=-1)

from utils import *
from abstractions import *
from trainers import *
from monitoring import *
from run.Runner import run

from abstractions.ConvexHull import ConvexHull


def run_script():
    # options
    seed = 0
    data_name = "MNIST"
    classes = [0, 1]
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(n=300, randomize=True, classes=[0, 1])
    data_test_monitor = DataSpec(n=100, randomize=False, classes=[2])
    data_run = DataSpec(n=200, randomize=True, classes=[0, 1, 9])
    model_name = "MNIST"
    model_path = "MNIST_2-model.h5"
    n_epochs = 2
    batch_size = 128
    score_fun = F1Score()
    # confidence_thresholds = [0]
    confidence_thresholds = uniform_bins(1, max=1)
    alpha = 0.95
    alphas = [0.99, 0.95, 0.9, 0.5]

    # bva = BooleanAbstraction(gamma=1)
    # bva.initialize(2)
    # bva.add_clustered([[0., 1.]], None)

    # chull = ConvexHull(2)
    # chull.create([[0., 0.1], [0., 1.], [1., 1.], [1., 0.], [0.5, 0.5]])
    # print(chull.contains([0., 0.]))
    # print(chull.contains([0., 1.]))
    # print(chull.contains([1., 1.]))
    # print(chull.contains([1., 0.]))
    # print(chull.contains([0.5, 0.5]))
    # print(chull.contains([1.5, 1.5]))
    # print(chull.contains([-0.5, -0.5]))
    # chull.plot([0, 1], "r", 0, False, ax=plt.figure().add_subplot())
    # plt.draw()
    # plt.pause(0.0001)

    # o = Octagon(2)
    # vertices = [[-0.75, -0.25], [-0.25, -0.75], [0.25, -0.75], [0.75, -0.25], [0.75, 0.25], [0.25, 0.75], [-0.25, 0.75],
    #             [-0.75, 0.25]]
    # o.create(vertices[0])
    # for v in vertices[1:]:
    #     print(v)
    #     o.add(v)
    # xs = [3., -3.]
    # ys = [3., -3.]
    # ax = plt.figure().add_subplot()
    # print()
    # for hs in o.half_spaces(epsilon=0.0, epsilon_relative=True):
    #     print(hs)
    # print()
    # for hs in o.half_spaces(epsilon=1.0, epsilon_relative=True):
    #     print(hs)
    # print()
    # for hs in o.half_spaces(epsilon=1.0, epsilon_relative=False):
    #     print(hs)
    # print()
    # o.plot(dims=[0, 1], ax=ax, color="b", epsilon=0.0, epsilon_relative=False)
    # o.box.plot(dims=[0, 1], ax=ax, color="b", epsilon=0.0, epsilon_relative=False)
    # o.plot(dims=[0, 1], ax=ax, color="r", epsilon=1.0, epsilon_relative=False)
    # o.box.plot(dims=[0, 1], ax=ax, color="r", epsilon=1.0, epsilon_relative=False)
    # o.plot(dims=[0, 1], ax=ax, color="g", epsilon=1.0, epsilon_relative=True)
    # o.box.plot(dims=[0, 1], ax=ax, color="g", epsilon=1.0, epsilon_relative=True)
    # ax.scatter(xs, ys, alpha=0.5)
    # plt.show()
    # print("end")

    # model trainer
    model_trainer = StandardTrainer()

    # abstractions
    confidence_fun_euclidean = euclidean_distance
    confidence_fun_half_space = halfspace_distance
    # abstraction_minus2 = CompoundAbstraction([BoxAbstraction(5, 0.1), ZoneAbstraction(5, 1.0)])
    # abstraction_minus2 = ConvexHullAbstraction(1, 0.0)
    # abstraction_minus2 = HypersphereAbstraction(1, 0.0)
    # abstraction_minus3 = BoxAbstraction(1, 0.1)
    # layer2abstraction = {-2: abstraction_minus2, -3: abstraction_minus3}
    # layer2dimensions = {-2: [1, 2], -3: [0, 1]}
    # monitor1 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    #
    # abstraction_minus3_2 = CompositeAbstraction([ZoneAbstraction(1, 0.6), BoxAbstraction(1, 0.1)])
    # layer2abstraction = {-3: abstraction_minus3_2}
    # layer2dimensions = {-3: [0, 1]}
    # monitor2 = Monitor(layer2abstraction, score_fun, layer2dimensions)

    box_abstraction = BoxAbstraction(confidence_fun_euclidean, epsilon=0.0)
    box_abstraction_hs = BoxAbstraction(confidence_fun_half_space, epsilon=0.0)
    zone_abstraction = ZoneAbstraction(confidence_fun_half_space)
    chull_abstraction = ConvexHullAbstraction(confidence_fun_half_space, remove_redundancies=True)
    meanball_abstraction_low_dim = MeanBallAbstraction(confidence_fun_euclidean)
    octagon_abstraction = OctagonAbstraction(confidence_fun_half_space)
    partition_meanball_abstraction = PartitionBasedAbstraction(1, partition=uniform_partition(40, 2),
                                                               abstractions=meanball_abstraction_low_dim)
    chull_abstraction_low_dim = ConvexHullAbstraction(confidence_fun_half_space, remove_redundancies=True, epsilon=0.0)
    partition__chull_abstraction = PartitionBasedAbstraction(1, partition=uniform_partition(40, 8),
                                                             abstractions=chull_abstraction_low_dim)
    cheap_boolean_abstraction_single = CheapBooleanAbstraction()
    layer2dimensions = {-3: [68, 69], -2: [12, 14]}
    layer2abstraction = {-2: box_abstraction}
    monitor1 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    monitor1_w_novelties = Monitor(layer2abstraction, score_fun, layer2dimensions, is_novelty_training_active=True)
    layer2abstraction = {-2: zone_abstraction}
    monitor2 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: partition_meanball_abstraction}
    monitor3 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: chull_abstraction}
    monitor4 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: partition__chull_abstraction}
    monitor5 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: box_abstraction_hs}
    monitor6 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: copy(zone_abstraction), -3: copy(zone_abstraction)}
    monitor7 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: octagon_abstraction}
    monitor8 = Monitor(layer2abstraction, score_fun, layer2dimensions)
    layer2abstraction = {-2: cheap_boolean_abstraction_single}
    monitor9 = Monitor(layer2abstraction, score_fun, layer2dimensions)

    monitors = [monitor1, monitor2, monitor3, monitor4, monitor5]
    monitors = [monitor1, monitor5, monitor6]
    monitors = [monitor5]
    monitors = [monitor1, monitor2, monitor6, monitor8]
    monitors = [monitor3, monitor5]
    monitors = [monitor9]
    monitors = [monitor9, monitor1]
    monitors = [monitor1, monitor1_w_novelties]
    monitor_manager = MonitorManager(monitors, clustering_threshold=0.1, n_clusters=3)

    # general run script
    # run(seed=seed, data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
    #     data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run,
    #     model_trainer=model_trainer, model_name=model_name, model_path=model_path, n_epochs=n_epochs,
    #     batch_size=batch_size, monitor_manager=monitor_manager, confidence_thresholds=confidence_thresholds,
    #     skip_image_plotting=True)
    # evaluate_combination(seed=seed, data_name=data_name, data_train_model=data_train_model,
    #                      data_test_model=data_test_model, data_train_monitor=data_train_monitor,
    #                      data_test_monitor=data_test_monitor, data_run=data_run, model_trainer=model_trainer,
    #                      model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
    #                      monitor_manager=monitor_manager, confidence_thresholds=confidence_thresholds,
    #                      skip_image_plotting=True)
    history_run, histories_alpha_thresholding, novelty_wrapper_run, novelty_wrappers_alpha_thresholding, \
        statistics = evaluate_all(seed=seed, data_name=data_name, data_train_model=data_train_model,
                         data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                         data_test_monitor=data_test_monitor, data_run=data_run, model_trainer=model_trainer,
                         model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
                         monitor_manager=monitor_manager, alphas=alphas)

    history_run.update_statistics(1)
    fn = history_run.false_negatives()
    fp = history_run.false_positives()
    tp = history_run.true_positives()
    tn = history_run.true_negatives()
    novelty_results = novelty_wrapper_run.evaluate_detection(1)
    storage_1 = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                               novelties_undetected=len(novelty_results["undetected"]))
    storages_1 = [storage_1]

    history_run.update_statistics(2)
    fn = history_run.false_negatives()
    fp = history_run.false_positives()
    tp = history_run.true_positives()
    tn = history_run.true_negatives()
    novelty_results = novelty_wrapper_run.evaluate_detection(2)
    storage_2 = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                               novelties_undetected=len(novelty_results["undetected"]))
    storages_2 = [storage_2]

    storages_at = []
    for history_alpha, novelty_wrapper_alpha, alpha in zip(
            histories_alpha_thresholding, novelty_wrappers_alpha_thresholding, alphas):
        # history_alpha.update_statistics(0, confidence_threshold=alpha)  # not needed: history is already set
        fn = history_alpha.false_negatives()
        fp = history_alpha.false_positives()
        tp = history_alpha.true_positives()
        tn = history_alpha.true_negatives()
        novelty_results = novelty_wrapper_alpha.evaluate_detection(0)
        storage = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn, novelties_detected=len(novelty_results["detected"]),
                                 novelties_undetected=len(novelty_results["undetected"]))
        storages = [storage]
        storages_at.append(storages)

    # store results
    store_core_statistics(storages_1, "monitor1")
    store_core_statistics(storages_2, "monitor2")
    store_core_statistics(storages_at, alphas)

    # load results
    storages_1b = load_core_statistics("monitor1")
    storages_2b = load_core_statistics("monitor2")
    storages_atb = load_core_statistics(alphas)
    pass


if __name__ == "__main__":
    run_script()

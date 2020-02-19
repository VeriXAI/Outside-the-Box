import matplotlib.pyplot as plt

# verbosity
VERBOSE_MODEL_TRAINING = False  # print debug output of training the model
N_PRINT_WARNINGS = 49  # number of monitor warnings printed
N_PRINT_NOVELTIES = 49  # number of monitor novelties printed
PRINT_CONVEX_HULL_SAVED_VERTICES = False  # print number of vertices saved by convex hulls
PRINT_CREDIBILITY = True  # print credibility of abstractions

# plotting
N_HISTOGRAM_PLOTS_UPPER_BOUND = 0  # maximum number of histogram plots
PLOT_MEAN = False  # plot mean of abstractions? (requires COMPUTE_MEAN == True)
PLOT_NON_EPSILON_SETS = False  # plot inner sets if epsilon is used?
_PLOT_MONITOR_TRAINING_AXIS = None
_PLOT_MONITOR_RATES_AXIS = None
PLOT_MONITOR_PERFORMANCE = True  # plot monitor performance (pie charts)?


def PLOT_MONITOR_TRAINING_AXIS():  # plot window for monitor training
    global _PLOT_MONITOR_TRAINING_AXIS
    if _PLOT_MONITOR_TRAINING_AXIS is None:
        _PLOT_MONITOR_TRAINING_AXIS = plt.figure().add_subplot()
    return _PLOT_MONITOR_TRAINING_AXIS


def PLOT_MONITOR_RATES_AXIS():
    global _PLOT_MONITOR_RATES_AXIS
    if _PLOT_MONITOR_RATES_AXIS is None:
        _PLOT_MONITOR_RATES_AXIS = plt.subplots(1, 3)
    return _PLOT_MONITOR_RATES_AXIS


# general
FILTER_ZERO_DIMENSIONS = False  # filter out dimensions that were zero in the monitor training?


# monitor-training related
COMPUTE_MEAN = False  # compute the mean of 'SetBasedAbstraction's?
CONVEX_HULL_REDUNDANCY_REMOVAL = False  # remove redundant vertices from convex hulls?
CONVEX_HULL_REMOVE_BATCHES = False  # remove several points at once in the convex hulls? (almost no practical effect)
ONLY_LEARN_FROM_CORRECT_CLASSIFICATIONS = True  # ignore misclassification samples during monitor training?
MONITOR_TRAINING_CONVERGENCE_RANGE = 0.001  # range that is considered "no change" during monitor training
MONITOR_TRAINING_WINDOW_SIZE = 5  # convergence window (= number of data points in which no change occurs) during
#                                   monitor training
PRINT_FLAT_CONVEX_HULL_WARNING = True  # print warning about flat convex hulls


def print_flat_convex_hull_warning():
    global PRINT_FLAT_CONVEX_HULL_WARNING
    if PRINT_FLAT_CONVEX_HULL_WARNING:
        PRINT_FLAT_CONVEX_HULL_WARNING = False
        print("Warning: Convex hull is flat, for which conversion to H-representation is not available.")


# monitor-running related
PROPOSE_CLASS = False  # let the monitor propose a class based on the mean? (requires COMPUTE_MEAN == True)
MAXIMUM_CONFIDENCE = 1.0  # maximum confidence for rejection
ACCEPTANCE_CONFIDENCE = 0.0  # confidence when accepting
INCREDIBLE_CONFIDENCE = 1.0  # confidence when rejecting due to incredibility
SKIPPED_CONFIDENCE_NOVELTY_MODE = -1.0  # confidence when training novelties (has no meaning)
SKIPPED_CONFIDENCE = 1.0  # confidence when no distance is used
CONVEX_HULL_HALF_SPACE_DISTANCE_CORNER_CASE = 0.0  # half-space confidence for flat convex hulls
COMPOSITE_ABSTRACTION_POLICY = 2  # policy for CompositeAbstraction and multi-layer monitors; possible values:
#                                   1: average
#                                   2: maximum


# --- defaults --- #


# data-loading related
N_TRAIN = 2000  # number of training data points
N_TEST = 1000  # number of testing data points
N_RUN = 1000  # number of running data points
RANDOMIZE_DATA = False  # randomize the data after loading?
CLASSES = [0, 1]  # classes for filtering (empty list: any)
# network related
N_EPOCHS = 10  # number of training epochs
BATCH_SIZE = 128  # batch size
# utilities
DEFAULT_SEED = 0  # default random seed

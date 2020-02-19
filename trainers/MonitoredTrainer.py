from .CustomTrainer import CustomTrainer
from monitoring import *
from utils import *


class MonitoredTrainer(CustomTrainer):
    def __init__(self, abstractions, layer_index, n_classes, adaptive_monitoring):
        super(MonitoredTrainer, self).__init__()
        self.abstractions = abstractions
        self.layer_index = layer_index
        self.n_classes = n_classes
        self.adaptive = adaptive_monitoring

    def __str__(self):
        return "MonitoredTrainer"

    def train_epoch(self, model, train_ds, test_ds, epoch, data_train: DataSpec, data_test: DataSpec):
        super(MonitoredTrainer, self).train_epoch(model, train_ds, test_ds, epoch, data_train, data_test)

        # train abstraction
        if epoch > 0:
            for a in self.abstractions:
                a.clear()
        monitors = [Monitor(self.layer_index, a, self.adaptive) for a in self.abstractions]
        monitor_manager = MonitorManager(model, monitors, TRAINING_POLICY_GROUND_TRUTH)
        monitor_manager.train_monitors(data_train)

        # evaluate abstraction
        statistics = Statistics()
        monitor_manager.run_monitors(self.n_classes, data_test, statistics, verbose=False,
                                     compute_alternative_classes=False, print_result=True)

        # terminate if there are no false characterizations in the test set
        for i in range(len(self.abstractions)):
            statistics.update_statistics(i)
            if statistics.false_negatives() + statistics.false_positives() > 0:
                return False
        return True

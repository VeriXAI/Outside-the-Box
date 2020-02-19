from utils import *


class History(object):
    def __init__(self):
        self.ground_truths = []
        self.predictions = []
        self.monitor2results = dict()
        self.layer2values = dict()
        self._tn = 0
        self._tp = 0
        self._fn = 0
        self._fp = 0
        self._zero_filtered = 0

    def set_ground_truths(self, ground_truths):
        self.ground_truths = ground_truths

    def set_layer2values(self, layer2values):
        self.layer2values = layer2values

    def set_predictions(self, predictions):
        self.predictions = predictions

    def set_monitor_results(self, m_id, results):
        self.monitor2results[m_id] = results

    def classification_statistics(self):
        correct_classifications = 0
        for j, (c_ground_truth, c_prediction) in enumerate(zip(self.ground_truths, self.predictions)):
            if c_prediction == c_ground_truth:
                correct_classifications += 1
        incorrect_classifications = len(self.ground_truths) - correct_classifications
        return correct_classifications, incorrect_classifications

    def true_negatives(self):
        return self._tn

    def true_positives(self):
        return self._tp

    def false_negatives(self):
        return self._fn

    def false_positives(self):
        return self._fp

    def zero_filtered(self):
        return self._zero_filtered

    def true_positive_rate(self):
        tp = self.true_positives()
        fn = self.false_negatives()
        return rate_fraction(tp, tp + fn)

    def true_negative_rate(self):
        tn = self.true_negatives()
        fp = self.false_positives()
        return rate_fraction(tn, tn + fp)

    def false_positive_rate(self):
        fp = self.false_positives()
        tn = self.true_negatives()
        return rate_fraction(fp, fp + tn)

    def false_negative_rate(self):
        fn = self.false_negatives()
        tp = self.true_positives()
        return rate_fraction(fn, fn + tp)

    def positive_predictive_value(self):
        tp = self.true_positives()
        fp = self.false_positives()
        return rate_fraction(tp, tp + fp)

    def negative_predictive_value(self):
        tn = self.true_negatives()
        fn = self.false_negatives()
        return rate_fraction(tn, tn + fn)

    # alias
    def precision(self):
        return self.positive_predictive_value()

    # alias
    def recall(self):
        return self.true_positive_rate()

    # F1 score = harmonic mean between precision and recall, see https://en.wikipedia.org/wiki/F1_score
    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return rate_fraction(2. * (p * r), p + r)

    def average_score(self):
        p = self.precision()
        r = self.recall()
        return (p + r) / 2

    def warnings(self, monitor, data, confidence_threshold=0.0):
        warning_list = []
        for i, (image, gt, p, res) in enumerate(zip(data.x(), self.ground_truths, self.predictions,
                                                    self.monitor2results[monitor.id()])):
            if not res.accepts(confidence_threshold):
                warning = Anomaly(input=image, c_ground_truth=gt, c_predicted=p, status=Anomaly.WARNING)
                warning_list.append(warning)
        return warning_list

    def novelties(self, data: DataSpec, classes_network, classes_rest):
        # collect new labels
        anomaly_labels = []
        for class_id in classes_rest:
            if class_id not in classes_network:
                anomaly_labels.append(class_id)

        # identify anomalies
        novelties = []
        novelty_indices = []
        ground_truths = []
        predictions = []
        for i, (image, gt, p) in enumerate(zip(data.x(), self.ground_truths, self.predictions)):
            if gt in anomaly_labels:
                novelties.append(image)
                novelty_indices.append(i)
                ground_truths.append(gt)
                predictions.append(p)

        return NoveltyWrapper(novelties, novelty_indices, ground_truths, predictions, self.monitor2results)

    def update_statistics(self, monitor_id, confidence_threshold=0.0, n_min_acceptance=None):
        results = self.monitor2results[monitor_id]
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        zero_filtered = 0
        for j, (c_ground_truth, c_prediction, result) in enumerate(zip(self.ground_truths, self.predictions, results)):
            if result.is_zero_filtered():
                zero_filtered += 1
            accepts = result.accepts(confidence_threshold, n_min_acceptance)
            is_correct = c_prediction == c_ground_truth
            if is_correct:
                if accepts:
                    true_negatives += 1
                else:
                    false_positives += 1
            else:
                if accepts:
                    false_negatives += 1
                else:
                    true_positives += 1
        self._tn = true_negatives
        self._tp = true_positives
        self._fn = false_negatives
        self._fp = false_positives
        self._zero_filtered = zero_filtered

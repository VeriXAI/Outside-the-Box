from utils import *


class ConditionalHistory(History):
    def __init__(self, histories_guard: list, histories_body: list, confidence_guard: float):
        super().__init__()
        assert histories_guard and histories_body, "ConditionalHistory needs two nonempty lists."
        for history in histories_guard[1:] + histories_body:
            assert histories_guard[0].ground_truths == history.ground_truths and \
                   histories_guard[0].predictions == history.predictions, "The histories are incompatible."
        self.set_ground_truths(histories_guard[0].ground_truths)
        self.set_predictions(histories_guard[0].predictions)
        self.monitor2results = None
        self.monitor2results_guard = CombinedHistory(histories_guard).monitor2results
        self.monitor2results_body = CombinedHistory(histories_body).monitor2results
        self.confidence_guard = confidence_guard

    def update_statistics(self, monitor_id, confidence_threshold=0.0, n_min_acceptance=None):
        results_guard = self.monitor2results_guard[monitor_id]
        results_body = self.monitor2results_body[monitor_id]
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        zero_filtered = 0
        for j, (c_ground_truth, c_prediction, result_guard, result_body) in \
                enumerate(zip(self.ground_truths, self.predictions, results_guard, results_body)):
            guard_accepts = result_guard.accepts(self.confidence_guard, n_min_acceptance)
            if guard_accepts:
                accepts = True
            else:
                accepts = result_body.accepts(confidence_threshold, n_min_acceptance)
                if not accepts and result_guard.is_zero_filtered() and result_body.is_zero_filtered():
                    zero_filtered += 1
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

    def novelties(self, data: DataSpec, classes_network, classes_rest):
        nw = super().novelties(data, classes_network, classes_rest)
        return ConditionalNoveltyWrapper(nw.novelties, nw.novelty_indices, nw.ground_truths, nw.predictions,
                                         self.monitor2results_guard, self.monitor2results_body, self.confidence_guard)

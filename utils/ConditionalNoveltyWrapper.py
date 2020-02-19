from utils import *


class ConditionalNoveltyWrapper(NoveltyWrapper):
    def __init__(self, novelties, novelty_indices, ground_truths, predictions, monitor2results_guard,
                 monitor2results_body, confidence_guard: float):
        super().__init__(novelties, novelty_indices, ground_truths, predictions, None)
        self.monitor2results_guard = monitor2results_guard
        self.monitor2results_body = monitor2results_body
        self.confidence_guard = confidence_guard

    def evaluate_detection(self, monitor_id, confidence_threshold=0.0, n_min_acceptance=None):
        detected = []
        undetected = []
        results_guard = self.monitor2results_guard[monitor_id]
        results_body = self.monitor2results_body[monitor_id]
        for i, image, gt, p in zip(self.novelty_indices, self.novelties, self.ground_truths, self.predictions):
            result_guard = results_guard[i]  # Type: MonitorResult
            result_body = results_body[i]  # Type: MonitorResult
            guard_accepts = result_guard.accepts(self.confidence_guard, n_min_acceptance)
            if guard_accepts:
                accepts = True
            else:
                accepts = result_body.accepts(confidence_threshold, n_min_acceptance)
            if accepts:
                novelty = Anomaly(input=image, c_ground_truth=gt, c_predicted=p, status=Anomaly.UNDETECTED)
                undetected.append(novelty)
            else:
                novelty = Anomaly(input=image, c_ground_truth=gt, c_predicted=p, status=Anomaly.DETECTED)
                detected.append(novelty)

        return {"detected": detected, "undetected": undetected}

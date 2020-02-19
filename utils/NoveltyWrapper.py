from utils import *


class NoveltyWrapper(object):
    def __init__(self, novelties, novelty_indices, ground_truths, predictions, monitor2results):
        self.novelties = novelties
        self.novelty_indices = novelty_indices
        self.ground_truths = ground_truths
        self.predictions = predictions
        self.monitor2results = monitor2results

    def evaluate_detection(self, monitor_id, confidence_threshold=0.0, n_min_acceptance=None):
        detected = []
        undetected = []
        results = self.monitor2results[monitor_id]
        for i, image, gt, p in zip(self.novelty_indices, self.novelties, self.ground_truths, self.predictions):
            res = results[i]  # Type: MonitorResult
            if res.accepts(confidence_threshold, n_min_acceptance):
                novelty = Anomaly(input=image, c_ground_truth=gt, c_predicted=p, status=Anomaly.UNDETECTED)
                undetected.append(novelty)
            else:
                novelty = Anomaly(input=image, c_ground_truth=gt, c_predicted=p, status=Anomaly.DETECTED)
                detected.append(novelty)

        return {"detected": detected, "undetected": undetected}

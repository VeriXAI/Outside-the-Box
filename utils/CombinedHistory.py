from utils import *


class CombinedHistory(History):
    def __init__(self, histories: list):
        super().__init__()
        for history in histories[1:]:
            assert histories[0].ground_truths == history.ground_truths and \
                   histories[0].predictions == history.predictions, "The histories are incompatible."
        self.set_ground_truths(histories[0].ground_truths)
        self.set_predictions(histories[0].predictions)
        self.monitor2results = self._merge_results(histories)

    def _merge_results(self, histories):
        results_new = [MonitorResult() for _ in range(len(self.ground_truths))]
        for history in histories:
            for results in history.monitor2results.values():
                for result_new, result_old in zip(results_new, results):  # Type: MonitorResult, MonitorResult
                    for confidence in result_old._confidences:
                        result_new.add_confidence(confidence)
        return {0: results_new}

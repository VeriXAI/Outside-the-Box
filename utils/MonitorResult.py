from utils import *


class MonitorResult(object):
    def __init__(self):
        self._confidences = []
        self._is_zero_filtered = False

    def __str__(self):
        if self.is_zero_filtered():
            result = "rejecting due to zero-dimension-pattern mismatch"
        else:
            result = "rejecting with confidence {:f}".format(self.confidence())
        return "{:d} abstractions considered (→ {})".format(len(self._confidences), result)

    def add_confidence(self, confidence: float):
        self._confidences.append(confidence)

    def accepts(self, confidence_threshold, n_min_acceptance=None):
        if self.is_zero_filtered():
            return False
        elif n_min_acceptance is None:
            return self.confidence() <= confidence_threshold
        else:
            n_accepting = 0
            for confidence in self._confidences:
                if confidence <= confidence_threshold:
                    n_accepting += 1
            if n_min_acceptance > 0:
                return n_accepting >= n_min_acceptance
            else:
                n_total = len(self._confidences)
                n_rejecting = n_total - n_accepting
                return n_rejecting < -n_min_acceptance

    def confidence(self):
        if self.is_zero_filtered():
            return MAXIMUM_CONFIDENCE
        if COMPOSITE_ABSTRACTION_POLICY == 1:
            # average
            return sum(self._confidences) / float(len(self._confidences))
        elif COMPOSITE_ABSTRACTION_POLICY == 2:
            # maximum
            return max(self._confidences)
        else:
            raise NotImplementedError("Policy {} is not available.".format(COMPOSITE_ABSTRACTION_POLICY))

    def set_zero_filter(self):
        self._is_zero_filtered = True

    def is_zero_filtered(self):
        return self._is_zero_filtered

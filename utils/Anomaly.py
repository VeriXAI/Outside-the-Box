class Anomaly(object):
    WARNING = 0
    DETECTED = 1
    UNDETECTED = 2

    def __init__(self, input, c_ground_truth, c_predicted, status):
        self.original_input = input
        self.c_ground_truth = c_ground_truth
        self.c_predicted = c_predicted
        self.status = status

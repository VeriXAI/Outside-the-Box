class CoreStatistics(object):
    def __init__(self, tn, fn, tp, fp, novelties_detected, novelties_undetected, time_training=-1, time_running=-1):
        self.tn = tn
        self.fn = fn
        self.tp = tp
        self.fp = fp
        self.novelties_detected = novelties_detected
        self.novelties_undetected = novelties_undetected
        self.time_training = time_training
        self.time_running = time_running

    @staticmethod
    def row_header():
        return ["tn", "fn", "tp", "fp", "nov_det", "nov_undet", "t_train", "t_run"]

    def as_row(self):
        return [self.tn, self.fn, self.tp, self.fp, self.novelties_detected, self.novelties_undetected,
                self.time_training, self.time_running]

    def get_n(self):
        return self.tn + self.fn + self.tp + self.fp

    @staticmethod
    def parse(row):
        assert len(row) == 8, "Illegal input of length {} received.".format(len(row))
        row_converted = [int(e) for e in row[0:6]] + [float(e) for e in row[6:8]]
        return CoreStatistics(*row_converted)

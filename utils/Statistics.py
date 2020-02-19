from .Helpers import *


# for terminology see https://en.wikipedia.org/wiki/Receiver_operating_characteristic
class Statistics(object):
    def __init__(self):
        self.time_training_model = -1
        self.time_training_monitor_value_extraction = -1
        self.time_training_monitor_clustering = -1
        self.time_training_monitor_tweaking = -1
        self.time_running_monitor_value_extraction = -1
        self.time_running_monitor_classification = -1
        self.time_tweaking_each_monitor = dict()
        self.time_running_each_monitor = dict()

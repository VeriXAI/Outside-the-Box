from utils import Statistics, MONITOR_TRAINING_CONVERGENCE_RANGE


class Score(object):
    def evaluate(self, stats):
        pass

    def isbetter(self, score1, score2):
        return score1 > score2  # default: bigger is better

    def name(self):
        return ""

    def termination(self, score_new, score_old):
        return not self.isbetter(score_new, score_old) and\
               abs(score_new - score_old < MONITOR_TRAINING_CONVERGENCE_RANGE)


class F1Score(Score):
    def evaluate(self, stats: Statistics):
        return stats.f1_score()

    def name(self):
        return "F1 score"


class AverageScore(Score):
    def evaluate(self, stats: Statistics):
        return stats.average_score()

    def name(self):
        return "AVG score"

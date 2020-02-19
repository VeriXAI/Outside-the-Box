import sys


# taken from https://stackoverflow.com/a/11325249
class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    @staticmethod
    def start(file):
        f = open(file, "w")
        logger = Logger(sys.stdout, f)
        sys.stdout = logger
        return logger

    def stop(self):
        sys.stdout = self.files[0]
        self.files[1].close()

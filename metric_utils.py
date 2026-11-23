"""Running averages for training and validation metrics."""


class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.total += value * count
        self.count += count
        self.avg = self.total / self.count

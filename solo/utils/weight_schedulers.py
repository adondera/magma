class TriangleScheduler():
    def __init__(self, start_weight, max_weight, end_weight, start_epoch, mid_epoch, end_epoch):
        self.start_weight = start_weight
        self.max_weight = max_weight
        self.end_weight = end_weight
        self.start_epoch = start_epoch
        self.mid_epoch = mid_epoch
        self.end_epoch = end_epoch

    def __call__(self, epoch):
        if epoch < self.start_epoch:
            return self.start_weight
        elif epoch < self.mid_epoch:
            slope = (self.max_weight - self.start_weight) / (self.mid_epoch - self.start_epoch)
            return slope * (epoch - self.start_epoch) + self.start_weight
        elif epoch < self.end_epoch:
            slope = (self.end_weight - self.max_weight) / (self.end_epoch - self.mid_epoch)
            return slope * (epoch - self.mid_epoch) + self.max_weight
        else:
            return self.end_weight


class WarmupScheduler():
    def __init__(self, base_weight, warmup_epochs, weight, reg_epochs):
        self.base_weight = base_weight
        self.warmup_epochs = warmup_epochs
        self.weight = weight
        self.reg_epochs = reg_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.base_weight
        elif epoch < self.reg_epochs + self.warmup_epochs:
            return self.weight
        else:
            return self.base_weight
        
class ConstantScheduler():
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, epoch):
        return self.weight

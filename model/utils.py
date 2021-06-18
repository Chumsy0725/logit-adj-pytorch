
class BasicLRScheduler:

    def __init__(self, optimizer, lr):
        self._m_optimizer = optimizer
        self._m_lr = lr

    def step(self, epoch):
        if (epoch + 1) % 20 == 0:
            self._m_lr /= 3
            for param_group in self._m_optimizer.param_groups:
                param_group['lr'] = self._m_lr

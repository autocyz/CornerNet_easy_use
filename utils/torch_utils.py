import os


def save_params(path, name, params):
    with open(os.path.join(path, name+'.txt'), 'w') as f:
        for key, val in params.items():
            print(key, ' : ', val)
            f.write(key + ': ')
            f.write(str(val))
            f.write('\n')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    from params import params_transform
    save_params('./', 'params_transform', params_transform)

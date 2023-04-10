import os
import sys
import time
import torch


class TextLogger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()


class CompleteLogger:
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.visualize_directory = os.path.join(self.root, "visualize")
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.visualize_directory, exist_ok=True)
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        # redirect std out
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger
        if phase != 'train':
            self.set_epoch(phase)

    def set_epoch(self, epoch):
        """Set the epoch number. Please use it during training."""
        os.makedirs(os.path.join(self.visualize_directory, str(epoch)), exist_ok=True)
        self.epoch = epoch

    def _get_phase_or_epoch(self):
        if self.phase == 'train':
            return str(self.epoch)
        else:
            return self.phase

    def get_image_path(self, filename):
        """
        Get the full image path for a specific filename
        """
        return os.path.join(self.visualize_directory, self._get_phase_or_epoch(), filename)

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.

        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, when the phase is ``train``, checkpoint will be saved to ``{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.

        """
        if name is None:
            name = self._get_phase_or_epoch()
        name = str(name)
        return os.path.join(self.checkpoint_directory, name + ".pth")

    def close(self):
        self.logger.close()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
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
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
from collections import deque, defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import logging, sys, os

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, record_dir, task='e2ec'):
        self.log_path = record_dir
        log_dir = record_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        if 'process_' + task in globals():
            self.processor = globals()['process_' + task]
        else:
            self.processor = None

        self._init_logger()
    
    def _init_logger(self) -> None:
        self.logger = logging.getLogger('info_logger')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt="[%(asctime)s] - %(levelname)s - %(message)s", datefmt=DATE_FORMAT)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        file_handler = logging.FileHandler(os.path.join(self.log_path, "info.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())
    
    def update_loss_stats_ddp(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v)

    def update_image_stats(self, image_stats):
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = ' '.join(loss_state)

        recording_state = ' '.join(['epoch:{}', 'step:{}', '{}', 'data:{:.4f}', 'batch:{:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(record_dir):
    return Recorder(record_dir=record_dir)


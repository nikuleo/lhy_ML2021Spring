import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# tensorboard --logdir=logs
def add_scalars(writer):
    r = 5
    for i in range(100):
        writer.add_scalars(main_tag='scalars1/P1',
                           tag_scalar_dict={'xsinx': i * np.sin(i / r),
                                            'xcosx': i * np.cos(i / r),
                                            'tanx': np.tan(i / r)},
                           global_step=i)
        writer.add_scalars('scalars1/P2',
                           {'xsinx': i * np.sin(i / (2 * r)),
                            'xcosx': i * np.cos(i / (2 * r)),
                            'tanx': np.tan(i / (2 * r))}, i)
        writer.add_scalars(main_tag='scalars2/Q1',
                           tag_scalar_dict={'xsinx': i * np.sin((2 * i) / r),
                                            'xcosx': i * np.cos((2 * i) / r),
                                            'tanx': np.tan((2 * i) / r)},
                           global_step=i)
        writer.add_scalars('scalars2/Q2',
                           {'xsinx': i * np.sin(i / (0.5 * r)),
                            'xcosx': i * np.cos(i / (0.5 * r)),
                            'tanx': np.tan(i / (0.5 * r))}, i)


def add_histogram(writer):
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers/p1', x + i, i)
        writer.add_histogram('distribution centers/p2', x + i * 2, i)


if __name__ == '__main__':
    writer = SummaryWriter(log_dir="logs/test", flush_secs=120)
    # for n_iter in range(1000):
    #     writer.add_scalar(tag='Loss/train',
    #                       scalar_value=np.random.random(),
    #                       global_step=n_iter)
    #     writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     time.sleep(0.01)

    add_scalars(writer)
    writer.close()

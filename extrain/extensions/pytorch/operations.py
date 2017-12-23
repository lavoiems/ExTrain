import torch
import time
import os

MODEL_FILE_NAME = 'model.pk'


class Snapshot(object):
    def __init__(self, path):
        self.path = os.join(path, 'snapshots')
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def __call__(self, model, **kwargs):
        snap_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        file_path = os.join(self.path, snap_time)
        model.save(model.state_dict(), file_path)


class Logging(object):
    def __init__(self, path):

def restore(model, path='./', **kwargs):
    model.load_state_dict(torch.load(path + MODEL_FILE_NAME))
    return model

import time
import os


class Snapshot(object):
    def __init__(self, path):
        self.path = os.join(path, 'snapshots')
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def __call__(self, model, **kwargs):
        snap_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        file_path = os.join(self.path, snap_time)
        model.save(model.state_dict(), file_path)


class LogResult(object):
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, epoch, results):
        self.logger.info('Epoch: %s. Results: %s' % (epoch, results))


class VisualizeSamples(object):
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def __call__(self, ):
        raise NotImplemented('visualizer not implemented yet')


class VisualizeTsne(object):
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def __call__(self, ):
        raise NotImplemented('visualizer not implemented yet')


class VisualizePlot(object):
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def __call__(self, ):
        raise NotImplemented('visualizer not implemented yet')

import signal
from functools import partial


def interupt_handler(interupt_fn, information, signums, frame):
    for fn in interupt_fn:
        fn(**information)


def post_epoch(interupt_fn, post_epoch_fn, checkpoint, epoch, information):
    signal.signal(signal.SIGINT, partial(interupt_handler, interupt_fn, information))
    if epoch % checkpoint == 0:
        for fn in post_epoch_fn:
            fn(**information)


def train(data, train_fn, batch_fn, model, max_epoch, batch_size, n_epoch=1, interupt_fn=[], post_epoch_fn=[],
          restore_fn=None, other=None):
    losses = []
    if restore_fn:
        model = restore_fn(model, **other)
    for epoch in range(max_epoch):
        for batch in batch_fn(batch_size, data):
            losses += train_fn(batch, model)
            information = dict(losses=losses, epoch=epoch, model=model, data=data, **other)
        if (epoch % n_epoch) == 0:
            post_epoch(interupt_fn, post_epoch_fn, n_epoch, epoch, information)
    return losses




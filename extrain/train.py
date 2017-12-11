
def train(data, train_fn, batch_fn, networks, max_epoch, batch_size, checkpoint=1):
    all_losses = []
    for epoch in range(max_epoch):
        for batch in batch_fn(batch_size, data):
            losses = train_fn(*(batch + networks))
            all_losses.append(losses)
        if (epoch % checkpoint) == 0:
            pass
    return all_losses




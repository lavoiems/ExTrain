import torch

MODEL_FILE_NAME = 'model.pk'


def save_model(model, path='./', **kwargs):
    torch.save(model.state_dict(), path + MODEL_FILE_NAME)


def restore(model, path='./', **kwargs):
    model.load_state_dict(torch.load(path + MODEL_FILE_NAME))
    return model

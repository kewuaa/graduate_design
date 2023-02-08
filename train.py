import torch

from project import model


if __name__ == "__main__":
    net = model.Automap()
    try:
        net.start_train()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        net.use_checkpoint()
        net.start_train()

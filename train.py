import torch

from project import model


if __name__ == "__main__":
    net = model.Automap(0.5)
    # net = model.UNet(3)
    try:
        net.start_train()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        net.use_checkpoint()
        net.start_train()

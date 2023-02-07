# import torch

# from project import model
from project import config as train_config
config = train_config.load()


def train():
    # net = model.Automap(projection_num, img_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.to(device)


if __name__ == "__main__":
    print(config)

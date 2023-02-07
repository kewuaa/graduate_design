from torch import optim
from torch import nn
import torch

from project import data
from project import model
from project import config as train_config
from project.logging import logger
img_num = 100
img_size = 140
projection_num = 140
max_circle_num = 3
min_circle_size = 10
max_circle_size = 30


def train_model(model, device, save_checkpoint: bool = True) -> None:
    config = train_config.load()
    epoch_num = config['epoch_num']
    # 1. Create dataset

    # 2. Split into train / validation partitions

    # 3. Create data loaders

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    # and the loss scaling for AMP
    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=config['learning_rate'],
    #     weight_decay=config['weight_decay'],
    #     momentum=momentum,
    #     foreach=True
    # )

    for epoch in range(epoch_num):
        pass


if __name__ == "__main__":
    # logger.info('init data......')
    # data.init(
    #     img_num,
    #     img_size,
    #     max_circle_num,
    #     min_circle_size,
    #     max_circle_size,
    #     180 / projection_num,
    # )
    # logger.info('data successfully init at "./data"')
    config = train_config.load()
    print(config)
    # net = model.Automap(projection_num, img_size)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net.to(device)

from pathlib import Path

from matplotlib import pyplot as plt
from skimage.transform import iradon
import numpy as np
import torch
import cv2

# from project import model
image_dir = Path('./data/transformed_imgs')
label_dir = Path('./data/imgs')


if __name__ == "__main__":
    # net = model.Automap()
    # net.load('./checkpoints/checkpoint_epoch_5.pth')
    i = 3
    img_path = image_dir / f'{i}.png'
    label_path = label_dir / f'{i}.png'
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    label = cv2.normalize(label, None, -0.5, 0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    img = torch.Tensor(np.expand_dims(img, axis=0))
    label = torch.Tensor(np.expand_dims(label, axis=0))
    # pre = net(img)
    plt.imshow(np.squeeze(img.detach().numpy()), cmap='gray')
    plt.show()
    print(img, label, sep='\n')

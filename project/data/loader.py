from pathlib import Path

from torch.utils import data
import torch
import cv2


class Dataset(data.Dataset):
    def __init__(self, images_dir: str, labels_dir: str, scale: float = 1.0):
        self.scale = scale
        self.images_dir = images_dir = Path(images_dir)
        self.labels_dir = labels_dir = Path(labels_dir)
        if not (images_dir.exists() and labels_dir.exists()):
            raise RuntimeWarning('check you dir path')

    def __getitem__(self, index):
        pattern = f'{index}.*'
        image_path = self.images_dir.glob(pattern)
        label_path = self.labels_dir.glob(pattern)
        if not (image_path and label_path):
            raise RuntimeWarning('check index')
        image = cv2.imread(str(image_path[0]), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path[0]), cv2.IMREAD_GRAYSCALE)
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(label.copy()).float().contiguous(),
        }

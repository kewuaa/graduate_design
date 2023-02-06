from functools import partial
from pathlib import Path
import asyncio
import json

import cv2
import numpy as np
current_path = Path(__file__).parent
source_path = current_path / 'imgs'
if not source_path.exists():
    raise RuntimeWarning('source images directory not exists')
target_path = current_path / 'transformed_imgs'
target_path.mkdir(exist_ok=True)
with open(current_path / 'setting.json') as f:
    setting = json.load(f)
img_size = setting['image_size']
img_num = setting['image_num']
radon = partial(
    cv2.ximgproc.RadonTransform,
    theta=180 / img_size,
    crop=True,
    start_angle=0,
    end_angle=180,
    norm=True
)


# def create_mask():
#     x = y = np.arange(140)
#     X, Y = np.meshgrid(x, y)
#     distance = (X - 70) ** 2 + (Y - 70) ** 2
#     mask = distance <= 70 ** 2
#     return mask


# mask = create_mask()


async def _transform(name):
    loop = asyncio.get_event_loop()
    image = await loop.run_in_executor(
        None,
        cv2.imread,
        str(source_path / name),
        cv2.IMREAD_GRAYSCALE
    )
    # image = np.where(mask, image, 0)
    sinogram = radon(image)
    await loop.run_in_executor(
        None,
        cv2.imwrite,
        str(target_path / name),
        sinogram
    )


def transform():
    async def transform():
        for i in range(0, img_num, 10):
            tasks = [
                asyncio.create_task(_transform(f'{i + j + 1}.png'))
                for j in range(10)
            ]
            for task in tasks:
                await task
    asyncio.get_event_loop().run_until_complete(transform())


def test():
    img = cv2.imread(str(source_path / '96.png'), cv2.IMREAD_GRAYSCALE)
    # img = np.where(mask, img, 0)
    print(np.unique(img))
    sinogram = radon(img)
    cv2.imshow('origin', img)
    cv2.imshow('radon image', sinogram)
    cv2.waitKey(0)


if __name__ == "__main__":
    # transform()
    test()

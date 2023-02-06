from functools import partial
from pathlib import Path
import random
import asyncio
import json

from PIL import Image, ImageDraw

from .cython_lib import circle
current_path = Path(__file__).parent
with open(current_path / '../setting.json') as f:
    setting = json.load(f)
img_num = setting['image_num']
img_size = setting['image_size']
min_circle_size = setting['min_circle_size']
max_circle_size = setting['max_circle_size']
max_circle_num = setting['max_circle_num']
# pixel_option = setting['pixel_option']
pixel_option = list(range(0, 128, 10))


async def main() -> None:
    async def generate_one(index: int) -> None:
        circles = await loop.run_in_executor(
            None,
            Circle.generate,
            random.randint(1, max_circle_num),
        )
        img = Image.new('L', (img_size, img_size), 255)
        draw = ImageDraw.Draw(img)
        for c in circles:
            alpha = random.choice(pixel_option)
            left_top, right_bottom = c
            draw.ellipse(
                (left_top, right_bottom),
                fill=alpha,
                outline=alpha,
            )
        await loop.run_in_executor(
            None,
            partial(
                img.save,
                str(img_save_path / f'{index + 1}.png'),
                dpi=(300, 300),
            )
        )

    Circle = circle.Circle(
        0.,
        img_size,
        0.,
        img_size,
        min_circle_size,
        max_circle_size
    )
    img_save_path = current_path / '../imgs'
    img_save_path.mkdir(exist_ok=True)
    batch_size = 10
    loop = asyncio.get_event_loop()
    for i in range(0, img_num, batch_size):
        tasks = [
            loop.create_task(generate_one(i + j))
            for j in range(batch_size)
        ]
        for task in tasks:
            await task


def run():
    asyncio.get_event_loop().run_until_complete(main())


if __name__ == "__main__":
    run()

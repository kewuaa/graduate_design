from functools import partial
from pathlib import Path
import random
import asyncio

from PIL import Image, ImageDraw

from cython_lib import circle
IMG_NUM = 100
IMG_SIZE = 140
MIN_SIZE = 10
MAX_SIZE = 30
MAX_CIRCLE_NUM = 3
current_path = Path(__file__).parent


async def main() -> None:
    async def generate_one(index: int) -> None:
        circles = await loop.run_in_executor(
            None,
            Circle.generate,
            random.randint(1, MAX_CIRCLE_NUM),
        )
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 255)
        draw = ImageDraw.Draw(img)
        for c in circles:
            alpha = random.randint(1, 130)
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

    Circle = circle.Circle(0., IMG_SIZE, 0., IMG_SIZE, MIN_SIZE, MAX_SIZE)
    img_save_path = current_path / '../img1s'
    img_save_path.mkdir(exist_ok=True)
    batch_size = 10
    loop = asyncio.get_event_loop()
    for i in range(0, IMG_NUM, batch_size):
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

from functools import partial
from pathlib import Path
import random
import asyncio

# from matplotlib import pyplot as plt
# from matplotlib import patches
from PIL import Image, ImageDraw

from cython_lib import circle as Circle
IMG_NUM = 10000
IMG_SIZE = 140
MAX_CIRCLE_NUM = 3
current_path = Path(__file__).parent


async def main() -> None:
    async def generate_one(index: int) -> None:
        # axis: plt.Axes = fig.add_axes([0, 0, 1, 1])
        # axis.axis([-half_size, half_size, -half_size, half_size])
        # axis.set_aspect(1)
        # axis.axis('off')
        # circles = await loop.run_in_executor(
        #     None,
        #     Circle.generate,
        #     random.randint(1, MAX_CIRCLE_NUM),
        # )
        # for circle in circles:
        #     alpha = random.random() / 2 + 0.5
        #     patch = patches.Circle((circle[0], circle[1]), circle[2])
        #     patch.set_color('black')
        #     patch.set_alpha(alpha)
        #     axis.add_patch(patch)
        # plt.savefig(str(img_save_path / f'{index + 1}.png'))
        # axis.clear()
        circles = await loop.run_in_executor(
            None,
            Circle.generate,
            random.randint(1, MAX_CIRCLE_NUM),
        )
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), 255)
        draw = ImageDraw.Draw(img)
        for circle in circles:
            alpha = random.randint(1, 130)
            left_top, right_bottom = circle
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

    # half_size = IMG_SIZE / 2
    # fig: plt.Figure = plt.figure(figsize=(IMG_SIZE / 100,) * 2)

    img_save_path = current_path / '../imgs'
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

import asyncio

from .generator import Generator
from .transformer import Transformer


def init(
    img_num: int,
    img_size: int,
    max_circle_num: int,
    min_circle_size: int,
    max_circle_size: int,
    theta_step: float,
) -> None:
    async def main():
        gtask = loop.create_task(generator.generate())
        ttask = loop.create_task(transformer.transform())
        await gtask
        await ttask
    generator = Generator(
        img_num,
        img_size,
        max_circle_num,
        min_circle_size,
        max_circle_size
    )
    transformer = Transformer(img_num, theta_step)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

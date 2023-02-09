from functools import partial
import asyncio

from rich.progress import Progress

from .generator import Generator
from .transformer import Transformer


def init(
    img_num: int,
    img_size: int,
    min_circle_num: int,
    max_circle_num: int,
    min_circle_size: int,
    max_circle_size: int,
    theta_step: float,
    start_angle: int,
    end_angle: int,
) -> None:
    async def main():
        progress = Progress()
        generate_task = progress.add_task(
            '[blue]generating',
            total=img_num
        )
        transform_task = progress.add_task(
            '[yellow]radon transformimg',
            total=img_num
        )
        with progress:
            gtask = loop.create_task(generator.generate(
                refresh=partial(progress.update, generate_task, advance=1)
            ))
            ttask = loop.create_task(transformer.transform(
                refresh=partial(progress.update, transform_task, advance=1)
            ))
            await gtask
            await ttask
    generator = Generator(
        img_num,
        img_size,
        min_circle_num,
        max_circle_num,
        min_circle_size,
        max_circle_size,
    )
    transformer = Transformer(
        img_num,
        theta_step,
        start_angle,
        end_angle,
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

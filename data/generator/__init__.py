import asyncio

from .main import main


def generate():
    asyncio.get_event_loop().run_until_complete(main())

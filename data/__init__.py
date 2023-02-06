def init():
    from . import generator
    generator.generate()
    from . import transformer
    transformer.transform()

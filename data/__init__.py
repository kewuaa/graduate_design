def init():
    from . import generator
    from . import transformer
    generator.generate()
    transformer.transform()

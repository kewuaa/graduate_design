try:
    from . import circle
except ImportError:
    import pyximport
    pyximport.install(language_level=3)
    from . import circle

try:
    from ._version import version as __version__
except (ImportError, ModuleNotFoundError, AttributeError):
    __version__ = "unknown"

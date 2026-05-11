try:
    from ._version import version as __version__
except (ImportError, ModuleNotFoundError, AttributeError):
    __version__ = "unknown"

from ._reader import napari_get_reader

__all__ = ("__version__", "napari_get_reader")

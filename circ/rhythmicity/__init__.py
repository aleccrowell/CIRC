from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("circ")
except PackageNotFoundError:
    __version__ = "unknown"

import re
import numpy as np
from pathlib import Path
import tifffile

from typing import Iterable
from numpy.typing import NDArray


__all__ = ["str2dict", "find_files"]


_PERPENDICULAR: np.array = np.array([-1, 1])
_COMPLEX: np.array = np.array([1j, 1])


def extract_metadata_imagej(path: Path) -> dict:
    with tifffile.TiffFile(path) as image:
        if not image.is_imagej:
            return {}
        else:
            meta = image.imagej_metadata

    meta.pop("ImageJ", None)
    meta.pop("spacing", None)
    meta.pop("Info", None)
    meta.pop("Labels", None)
    meta.pop("LUTs", None)
    meta.pop("Ranges", None)
    meta.pop("loop")

    meta["channel_shows"] = meta["channels"] * [None]
    meta["staining"] = meta["channels"] * [None]

    return meta


def str2dict(x: str, sep: str = "_", ksep: str = "-") -> dict[str, str]:
    """Dissemble a string into a dictionary.

    Parameters
    ----------
    x : str
        A string that shall be turned into a dictionary by predefined
        symbols.
    sep : str, optional
        A single character or a string at which `x` shall be split. Each
        element of this splitting will result in a key-value pair in the
        final dict.
    ksep : str, optional
        A single character or a string. Each element from splitting `x` by
        `sep` will be split again at `ksep`. Splitting will only occur at
        the
        first occurrence of `ksep`. The first will be the key, the second
        will
        be the value.

    Returns
    -------
    dict
        A dict containing key-value pairs by splitting the original string
        into pieces.

    Raises
    ------
    ValueError
        If any of the strings resulting from splitting `x` by `sep` does not
        contain `ksep`. In this case, no key-value pair can be built.

    Examples
    --------
    >>> str2dict("key1-value1_key2-value2_key3-value3", sep="_", ksep="-")
    {"key1": "value1", "key2": "value2", "key3": "value3"}
    """

    temp = [item.split(ksep, 1) for item in x.split(sep)]
    return {key.capitalize(): item.capitalize() for key, item in temp}


def find_files(
    path: str | Path,
    pattern: str | re.Pattern[str] = ".*",
    ignore: str | re.Pattern[str] = "^_.*",
    recursive: bool = True,
) -> Iterable[Path]:
    """
    Find all files that match a name pattern.

    Parameters
    ----------
    path: str | Path
        A starting path to look for files.
    pattern: str | re.Pattern[str], optional
        A regular expression that a filename must match to be returned. This
        can be either string or a compiled pattern. Default is '.*',
        which matches any filename.
    ignore: str | re.Pattern[str], optional
        A regular expression defining which folder (and subsequently
        subfolders) shall be ignored. By default, folders starting with an
        underscore are ignored during search.
    recursive: bool, optional
        A flag indicating if subfolders shall be returned.

    Yields
    -------
    Path
        Yields Path objects to the files matching the pattern.
    """

    if isinstance(path, str):
        path = Path(path)

    if isinstance(ignore, str):
        ignore = re.compile(ignore)

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    for p in path.iterdir():
        if p.is_dir() and not ignore.match(p.name) and recursive:
            yield from find_files(p, pattern=pattern, recursive=recursive)
        elif p.is_file() and pattern.match(p.name):
            yield p


def perpendicular(arr: NDArray) -> NDArray:
    """Calculate a perpendicular arr.

    Parameters
    ----------
    arr : NDArray
        The arr.

    Returns
    -------
    NDArray
        Array perpendicular to arr with length 1.

    Examples
    --------
    >>> perpendicular(np.array([0, 1]))
    array([-1.,  0.])
    >>> perpendicular(np.array([1, 1]))
    array([-0.70710678,  0.70710678])
    """

    perp = arr[::-1] * _PERPENDICULAR
    return perp / np.linalg.norm(perp)


def orthogonal_line(arr0: NDArray, arr1: NDArray, length: float = 1.0) -> tuple:
    """Find orthogonal arrays (vectors) of specific length.

    Parameters
    ----------
    arr0 : array
        Array that points to the first endpoint of the line of which the
        orthogonal shall be determined.
    arr1 : array
        Array that points to the second endpoint of the line of which the
        orthogonal shall be determined.
    length : float = 1.
        Half distance between the mask vectors. Euclidean distance between
        each mask point and the line defined by arr0 and arr1.

    Returns
    -------
    tuple of arrays
        A tuple with two arrays. Each points to one of the endpoints of the
        orthogonal.

    Examples
    --------
    >>> orthogonal_line(np.array([0, 0]), np.array([10, 10]))
    (array([4, 5]), array([5, 4]))
    >>> orthogonal_line(np.array([0, 0]), np.array([10, 10]), length=2)
    (array([3, 6]), array([6, 3]))
    >>> orthogonal_line(np.array([0, 2]), np.array([10, 12]), length=2)
    (array([3, 8]), array([6, 5]))
    """

    mid = (arr0 + arr1) / 2
    orth = perpendicular(arr1 - arr0) * length

    return (mid + orth).astype(int), (mid - orth).astype(int)

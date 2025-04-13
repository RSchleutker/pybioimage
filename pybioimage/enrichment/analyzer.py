from pybioimage.utils import find_files
from pybioimage.utils import str2dict
from pybioimage.utils import orthogonal_line

import re
from pathlib import Path
from math import ceil
from scipy import ndimage as ndi

import numpy as np
from skimage import io
from skimage import color
from skimage import filters
from skimage import draw
from skimage.util import img_as_ubyte
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from skimage.morphology import disk

import pandas as pd
from numpy.typing import NDArray
from typing import Optional


__all__ = ["Analyzer", "Cell"]


class Analyzer:

    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = io.imread(self.path)
        self.meta = str2dict(self.path.parent.name)

    def __repr__(self):
        return f"Analyzer(path={self.path}, ...)"

    def cells(self, pattern: Optional[str | re.Pattern[str]] = None) -> list["Cell"]:

        if pattern is None:
            pattern = ".*\\.csv$"

        cells = []
        for path in find_files(self.path, pattern):
            cells.append(Cell(path, image=self.image))

        return cells

    def analyze(
        self,
        pattern: Optional[str | re.Pattern[str]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Analyze enrichment of cells.

        Analyzes the attached region by measuring enrichment of all attached cells.

        Parameters
        ----------
        pattern : str | re.Pattern[str], optional
            A pattern to define the cells to be analyzed. This is passed down to the `cells()`
            method and thus ultimately to the `find_files` function. By default, all CSV files in
            the same folder are interpreted as cells.
        kwargs
            Further keyworded arguments. These are passed down to the `analyze()` method of the
            `Cell` instance.

        Returns
        -------
        Optional[pd.DataFrame]
            Either `None` if no cells were found, or a dataframe containing cell enrichment
            values. This dataframe also includes the metadata of the cells as well as of this
            instance.
        """

        measurements = []
        for cell in self.cells(pattern=pattern):
            measurement = cell.measure(**kwargs)
            if measurement is not None:
                measurements.append(measurement)

        # If no cells or no measurements.
        if not measurements:
            return None

        # Otherwise, combine into one table.
        measurements = pd.concat(measurements)
        for key, value in self.meta.items():
            measurements[key] = value

        return measurements


class Cell:

    _COMPLEX: NDArray = np.array([1j, 1])

    def __init__(self, path: Path, image: NDArray) -> None:
        self.path = path

        df = pd.read_csv(self.path)
        self.vertices = df[["Y", "X"]].round(0).to_numpy(dtype=np.int_)
        self.z_pos = df["Slice"].iloc[0] - 1

        self.anchors = None
        self.image = image[self.z_pos]
        self.image_smoothed = filters.gaussian(self.image, sigma=1)
        self.meta = str2dict(self.path.name)
        self.measurements = None

        self._mask_vertices = None
        self._mask_bicellular = None
        self._mask_cytoplasm = None

    def __repr__(self):
        return f"Cell(path={self.path}, ...)"

    @property
    def visualize(self) -> NDArray[np.ubyte]:
        labels = np.zeros_like(self.image, dtype=np.uint8)
        labels[self._mask_vertices] = 1
        labels[self._mask_bicellular] = 2
        labels[self._mask_cytoplasm] = 3

        visualization = color.label2rgb(labels, self.image)
        return img_as_ubyte(visualization)

    def find_anchors(self, max_length: float = 8.0, iterations: int = 0) -> None:
        n, *_ = self.vertices.shape
        m = n * 2**iterations  # Number of new anchor points.
        mask = np.ones_like(self.image, dtype="bool")

        # Create output array with existing anchors at correct positions.
        anchors = np.empty((m, 2), dtype="int")
        for i in range(n):
            j = i * 2**iterations
            anchors[j] = self.vertices[i]

        # In each iteration add one anchor between two existing anchors.
        for it in range(iterations):
            k = n * 2**it  # Number of new anchors in this iteration.

            # Find all indices of already filled anchor points. Shifting these indices by -1
            # creates pairs of anchors in between of which a new anchor is found.
            indices = np.array(j * 2 ** (iterations - it) for j in range(k))
            shifted = np.roll(indices, -1)

            for i1, i2 in zip(indices, shifted):
                anchor1 = anchors[i1]
                anchor2 = anchors[i2]
                distance = np.linalg.norm(anchor1 - anchor2)
                length = min(max_length, distance / 3)
                new_anchor = self._find_anchor(anchor1, anchor2, length, mask)

                new_index = i1 + 2 ** (iterations - it - 1)
                anchors[new_index] = new_anchor

        self.anchors = anchors

    def _find_anchor(
        self, anchor1: NDArray, anchor2: NDArray, length: float, mask: NDArray[np.bool_]
    ) -> NDArray:
        r, c = orthogonal_line(anchor1, anchor2, length)
        rows, cols = draw.line(r[0], r[1], c[0], c[1])
        mask[rows, cols] = False
        image_masked = np.ma.array(self.image_smoothed, mask=mask)
        arg = np.unravel_index(np.argmax(image_masked), image_masked.shape)

        if arg == (np.int64(0), np.int64(0)):
            return (anchor1 + anchor2).mean().astype(int)
        else:
            return np.array(arg, dtype=int)

    def measure(self, width: int = 2, **kwargs) -> pd.DataFrame:
        self.find_anchors(**kwargs)

        # Define mask for cell vertices.
        self._mask_vertices = np.zeros_like(self.image, dtype=np.bool_)
        self._mask_vertices[tuple(self.vertices.T)] = True
        binary_dilation(self._mask_vertices, disk(width), out=self._mask_vertices)

        # Define mask for bicellular region.
        self._mask_bicellular = np.zeros_like(self.image, dtype=np.bool_)
        rows, cols = tuple(self.anchors.T)
        half_width = ceil(width / 2)
        self._mask_bicellular[draw.polygon_perimeter(rows, cols)] = True
        binary_dilation(self._mask_bicellular, disk(half_width), out=self._mask_bicellular)
        self._mask_bicellular &= self._mask_vertices

        # Define mask for cytoplasm (background).
        self._mask_cytoplasm = np.zeros_like(self.image, dtype=np.bool_)
        ndi.binary_fill_holes(
            self._mask_vertices | self._mask_bicellular, output=self._mask_cytoplasm
        )
        binary_erosion(self._mask_cytoplasm, disk(width + 5), out=self._mask_cytoplasm)

        int_vertices = self.image[self._mask_vertices].mean()
        int_bicellular = self.image[self._mask_bicellular].mean()
        int_cytoplasm = self.image[self._mask_cytoplasm].mean()
        enrichment = (int_vertices - int_cytoplasm) / (int_bicellular - int_cytoplasm)

        measurements = {
            "Vertices": [int_vertices],
            "Bicellular": [int_bicellular],
            "Background": [int_cytoplasm],
            "Enrichment": [enrichment],
            **self.meta,
        }
        self.measurements = pd.DataFrame(measurements)

        return self.measurements

    def _sort_vertices(self) -> None:
        centered = self.vertices - self.vertices.mean(axis=0)
        compl = np.matmul(centered, self._COMPLEX)
        order = np.argsort(np.angle(compl))
        self.vertices[...] = self.vertices[order, :]

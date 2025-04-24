import time

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from pybioimage.utils import str2dict
from .image import peak_prominence2

from skimage import io
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import disk
from skimage.color import label2rgb
from pykuwahara import kuwahara

from pathlib import Path
from functools import cached_property

from numpy.typing import NDArray
import logging


__all__ = ["Analyzer"]

logger = logging.getLogger(__name__)


class Analyzer:
    """Analyzer class for cell aggregation assays.

    This class takes an image and analyzes how many cells fall into the same
    cluster.

    Parameters
    ----------
    path : Path
        A path pointing to the image that should be analyzed. The image is
        supposed to have two channels. The first shows the nuclei (e.g. by DAPI
        staining) and the second shows transfected cells.

    Attributes
    ----------
    labeled_clusters
    transfected_cells
    image_cells : NDArray
        The channel that shows the transfected cells. This image is already
        processed when creating the class instance.
    image_nuclei : NDArray
        The channel that shows the cell nuclei. This image is already processed
        when creating the class instance.
    peaks : None | NDArray[np.int_]
        An Nx2 array giving the coordinates of all peaks. Note that this is None
        if the `analyze()` method was not yet called.
    measurements : None | pd.DataFrame
        A dataframe holding the number of cells for each identified cluster.
        Note that this is None if the `analyze()` method was not yet called.
    """

    def __init__(self, path: Path):
        self.path = path
        self.metadata = self._extract_metadata()
        logger.info("Initializing %s.", self)

        self.image = self._read_image()
        self.mask = self._read_mask()

        self.peaks = None
        self.measurements = None
        self._last_args = {}

    def __repr__(self):
        return f"Analyzer({self.path.name})"

    def _extract_metadata(self) -> dict[str, str]:
        return {"Scan": self.path.parent.name[-1]} | str2dict(self.path.parent.parent.name)

    def analyze(self, **kwargs) -> pd.DataFrame:
        """Analyze the cell aggregation image.

        Parameters
        ----------
        kwargs
            Further arguments passed down to the `peak_prominence2()` function.

        Returns
        -------
        pd.DataFrame
            A dataframe holding the identified number of cells per cluster. This
            dataframe can also later be accessed through the `measurements`
            attribute.
        """

        logger.info("Starting analysis.")
        start_time = time.perf_counter()

        if self.peaks is None or kwargs != self._last_args:
            self.peaks = peak_prominence2(self.image_nuclei, labels=self.labeled_clusters, **kwargs)
            self._last_args = kwargs

        cluster, cluster_size = np.unique(self.peaks, return_counts=True)
        df = {"cluster": cluster[1:], "cells": cluster_size[1:], **self.metadata}

        self.measurements = pd.DataFrame(df)

        end_time = time.perf_counter()

        logger.info(
            "Finished analysis. Found %4d cells in %4d clusters in %.3f sec.",
            sum(cluster_size[1:]),
            len(cluster_size[1:]),
            end_time - start_time,
        )

        return self.measurements

    def visualize_segmentation(self) -> NDArray[np.uint8]:
        """Highlight transfected cells and identified nuclei.

        Returns
        -------
        NDArray[np.uint8]
            An 8-bit RGB image of the nuclei channel with identified nuclei and
            transfected cells highlighted. Can be used to check sanity of
            segmentation.
        """

        if self.peaks is None:
            raise AttributeError("No peaks. Run analyzer.analyse() first.")

        visualization = label2rgb(self.labeled_clusters, self.image_nuclei, alpha=0.2)

        # _footprint = footprint_rectangle((3, 3), decomposition="separable")
        # binary_dilation(img_peaks, footprint=_footprint, out=img_peaks)
        rows, cols = np.nonzero(self.peaks)
        visualization[rows, cols] = np.array([1, 0, 0])

        return img_as_ubyte(visualization)

    def _read_image(self) -> NDArray[np.uint8]:
        start_slice, end_slice = self.metadata.get("projection_slices", [None, None])
        image = io.imread(self.path)
        # image = image[start_slice:end_slice].max(axis=0)
        shape = image.shape

        # Some edit...

        logger.info("Reading image with shape %s.", image.shape)

        # Images with 3 channels have a different axes order. Put the channel axis to front.
        if image.shape[-1] <= 3:
            image = np.moveaxis(image, -1, 0)
            logger.debug("Rearranged shape from %s to %s.", shape, image.shape)

        # Apply mask by setting all pixels outside the mask to 0.
        # image[:, ~self.mask] = 0

        if self.metadata["Exp"] == "Extracellular":
            image[[1, 2]] = image[[2, 1]]

        return image

    def _read_mask(self) -> NDArray[np.bool_]:
        # stem = self.path.stem + "_mask"
        path = self.path.with_stem("mask")

        if not path.exists():
            return np.ones_like(self.image[0], dtype=np.bool_)
        else:
            return io.imread(path).astype(np.bool_)

    @cached_property
    def image_nuclei(self) -> NDArray[np.uint8]:
        kernel = disk(radius=25)
        cv2.GaussianBlur(self.image[0], (0, 0), sigmaX=2.0, sigmaY=2.0, dst=self.image[0])
        cv2.morphologyEx(self.image[0], op=cv2.MORPH_TOPHAT, kernel=kernel, dst=self.image[0])
        self.image[0, ~self.mask] = 0

        return self.image[0]

    @cached_property
    def image_cells(self) -> NDArray[np.uint8]:
        cv2.GaussianBlur(self.image[1], (0, 0), sigmaX=4.0, sigmaY=4.0, dst=self.image[1])
        self.image[1] = kuwahara(self.image[1])
        self.image[1, ~self.mask] = 0

        return self.image[1]

    @cached_property
    def labeled_clusters(self) -> NDArray[np.uint16]:
        _, image = cv2.connectedComponents(self.transfected_cells, connectivity=8, ltype=cv2.CV_16U)

        return image

    @cached_property
    def transfected_cells(self) -> NDArray[np.uint8]:
        hi = threshold_otsu(self.image_cells)
        lo = threshold_triangle(self.image_cells)
        image = apply_hysteresis_threshold(self.image_cells, low=lo, high=hi).astype(np.uint8)

        kernel = disk(2)

        cv2.dilate(image, kernel, dst=image)
        ndi.binary_fill_holes(image, output=image)
        cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, dst=image)

        return image

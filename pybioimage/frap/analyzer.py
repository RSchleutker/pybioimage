from pybioimage.utils import find_files
from pybioimage.utils import str2dict

import re
import logging
import numpy as np
import pandas as pd

from skimage import io
from skimage import draw
from pathlib import Path
from functools import cached_property

from typing import Optional
from numpy.typing import NDArray


__all__ = ["Analyzer", "Region"]


# User logger to log messages.
logger = logging.getLogger(__name__)


class Analyzer:
    """Analyzer class for FRAP data.

    This class takes a movie and allows analysis of different bleached regions.

    Parameters
    ----------
    path : Path
        A path pointing to the movie file. Coordinate files for different regions are assumed to
        be located in the same folder as the movie file.
    prebleach_frames : int
        The number of prebleach frames in the movie.
    interval : float, optional
        The intervall time of the movie. The unit (seconds, minutes) is arbitrary. The frame
        number will be multiplied by this number. Default is 1.0.

    Attributes
    ----------
    path : Path
        The path of the movie file.
    movie : NDArray
        The FRAP movie.
    meta : dict[str, str]
        Metadata inferred from the filename of the movie. The dict is build using the `str2dict()`
        function from the `utils` module.
    """

    def __init__(self, path: Path, prebleach_frames: int, interval: float = 1.0):
        self.path = path
        self.movie = io.imread(self.path)
        self.prebleach_frames = prebleach_frames
        self.interval = interval
        self.metadata = self._extract_metadata()

    def __repr__(self):
        return f"Analyzer(path='{self.path}', ...)"

    def _extract_metadata(self) -> dict[str, str]:
        return str2dict(self.path.parent.name)

    def regions(self, pattern: Optional[str | re.Pattern[str]] = None) -> list["Region"]:
        """Get a list of bleach regions.

        Parameters
        ----------
        pattern : str | re.Pattern[str], optional
            Define a pattern to find suitable files. By default, all CSV
            files in the same folder as the movie are assumed to define
            regions and are tried to be read as such.

        Returns
        -------
        list[Region]
            A list of regions for further analysis.

        Notes
        -----
        This method is rarely called directly by the user. Instead,
        users should call the `analyze()` method, which internally calls the
        `regions()` method and adds automatically metadata to the results.
        """

        if pattern is None:
            pattern = ".*\\.csv$"

        regions = []

        for file in find_files(self.path.parent, pattern, recursive=False):
            regions.append(Region(file, self))

        return regions

    def analyze(
        self,
        pattern: Optional[str | re.Pattern] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Analyze the FRAP movie.

        Analyzes the FRAP movie by measuring intensities in all defined regions.

        Parameters
        ----------
        pattern : str | re.Pattern[str], optional
            A pattern to define the regions to be analyzed. This is passed down to the `regions()`
            method. By default, all CSV files in the same folder are interpreted as regions.
        kwargs
            Further keyworded arguments. These will be passed down to the `measure()` method of
            the Region instance. Currently, this only allows to adjust the radius of the measured
            region.

        Returns
        -------
        Optional[pd.DataFrame]
            A dataframe containing the measured intensities for each region attached to this
            movie or None if no regions were found. The dataframe also contains the metadata from
            the `meta` attribute of this class as well as of the Region class.
        """

        logger.info(f"Running {self}.")

        measurements = []
        regions = self.regions(pattern=pattern)

        logger.info(f"Detected {len(regions)} regions with pattern `{pattern}`.")

        for region in regions:
            logger.info(f"Analyzing {region}.")
            region.measure(**kwargs)
            region.normalize_measurement(prebleach_frames=self.prebleach_frames)
            measurements.append(region.measurements)

        # If no regions were found.
        if not measurements:
            return None

        # Combine data from all regions into one table and add metadata.
        measurements = pd.concat(measurements)
        for key, value in self.meta.items():
            measurements[key] = value

        return measurements


class Region:
    """A class for single bleach regions.

    Parameters
    ----------
    path : Path
    movie : NDArray

    Attributes
    ----------
    path : Path
        The path of the CSV file used to create the region.
    movie : NDArray
        The FRAP movie.
    trajectory : NDArray[np.int_]
        A Nx3 array representing the position of the bleach region. The
        columns represent the frame, y (row), and x (column) position of the
        region. This trajectory is used to generate the registered movie.
    meta : dict[str, str]
        Metadata for this region. This is constructed from the filename using
        the `str2dict()` function from the `utils` module.
    position : NDArray[np.int_]
        The position of the tracjectory for the first frame. This is the
        position in the registered movie at which the bleach region is kept
        at a constant place.
    registered : NDArray
        A movie the same dtype and shape as `movie`. The movie has been
        registered such that the bleach region is kept at a constant position.
    measurements : pd.DataFrame
        The measured intensities from the movie in a dataframe with metadata
        and columns for corrected and normalized values.
    """

    def __init__(self, path: Path, analyzer: Analyzer):
        self.path = path
        self.analyzer = analyzer
        self.trajectory: NDArray[np.int_] = self._read_csv()

        self.measurements = None

        self.meta = str2dict(self.path.stem)

    def __repr__(self) -> str:
        return f"Region(path='{self.path}', ...)"

    @cached_property
    def registered(self) -> NDArray[np.int_]:
        """The movie registered for this region.

        A movie of the same shape and dtype as the original movie. Each frame
        has been shifted such that this region is kept at a constant position.

        Notes
        -----
        Each bleach region in a FRAP movie might shift differently over the
        course of the experiment. Therefor, the movie should be registered
        separately for each movie. This is achieved by this method.
        """

        registered = np.zeros_like(self.analyzer.movie)

        shifts = self.position - self.trajectory

        for frame, *shift in shifts:
            registered[-frame] = np.roll(self.analyzer.movie[-frame], shift, axis=(0, 1))

        return registered

    def kymograph(self, width: int = 3, height: int = 51) -> NDArray:
        """create a kymograph for the bleach region.

        Parameters
        ----------
        width : int, optional
            The width of the window to take from each frame. Has to be an odd
            integer. Default is 3.
        height : int, optional
            The height of the window to take from each frame. Has to be an
            odd integer. Default is 51.

        Returns
        -------
        NDArray
            An NDArray of the same dtype as the original movie with varying
            shape.

        Raises
        ------
        ValueError
            Raised when either `width` or `height` are not odd.
        IndexError
            Raised when the kymograph would include regions outside the image.
            This happens, if the bleached region is close to the border and
            the selected region for the kymograph is too large.
        """

        if (width % 2) != 1 or (height % 2) != 1:
            raise ValueError("'width' and 'height' must be odd!")

        _, r, c = self.position
        _, rows, cols = self.registered.shape

        rlo, rhi = r - (height // 2), r + (height // 2) + 1
        clo, chi = c - (width // 2), c + (width // 2) + 1

        if rlo < 0 or clo < 0 or rhi > rows or chi > cols:
            raise IndexError("Kymograph too close to image border!")

        subset = self.registered[:, rlo:rhi, clo:chi]

        return np.concatenate(subset, axis=1)

    @cached_property
    def position(self) -> NDArray[np.int_]:
        for row in self.trajectory:
            if row[0] == 0:
                return row

    def measure(self, radius: int = 5) -> pd.DataFrame:
        """Measure intensities in the bleach region.

        Parameters
        ----------
        radius : int, optional
            The radius of the region to be measured. Default is 5 (pixels).

        Returns
        -------
        pd.DataFrame
            A dataframe containing the measured intensities for this region.
            Each timepoint in the movie is represented by one row. The
            dataframe also contains the metadata stored in the `meta`
            attribute. The columns are:
                * Frame: The frame/timepoint.
                * Raw: The raw mean intensity in the measured region.
                * Foreground: The mean of the 5% brightest pixels (can be used for bleach
                correction).
                * Background: The mean intensity of the 95% darkest pixels (can be used for
                background correction).
        """

        frames, rows, cols = self.analyzer.movie.shape
        buffer = np.zeros((rows, cols), dtype="bool")
        _, r, c = self.position
        dr, dc = draw.disk((r, c), radius=radius, shape=(rows, cols))

        values = []

        for frame in range(frames):
            image = self.registered[frame]

            # Determine foreground.
            np.greater_equal(image, np.quantile(image, 0.95), out=buffer)

            # Save measurements.
            values.append(
                {
                    "Frame": frame,
                    "Raw": image[dr, dc].mean(),
                    "Foreground": image[buffer].mean(),
                    "Correction": np.quantile(image, 0.99),
                    "Background": image[~buffer].mean(),
                    **self.meta,
                }
            )

        self.measurements = pd.DataFrame(values)
        return self.measurements

    def _read_csv(self) -> NDArray[np.int_]:
        """Read the trajectory from a CSV file.

        Returns
        -------
        NDArray[np.int_]
            An Nx3 array representing the trajectory of this bleach region,
            i.e., the center for the region at each timepoint.
        """

        df = pd.read_csv(self.path).round(decimals=0)
        df -= 1

        return df[["Slice", "Y", "X"]].to_numpy(dtype="int")

    def normalize_measurement(self, prebleach_frames: int, interval: float = 1.0) -> pd.DataFrame:
        """Correct and normalize intensities in the bleach region.

        Parameters
        ----------
        prebleach_frames : int
            The number of prebleach frames.
        interval : float, optional
            The intervall time of the movie. The unit (seconds, minutes) is
            arbitrary. The frame number will be multiplied by this number.
            Default is 1.0.

        Returns
        -------
        pd.DataFrame
            The same dataframe as `df` but with additional columns  for
            corrected and normalized intensities.
        """

        pb = self.analyzer.prebleach_frames

        def correct_background(df: pd.DataFrame) -> pd.DataFrame:
            df["Corrected"] = df["Raw"] / df["Foreground"]
            return df

        def normalize_intensity(df: pd.DataFrame) -> pd.DataFrame:
            df["Normalized"] = df["Corrected"] - df["Corrected"][pb]
            df["Normalized"] = df["Normalized"] / df["Normalized"][:pb].mean()
            return df

        def add_time(df: pd.DataFrame) -> pd.DataFrame:
            df["Time"] = (df["Frame"] - pb) * interval
            return df

        return self.measurements.pipe(correct_background).pipe(normalize_intensity).pipe(add_time)

import numpy as np
from scipy import ndimage as ndi
import cv2

from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import reconstruction
from skimage.morphology import footprint_rectangle

from typing import Optional
from numpy.typing import NDArray


__all__ = ["peak_prominence", "peak_prominence2"]


def peak_prominence2(
    image: NDArray[np.uint8],
    labels: Optional[NDArray[np.uint16]] = None,
    dst: Optional[NDArray[np.uint16]] = None,
    prominence: int = 10,
) -> NDArray[np.uint16]:

    if dst is None:
        dst = np.zeros_like(image, dtype=np.uint16)

    if labels is None:
        shift_img = cv2.subtract(image, prominence)
        recon_img = reconstruction(shift_img, image, method="dilation")
        resid_img = image - recon_img
        np.greater_equal(resid_img, prominence, out=dst)

        # Select a random single pixel for shallow plateaus.
        taken = np.zeros_like(image, dtype=np.bool_)
        rows, cols = np.nonzero(dst)
        dst[...] = 0
        for r, c in zip(rows, cols):
            if not taken[r, c]:
                dst[r, c] = 1
                r0, r1 = r - 2, r + 3
                c0, c1 = c - 2, c + 3
                taken[r0:r1, c0:c1] = True

    else:
        for region in regionprops(labels):
            peak_prominence2(image[region.slice], dst=dst[region.slice], prominence=prominence)

            # Label peaks according to labels.
            rows, cols = np.nonzero(dst[region.slice])
            dst[region.slice][rows, cols] = labels[region.slice][rows, cols]

    return dst


def peak_prominence(
    image: NDArray[np.uint8],
    labels: Optional[NDArray[np.int32]] = None,
    prominence: int = 10,
    min_distance: int = 5,
    **kwargs
) -> NDArray[np.uint16]:
    """Find all peaks in an image with certain prominence.

    Parameters
    ----------
    image : NDArray[np.uint8]
        The image in which to search for peaks. This is currently limited to
        8-bit grayscale images to allow efficient processing.
    labels : NDArray[np.int32], optional
        A label image the same shape as `image`. When provided, only non-zero
        pixels will be considered during search. Restricting the search to
        certain regions this way can significantly speed up the search. The
        expected dtype is `np.int32` as returned from the `label` function in
        scipy but in principle any integer dtype works.
    prominence : int, optional
        The minimum prominence a local maxima must have to be returned. When
        thinking of an image as an elevation map, then prominence is the
        minimum vertical distance one has to descent in order to reach a
        pixel with a higher value. This is useful to filter out local maxima
        that can be considered noise. Default is 10.
    min_distance : int, optional
        The minimum distance between two local maxima in order to consider
        both. If two maxima are closer together, the larger one is selected.
        Default is 5.
    kwargs : dict, optional
        Additional keyworded arguments passed down to `peak_local_max()` from
        scikit-image package.

    Returns
    -------
    NDArray[np.uint16]
        A Nx2 array holding the row and column coordinates of all local
        maxima with the specified prominence.

    Raises
    ------
    ValueError
        If `image` is not of dtype `np.uint8` or if `labels` and `image` are
        of different shape.

    Notes
    -----
    Computational speed is not great when compared to the equivalent function
    from ImageJ, but it is still reasonably fast to process also large images
    (say 8000x6000px) when using multiprocessing to process images
    simultaneously.
    """

    if image.dtype != np.uint8:
        raise ValueError("'image' must have dtype uint8!")

    if labels is not None and labels.shape != image.shape:
        raise ValueError("'labels' and 'image' must have the same shape!")

    # Construct some temporary images.
    img_mask = np.zeros_like(image, dtype=np.uint8)
    img_lbls = np.zeros_like(image, dtype=np.uint16)

    img_buff = np.zeros_like(image, dtype=np.uint8)
    if labels is not None:
        np.greater(labels, 0, out=img_buff)

    # Start by searching local maxima as candidate points.
    candidates = peak_local_max(
        image, labels=labels, threshold_abs=prominence, min_distance=min_distance, **kwargs
    )

    # Build a dict with coordinates for each max intensity value.
    rows, cols = candidates.T
    intensities, indices = np.unique(image[rows, cols], return_inverse=True)
    max_intensities = {}
    for i, intensity in enumerate(intensities):
        max_intensities[intensity] = candidates[indices == i]

    peaks = []

    # Now iterate through all candidates to check if they have the required
    # prominence.
    for intensity, coordinates in max_intensities.items():

        r, c = coordinates.T
        np.greater(image, intensity - prominence, out=img_mask)

        # Remove pixels that are too far away from any candidate.
        cv2.bitwise_and(img_mask, img_buff, dst=img_mask, mask=img_mask)

        cv2.connectedComponents(
            image=img_mask,
            labels=img_lbls,
            connectivity=8,  # Significantly faster than 4-connectivity.
            ltype=cv2.CV_16U,
        )

        # Get all relevant labels for this iteration.
        unique_labels, indices = np.unique(img_lbls[r, c], return_inverse=True)

        # Calculate maxima for all relevant labels.
        maxima = ndi.labeled_comprehension(
            image, img_lbls, index=unique_labels, func=np.max, out_dtype=image.dtype, default=0
        )
        maxima = maxima[indices]
        np.equal(maxima, intensity, out=maxima)

        peaks.append(coordinates[maxima.view(np.bool_)])

    r, c = np.concatenate(peaks).T
    out = np.zeros_like(image, dtype=np.uint16)
    out[r, c] = labels[r, c]

    # Convert to Nx2 array of coordinates.
    return out

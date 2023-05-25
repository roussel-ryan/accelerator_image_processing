from copy import deepcopy

import numpy as np
from matplotlib import patches, pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_triangle


def get_beam_data(
    img,
    roi_data=None,
    n_stds=2,
    min_beam_intensity=10000,
    enforce_penalty=True,
    visualize=False,
):
    """
    Processes a 2D image with a specified region of interest.
    Processing steps:
    - applies a Gaussian smoothing filter
    - applies a threshold calculated using triangle thresholding
    https://forum.image.sc/t/understanding-imagej-implementation-of-the-triangle-algorithm-for-threshold/752

    This function calculates a penalty value based on the maximum distance between
    the corners of a 2*sigma bounding box around the beam and the center of the
    ROI. If this distance is outside the radius of a circle inscribed inside the ROI
    then the penalty value is positive, otherwise it is negative. If no beam is
    detected (total pixel value is below `min_beam_intensity`) the penalty value is
    1000.

    If `penalty` > 0 and `enforce_penalty` is True, beam statistics returned are
    Nans.

    Returns results in a dict with the following keys (units in pixels)
    - `Cx`: x-centroid
    - `Cy`: y-centroid
    - `Sx`: x-variance
    - `Sy`: y-variance
    - `penalty`: penalty function used for constraints


    :param img: N x M nd.array containing image data
    :param roi_data: matplotlib style bounding box coordinates [lower_left_x,
    lower_left_y, width, height]
    :param n_stds: float, size of box around beam in standard deviations
    :param visualize: bool, visualize image processing
    :param min_beam_intensity: float, minimum total pixel intensity required to
    positively identify a beam
    :param enforce_penalty: bool, flag to replace measurements with Nans if `penalty` > 0

    :return: dict
    """
    if roi_data is not None:
        cropped_image = deepcopy(img)[
            roi_data[0] : roi_data[0] + roi_data[2],
            roi_data[1] : roi_data[1] + roi_data[3],
        ]
    else:
        cropped_image = deepcopy(img)
        roi_data = [0, 0, *cropped_image.shape]

    filtered_image = gaussian_filter(cropped_image, 3.0)

    threshold = threshold_triangle(filtered_image)
    thresholded_image = np.where(
        filtered_image - threshold > 0, filtered_image - threshold, 0
    )

    total_intensity = np.sum(thresholded_image)

    cx, cy, sx, sy = calculate_stats(thresholded_image)
    c = np.array((cx, cy))
    s = np.array((sx, sy))

    # get beam region
    pts = np.array(
        (
            c - n_stds * s,
            c + n_stds * s,
            c - n_stds * s * np.array((-1, 1)),
            c + n_stds * s * np.array((-1, 1)),
        )
    )

    # get distance from beam region to ROI center
    roi_c = np.array((roi_data[2], roi_data[3])) / 2
    roi_radius = np.min((roi_c * 2, np.array(thresholded_image.shape))) / 2

    # validation
    if visualize:
        fig, ax = plt.subplots()
        c = ax.imshow(thresholded_image, origin="lower")
        ax.plot(cx, cy, "+r")
        fig.colorbar(c)

        rect = patches.Rectangle(
            pts[0], *s * n_stds * 2.0, facecolor="none", edgecolor="r"
        )
        ax.add_patch(rect)

        circle = patches.Circle(roi_c, roi_radius, facecolor="none", edgecolor="r")
        ax.add_patch(circle)

    distances = np.linalg.norm(pts - roi_c, axis=1)

    # subtract radius to get penalty value
    penalty = np.max(distances) - roi_radius

    results = {
        "Cx": cx,
        "Cy": cy,
        "Sx": sx,
        "Sy": sy,
        "penalty": penalty,
    }

    # penalize no beam
    if total_intensity < min_beam_intensity:
        penalty = 1000
        for name in ["Cx", "Cy", "Sx", "Sy"]:
            results[name] = None

    # enforce penalty
    if penalty > 0 and enforce_penalty:
        for name in ["Cx", "Cy", "Sx", "Sy"]:
            results[name] = None

    return results


def calculate_stats(img):
    rows, cols = img.shape
    row_coords = np.arange(rows)
    col_coords = np.arange(cols)

    m00 = np.sum(img)
    m10 = np.sum(col_coords[:, np.newaxis] * img.T)
    m01 = np.sum(row_coords[:, np.newaxis] * img)

    Cx = m10 / m00
    Cy = m01 / m00

    m20 = np.sum((col_coords[:, np.newaxis] - Cx) ** 2 * img.T)
    m02 = np.sum((row_coords[:, np.newaxis] - Cy) ** 2 * img)

    sx = (m20 / m00) ** 0.5
    sy = (m02 / m00) ** 0.5

    return Cx, Cy, sx, sy


def rectangle_union_area(llc1, s1, llc2, s2):
    # Compute the intersection of the two rectangles
    x1, y1 = llc1
    x2, y2 = llc2
    w1, h1 = s1
    w2, h2 = s2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap

    # Compute the areas of the two rectangles
    rect1_area = w1 * h1
    rect2_area = w2 * h2

    # Compute the area of the union
    union_area = rect1_area + rect2_area - overlap_area

    return union_area

from typing import *

from collections import namedtuple

import cv2
import numpy as np

from video_stabilizer.io import FrameGenerator, generate_frames_from_video


AffineTransformation = namedtuple('AffineTransformation', ['dx', 'dy', 'da'])


def stabilize_video(frame_generator: Optional[FrameGenerator] = None,
                    video_capture: Optional[cv2.VideoCapture] = None,
                    full_affine: bool = True,
                    rotate: bool = False,
                    center_crop: bool = False,
                    smoothing_radius: int = 50) -> Generator[Tuple[np.ndarray, AffineTransformation], None, None]:
    """ Given a `FrameGenerator` OR a `cv2.VideoCapture` yield frames from stabilized video.

    Before yielding first stabilized frame, the complete sequence is iterated over to calculate
    a smoothed trajectory that is then used to  stabilize frames.

    Based on: https://github.com/krutikabapat/Video-Stabilization-using-OpenCV/blob/master/video_stabilization.py

    Parameters
    ----------
    frame_generator : FrameGenerator, optional
        Generator yielding frames to stabilze. Assumed to be finite.
    video_capture : cv2.VideoCapture, optional
        Video containing frames to stabilize. Assumed to be finite.
    full_affine : bool
        If True, `estimateAffine2D` will be used to estimate movement otherwise `estimateAffinePartial2D`.
    rotate : bool
        If True, rotation will be used otherwise only translation.
    center_crop : bool
        If True, the center of the image will be cropped out by a fixed margin to remove border artifacts.
    smoothing_radius : int
        Number of frames used to smooth trajectory.

    Yields
    ------
    stabilized_frame : array
        Array with stabilized frame.
    transform : AffineTransformation
        Named tuple of delta-x, -y, -angle of transformation.
    """

    assert not (frame_generator is not None and video_capture is not None), 'pass frame_generator OR video_capture, ' \
                                                                            'not both.'
    assert (frame_generator is not None or video_capture is not None), 'pass frame_generator OR video_capture.'
    if frame_generator is None:
        frame_generator = generate_frames_from_video(video_capture)
        n_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        n_frames = frame_generator.n_frames

    frame_iterator = iter(frame_generator)
    first_frame = next(frame_iterator)
    height, width = first_frame.shape[:2]

    previous_frame = _safe_as_gray(first_frame)
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 1):
        previous_points = cv2.goodFeaturesToTrack(previous_frame,
                                                  maxCorners=200,
                                                  qualityLevel=0.01,
                                                  minDistance=30,
                                                  blockSize=3)

        next_frame = next(frame_iterator)
        current_frame = _safe_as_gray(next_frame)

        da, dx, dy = _movement_from_previous_frame(current_frame, previous_frame, previous_points, full_affine)
        transforms[i] = [dx, dy, da]
        previous_frame = current_frame

    if video_capture is not None:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_generator = generate_frames_from_video(video_capture)

    frame_iterator = iter(frame_generator)

    # Find the cumulative sum of transform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = _smooth_trajectory(trajectory, smoothing_radius=smoothing_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 1):
        # Read next frame
        frame = next(frame_iterator)
        assert frame.shape[:2] == (height, width)

        # Extract transformations from the new transformation array
        dx, dy, da = transforms_smooth[i]
        transform = AffineTransformation(dx, dy, da)
        stabilized_frame = transform_frame(frame, transform, rotate, center_crop)
        yield stabilized_frame, transform


def transform_frame(frame: np.array,
                    transform: AffineTransformation,
                    rotate: bool = False,
                    center_crop: bool = False) -> np.array:
    """ Perform affine transformation of a single image-frame.

    Parameters
    ----------
    frame : array
        Image frame.
    transform : AffineTransformation
        Delta-x, -y, -angle to use for transformation.
    rotate : bool
        If True, rotation will be used otherwise only translation.
    center_crop : bool
        If True, the center of the image will be cropped out by a fixed margin to remove border artifacts.

    Returns
    -------
    array
    """
    dx, dy, da = transform
    height, width = frame.shape[:2]
    # Reconstruct transformation matrix accordingly to new values
    transformation_matrix = np.zeros((2, 3), np.float32)

    if rotate:
        transformation_matrix[0, 0] = np.cos(da)
        transformation_matrix[0, 1] = -np.sin(da)
        transformation_matrix[1, 0] = np.sin(da)
        transformation_matrix[1, 1] = np.cos(da)
    else:
        transformation_matrix[0, 0] = 1
        transformation_matrix[0, 1] = 0
        transformation_matrix[1, 0] = 0
        transformation_matrix[1, 1] = 1

    transformation_matrix[0, 2] = dx
    transformation_matrix[1, 2] = dy

    # Apply affine wrapping to the given frame
    stabilized_frame = cv2.warpAffine(frame, transformation_matrix, (width, height), flags=cv2.INTER_NEAREST)
    if center_crop:
        stabilized_frame = _scale_around_center(stabilized_frame)
    return stabilized_frame


def _movement_from_previous_frame(current_frame, previous_frame, previous_points, full_affine):
    # Track feature points
    # status = 1. if flow points are found
    # err if flow was not find the error is not defined
    # current_points = calculated new positions of input features in the second image
    current_points, status, err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, previous_points, None)
    valid_indexes = np.where(status == 1)[0]
    previous_points = previous_points[valid_indexes]
    current_points = current_points[valid_indexes]
    assert previous_points.shape == current_points.shape
    if full_affine:
        transformation_matrix = cv2.estimateAffine2D(previous_points, current_points)[0]
    else:
        transformation_matrix = cv2.estimateAffinePartial2D(previous_points, current_points)[0]
    dx = transformation_matrix[0, 2]
    dy = transformation_matrix[1, 2]
    # Extract rotation angle
    da = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
    return da, dx, dy


def _moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size

    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def _smooth_trajectory(trajectory, smoothing_radius):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = _moving_average(trajectory[:, i], radius=smoothing_radius)

    return smoothed_trajectory


def _scale_around_center(frame, scaling_factor=1.04):
    height, width = frame.shape
    T = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scaling_factor)
    frame = cv2.warpAffine(frame, T, (width, height))
    return frame


def _safe_as_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        return img
    else:
        raise ValueError(f'img should be 3- or 2-dimensional not {img.ndim}-dimensional.')

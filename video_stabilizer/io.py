from pathlib import Path
from typing import Sequence, Union, Optional, Callable, Generator

import cv2
import numpy as np


class FrameGenerator:

    """ Yields image frames given sequence of paths to images.

    Parameters
    ----------
    paths : Sequence
        Ordered sequence of paths.
    flag : int
        Flag passed to `cv2.imread`, default `cv2.IMREAD_UNCHANGED`.
    preprocessing_fn : callable, optional
        Function to preprocess frames before yielding.

    Yields
    ------
    frame : array
        Loaded image frame.
    """

    def __init__(self,
                 paths: Sequence[Union[str, Path]],
                 flag: int = cv2.IMREAD_UNCHANGED,
                 preprocessing_fn: Optional[Callable] = None):
        self.paths = paths
        self.n_frames = len(paths)
        self.flag = flag
        self.preprocessing_fn = preprocessing_fn

    def __iter__(self) -> np.ndarray:
        for path in self.paths:
            img = cv2.imread(path, self.flag)
            if self.preprocessing_fn is not None:
                img = self.preprocessing_fn(img)
            yield img


def generate_frames_from_video(video_capture: cv2.VideoCapture) -> Generator[np.array, None, None]:
    """ Generate image frames from video-file.

    Parameters
    ----------
    video_capture : cv2.VideoCapture
        Video-buffer.

    Yields
    ------
    array, tuple, int
        Frame, (width, height), number of frames.
    """
    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            yield frame
    finally:
        video_capture.release()

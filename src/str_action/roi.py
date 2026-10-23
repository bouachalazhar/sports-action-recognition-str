"""Region-of-interest extraction from green chromaticity and optical flow."""

from __future__ import annotations

import itertools

import av
import cv2
import numpy as np


def optical_flow(path: str) -> np.ndarray:
    """Return the union of per-frame Farneback motion masks."""
    capture = cv2.VideoCapture(path)
    success, first = capture.read()
    if not success:
        raise ValueError("could not read the first frame")

    previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first)
    hsv[..., 1] = 255
    masks: list[np.ndarray] = []

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = np.zeros((*current.shape, 2), dtype=np.float32)
        cv2.calcOpticalFlowFarneback(previous, current, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        normalized = np.zeros_like(magnitude, dtype=magnitude.dtype)
        cv2.normalize(magnitude, normalized, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = normalized
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        _, mask = cv2.threshold(bgr[:, :, 2], 0, 255, cv2.THRESH_OTSU)
        masks.append(mask)
        previous = current

    capture.release()
    if not masks:
        raise ValueError("no frame processed for optical flow")

    intersection = masks[0]
    for mask in masks[1:]:
        intersection = cv2.bitwise_or(intersection, mask)
    return intersection


def green_chromaticity(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the field mask and the player mask of a video."""
    with av.open(path) as container:
        stream = container.decode(video=0)
        try:
            first = next(stream)
        except StopIteration as error:
            raise ValueError("no frame could be decoded") from error

        alpha = 0.5
        height, width, _channels = first.to_ndarray(format="rgb24").shape
        kernel_size = max(3, int(alpha * np.sqrt(height**2 + width**2) / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        def chromaticity(image: np.ndarray) -> np.ndarray:
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel).astype(np.float32)
            total = np.sum(opened, axis=-1)
            green = np.zeros((height, width), dtype=np.float32)
            valid = total != 0
            green[valid] = opened[..., 1][valid] / total[valid]
            return (green * 255).astype(np.uint8)

        field = np.zeros((height, width), dtype=np.uint8)
        players = np.zeros((height, width), dtype=np.uint8)
        for index, frame in enumerate(itertools.chain((first,), stream)):
            image = frame.to_ndarray(format="rgb24")
            if index % 30 == 0:
                _, otsu = cv2.threshold(chromaticity(image), 0, 255, cv2.THRESH_OTSU)
                field = cv2.bitwise_or(field, otsu)
            if index % 45 == 0:
                _, otsu = cv2.threshold(~chromaticity(image), 0, 255, cv2.THRESH_OTSU)
                players = cv2.bitwise_or(players, otsu)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        field, connectivity=8
    )
    if labels_count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == largest).astype("uint8") * 255
    else:
        component = np.zeros_like(field)

    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    field_closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, closing)
    return field_closed, players


def roi_str(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine chromaticity and motion into a region of interest.

    Returns:
        The field mask, the raw player mask and the filtered player mask.
    """
    field, players = green_chromaticity(path)
    motion = optical_flow(path)

    on_field = cv2.bitwise_and(field, motion)
    moving_players = cv2.bitwise_and(on_field, players)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    moving_players = cv2.morphologyEx(moving_players, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        moving_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = np.zeros_like(moving_players)
    for contour in contours:
        if cv2.contourArea(contour) >= 100:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)

    return field, players, filtered


__all__ = ["green_chromaticity", "optical_flow", "roi_str"]

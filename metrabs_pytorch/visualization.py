"""Pose visualization utilities using OpenCV."""

import numpy as np
import cv2


def draw_poses(frame, poses2d, joint_edges):
    """Draw 2D pose skeletons on a BGR frame in-place.

    Args:
        frame: BGR image (H, W, 3) as a numpy array.
        poses2d: Array of shape (N, J, 2) with 2D joint coordinates.
        joint_edges: Array of shape (E, 2) with joint index pairs.
    """
    if poses2d.size == 0:
        return

    for pose in poses2d:
        for joint_from, joint_to in joint_edges:
            pt1 = pose[joint_from]
            pt2 = pose[joint_to]
            if np.isnan(pt1).any() or np.isnan(pt2).any():
                continue
            cv2.line(
                frame,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                (255, 0, 0),
                2,
            )

        for joint in pose:
            if np.isnan(joint).any():
                continue
            cv2.circle(frame, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1)

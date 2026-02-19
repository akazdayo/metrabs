"""UDP pose sender for streaming 3D pose data."""

import json
import socket

import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100


class UdpPoseSender:
    """Send 3D pose data as JSON over UDP.

    Coordinate conversion: MeTRAbs camera space (mm, Y-down, Z-forward)
    -> Godot world space (meters, Y-up, Z-backward).

    Joints whose 2D projection falls outside the camera frame are sent
    as null.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)

    def send(self, pose3d, pose2d, frame_shape, joint_names, joint_edges):
        """Encode and send a single pose via UDP."""
        h, w = frame_shape[:2]
        positions = []
        for joint3d, joint2d in zip(pose3d, pose2d):
            if np.isnan(joint3d).any() or np.isnan(joint2d).any():
                positions.append(None)
                continue
            px, py = float(joint2d[0]), float(joint2d[1])
            if px < 0 or px >= w or py < 0 or py >= h:
                positions.append(None)
                continue
            x = float(joint3d[0]) / 1000
            y = -float(joint3d[1]) / 1000
            z = -float(joint3d[2]) / 1000
            positions.append([x, y, z])

        data = json.dumps(
            {
                "joint_names": [str(n) for n in joint_names],
                "joint_positions": positions,
                "joint_edges": joint_edges.tolist(),
            }
        ).encode("utf-8")
        try:
            self.sock.sendto(data, self.addr)
        except OSError as e:
            print(f"UDP send failed: {e}")

    def close(self):
        self.sock.close()

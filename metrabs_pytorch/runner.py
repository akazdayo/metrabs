"""Real-time camera pose estimation loop."""

import time

import cameralib
import cv2
import torch

from metrabs_pytorch.model_loader import (
    MODEL_DIR_NAME,
    ensure_model_dir,
    load_multiperson_model,
)
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.sender.udp import UdpPoseSender
from metrabs_pytorch.visualization import draw_poses


def run(
    model_dir=MODEL_DIR_NAME,
    skeleton="smpl_24",
    device_name="cuda",
    camera_index=0,
    fov_degrees=55,
    udp_host="127.0.0.1",
    udp_port=9000,
    num_aug=5,
    detector_threshold=0.2,
    max_detections=1,
):
    """Run the real-time pose estimation loop.

    Args:
        model_dir: Path to the model directory.
        skeleton: Skeleton format name (e.g. "smpl_24").
        device_name: Torch device string ("cuda" or "cpu").
        camera_index: OpenCV camera index.
        fov_degrees: Camera field-of-view in degrees.
        udp_host: UDP destination host.
        udp_port: UDP destination port.
        num_aug: Number of test-time augmentation crops.
        detector_threshold: Person detector confidence threshold.
        max_detections: Maximum number of persons to detect.
    """
    ensure_model_dir(model_dir)
    device = torch.device(device_name)

    print("loading model...")
    estimator = load_multiperson_model(model_dir, device)
    joint_edges = estimator.per_skeleton_joint_edges[skeleton].cpu().numpy()
    joint_names = estimator.per_skeleton_joint_names[skeleton]

    sender = UdpPoseSender(udp_host, udp_port)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("camera not available")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("failed to read from camera")

    camera = cameralib.Camera.from_fov(fov_degrees=fov_degrees, imshape=frame.shape[:2])
    intrinsic_matrix = torch.as_tensor(camera.intrinsic_matrix, device=device)
    distortion_coeffs = torch.as_tensor(
        multiperson_model.DEFAULT_DISTORTION, device=device
    )
    extrinsic_matrix = torch.as_tensor(
        multiperson_model.DEFAULT_EXTRINSIC_MATRIX, device=device
    )
    world_up_vector = torch.as_tensor(multiperson_model.DEFAULT_WORLD_UP, device=device)

    print("starting realtime prediction...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            pred = None
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1)
                with torch.inference_mode():
                    pred = estimator.detect_poses(
                        image,
                        intrinsic_matrix=intrinsic_matrix,
                        distortion_coeffs=distortion_coeffs,
                        extrinsic_matrix=extrinsic_matrix,
                        world_up_vector=world_up_vector,
                        default_fov_degrees=fov_degrees,
                        skeleton=skeleton,
                        num_aug=num_aug,
                        detector_threshold=detector_threshold,
                        max_detections=max_detections,
                    )
            except (ValueError, RuntimeError) as exc:
                if "expected a non-empty list of Tensors" not in str(exc):
                    raise

            elapsed = time.time() - start
            if pred is not None:
                poses2d = pred["poses2d"].detach().cpu().numpy()
                poses3d = pred["poses3d"].detach().cpu().numpy()
                if poses3d.size > 0:
                    sender.send(
                        poses3d[0],
                        poses2d[0],
                        frame.shape,
                        joint_names,
                        joint_edges,
                    )
                draw_poses(frame, poses2d, joint_edges)
            cv2.putText(
                frame,
                f"{elapsed * 1000:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Metrabs Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sender.close()

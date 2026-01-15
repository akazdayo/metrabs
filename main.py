import time
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import cameralib
import cv2
import posepile.joint_info
import simplepyutils as spu
from pythonosc import udp_client

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config


MODEL_TARBALL_URL = "https://bit.ly/metrabs_l_pt"
MODEL_DIR_NAME = "metrabs_eff2l_384px_800k_28ds_pytorch"
OSC_DEFAULT_HOST = "127.0.0.1"
OSC_DEFAULT_PORT = 9000


def ensure_model_dir(model_dir: str) -> Path:
    model_path = Path(model_dir)
    if model_path.exists():
        return model_path

    print(f"model dir not found, downloading {MODEL_TARBALL_URL}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "metrabs_model.tar.gz"
        urllib.request.urlretrieve(MODEL_TARBALL_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_path.parent)

    if not model_path.exists():
        raise FileNotFoundError(f"Expected model dir '{model_path}' was not created.")
    return model_path


def load_multiperson_model(model_dir: str, device: torch.device):
    # config.yaml を読む（Hydra設定）
    config_path = os.path.abspath(f"{model_dir}/config.yaml")
    get_config(config_path)
    cfg = get_config()

    # joint_info / backbone / metrabs本体
    ji_np = np.load(f"{model_dir}/joint_info.npz")
    ji = posepile.joint_info.JointInfo(ji_np["joint_names"], ji_np["joint_edges"])

    backbone_raw = getattr(effnet_pt, f"efficientnet_v2_{cfg.efficientnet_size}")()
    backbone = torch.nn.Sequential(effnet_pt.PreprocLayer(), backbone_raw.features)

    model = metrabs_pt.Metrabs(backbone, ji).to(device).eval()

    # 公式demoと同様にダミーforwardしてから ckpt.pt をロード
    inp = torch.zeros(
        (1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32, device=device
    )
    intr = torch.eye(3, dtype=torch.float32, device=device)[None]
    model((inp, intr))
    model.load_state_dict(torch.load(f"{model_dir}/ckpt.pt", map_location=device))

    # skeleton定義など
    skeleton_infos = spu.load_pickle(f"{model_dir}/skeleton_infos.pkl")
    joint_transform_matrix = np.load(f"{model_dir}/joint_transform_matrix.npy")

    with torch.device(device):
        return multiperson_model.Pose3dEstimator(
            model, skeleton_infos, joint_transform_matrix
        )


def draw_poses(frame, poses2d, joint_edges):
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


def resolve_joint_index(joint_names, candidates, label):
    joint_index_map = {name.lower(): idx for idx, name in enumerate(joint_names)}
    for candidate in candidates:
        key = candidate.lower()
        if key in joint_index_map:
            return joint_index_map[key]
    available = ", ".join(joint_names)
    raise ValueError(f"{label} joint not found. Available joints: {available}")


def send_osc_trackers(client, pose3d, tracker_indices):
    for tracker_id, joint_index in tracker_indices.items():
        joint = pose3d[joint_index]
        if np.isnan(joint).any():
            continue
        position = [float(value) for value in joint]
        client.send_message(f"/tracking/trackers/{tracker_id}/position", position)
        client.send_message(
            f"/tracking/trackers/{tracker_id}/rotation", [0.0, 0.0, 0.0]
        )


def main():
    model_dir = MODEL_DIR_NAME
    ensure_model_dir(model_dir)
    skeleton = "smpl_24"  # もしKeyErrorなら下のprintで候補を見て変更

    device = torch.device("cuda")

    print("loading model...")
    estimator = load_multiperson_model(model_dir, device)
    joint_edges = estimator.per_skeleton_joint_edges[skeleton].cpu().numpy()
    joint_names = estimator.per_skeleton_joint_names[skeleton]
    hip_index = resolve_joint_index(
        joint_names, ["pelv", "pelvis", "hip", "hips", "root", "spi1"], "hip"
    )
    left_foot_index = resolve_joint_index(
        joint_names,
        [
            "lank",
            "left_ankle",
            "leftankle",
            "ltoe",
            "left_toe",
            "lefttoe",
            "l_foot",
            "left_foot",
            "leftfoot",
        ],
        "left foot",
    )
    right_foot_index = resolve_joint_index(
        joint_names,
        [
            "rank",
            "right_ankle",
            "rightankle",
            "rtoe",
            "right_toe",
            "righttoe",
            "r_foot",
            "right_foot",
            "rightfoot",
        ],
        "right foot",
    )
    tracker_indices = {
        1: hip_index,
        3: left_foot_index,
        4: right_foot_index,
    }
    osc_client = udp_client.SimpleUDPClient(OSC_DEFAULT_HOST, OSC_DEFAULT_PORT)

    # skeleton名の候補確認（困ったらこれ）
    # print("available skeletons:", list(estimator.per_skeleton_joint_names.keys()))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera not available")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("failed to read from camera")

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=frame.shape[:2])
    intrinsic_matrix = torch.as_tensor(camera.intrinsic_matrix, device=device)
    distortion_coeffs = torch.as_tensor(
        multiperson_model.DEFAULT_DISTORTION, device=device
    )
    extrinsic_matrix = torch.as_tensor(
        multiperson_model.DEFAULT_EXTRINSIC_MATRIX, device=device
    )
    world_up_vector = torch.as_tensor(multiperson_model.DEFAULT_WORLD_UP, device=device)

    print("starting realtime prediction...")
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
                    default_fov_degrees=55,
                    skeleton=skeleton,
                    num_aug=5,
                    detector_threshold=0.2,
                    max_detections=1,
                )
        except (ValueError, RuntimeError) as exc:
            if "expected a non-empty list of Tensors" not in str(exc):
                raise

        elapsed = time.time() - start
        if pred is not None:
            poses2d = pred["poses2d"].detach().cpu().numpy()
            poses3d = pred["poses3d"].detach().cpu().numpy()
            if poses3d.size > 0:
                send_osc_trackers(osc_client, poses3d[0], tracker_indices)
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import time
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision.io
import cameralib
import posepile.joint_info
import simplepyutils as spu

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config


MODEL_TARBALL_URL = "https://bit.ly/metrabs_l_pt"
MODEL_DIR_NAME = "metrabs_eff2l_384px_800k_28ds_pytorch"


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


def main():
    model_dir = MODEL_DIR_NAME
    ensure_model_dir(model_dir)
    image_path = "test_image_3dpw.jpg"
    skeleton = "smpl_24"  # もしKeyErrorなら下のprintで候補を見て変更

    device = torch.device("cuda")

    print("loading model...")
    estimator = load_multiperson_model(model_dir, device)

    # skeleton名の候補確認（困ったらこれ）
    # print("available skeletons:", list(estimator.per_skeleton_joint_names.keys()))

    print("decoding image...")
    image = torchvision.io.read_image(image_path).to(device)  # uint8, CHW

    print("starting prediction...")
    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=image.shape[1:])
    intrinsic_matrix = torch.as_tensor(camera.intrinsic_matrix, device=device)
    distortion_coeffs = torch.as_tensor(
        multiperson_model.DEFAULT_DISTORTION, device=device
    )
    extrinsic_matrix = torch.as_tensor(
        multiperson_model.DEFAULT_EXTRINSIC_MATRIX, device=device
    )
    world_up_vector = torch.as_tensor(multiperson_model.DEFAULT_WORLD_UP, device=device)

    for i in range(500):
        start = time.time()
        with torch.inference_mode():
            pred = estimator.detect_poses(
                image,
                intrinsic_matrix=intrinsic_matrix,
                distortion_coeffs=distortion_coeffs,
                extrinsic_matrix=extrinsic_matrix,
                world_up_vector=world_up_vector,
                default_fov_degrees=55,
                skeleton=skeleton,
                num_aug=1,
                detector_threshold=0.3,
            )

        # numpy化（TFの tf.nest.map_structure 相当）
        pred_np = {k: v.detach().cpu().numpy() for k, v in pred.items()}
        # print(pred_np)
        print("elapsed:", time.time() - start)


if __name__ == "__main__":
    main()

"""Model downloading and loading utilities."""

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import posepile.joint_info
import simplepyutils as spu
import torch

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

MODEL_TARBALL_URL = "https://bit.ly/metrabs_l_pt"
MODEL_DIR_NAME = "metrabs_eff2l_384px_800k_28ds_pytorch"


def ensure_model_dir(model_dir):
    """Download and extract the model tarball if the directory does not exist.

    Returns the resolved Path to the model directory.
    """
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


def load_multiperson_model(model_dir, device):
    """Load the MeTRAbs multiperson pose estimator.

    Reads config.yaml, builds the backbone and model, performs a dummy
    forward pass, loads checkpoint weights, and wraps everything into a
    ``Pose3dEstimator``.
    """
    config_path = os.path.abspath(f"{model_dir}/config.yaml")
    get_config(config_path)
    cfg = get_config()

    ji_np = np.load(f"{model_dir}/joint_info.npz")
    ji = posepile.joint_info.JointInfo(ji_np["joint_names"], ji_np["joint_edges"])

    backbone_raw = getattr(effnet_pt, f"efficientnet_v2_{cfg.efficientnet_size}")()
    backbone = torch.nn.Sequential(effnet_pt.PreprocLayer(), backbone_raw.features)

    model = metrabs_pt.Metrabs(backbone, ji).to(device).eval()

    # Dummy forward (required before loading checkpoint)
    inp = torch.zeros(
        (1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32, device=device
    )
    intr = torch.eye(3, dtype=torch.float32, device=device)[None]
    model((inp, intr))
    model.load_state_dict(torch.load(f"{model_dir}/ckpt.pt", map_location=device))

    skeleton_infos = spu.load_pickle(f"{model_dir}/skeleton_infos.pkl")
    joint_transform_matrix = np.load(f"{model_dir}/joint_transform_matrix.npy")

    with torch.device(device):
        return multiperson_model.Pose3dEstimator(
            model, skeleton_infos, joint_transform_matrix
        )

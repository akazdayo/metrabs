import argparse
import urllib.request

import cameralib
import poseviz
import simplepyutils as spu
import torch
import torchvision.io

from metrabs_pytorch.model_loader import load_multiperson_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--image-path", type=str)
    spu.argparse.initialize(parser)

    model_dir = spu.FLAGS.model_dir
    device = torch.device("cuda")
    skeleton = "smpl+head_30"

    estimator = load_multiperson_model(model_dir, device)
    joint_names = estimator.per_skeleton_joint_names[skeleton]
    joint_edges = estimator.per_skeleton_joint_edges[skeleton].cpu().numpy()

    with torch.inference_mode(), torch.device("cuda"):
        with poseviz.PoseViz(joint_names, joint_edges, paused=True) as viz:
            image_filepath = get_image(spu.argparse.FLAGS.image_path)
            image = torchvision.io.read_image(image_filepath).cuda()
            camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=image.shape[1:])

            for num_aug in range(1, 50):
                pred = estimator.detect_poses(
                    image,
                    detector_threshold=0.01,
                    suppress_implausible_poses=False,
                    max_detections=1,
                    intrinsic_matrix=camera.intrinsic_matrix,
                    skeleton=skeleton,
                    num_aug=num_aug,
                )

                viz.update(
                    frame=image.cpu().numpy().transpose(1, 2, 0),
                    boxes=pred["boxes"].cpu().numpy(),
                    poses=pred["poses3d"].cpu().numpy(),
                    camera=camera,
                )


def get_image(source, temppath="/tmp/image.jpg"):
    if not source.startswith("http"):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath


if __name__ == "__main__":
    main()

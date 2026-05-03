#!/root/anaconda3/envs/MonoGS/bin/python
"""
Post-hoc rendering evaluation for saved MonoGS results.

Loads saved Gaussian models and computes PSNR, SSIM, LPIPS using GT camera poses.
Results are saved as psnr/gt_poses/final_result.json in each result directory.

Usage (from /root/MonoGS):
    python eval_rendering_postprocess.py
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from munch import munchify
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from utils.camera_utils import Camera
from utils.dataset import load_dataset

RESULT_DIRS = {
    "bonn":       "/root/results/monogs/results/datasets_bonn/2026-04-27-12-54-05",
    "smallcity":  "/root/results/monogs/results/datasets_smallcity/2026-04-27-13-19-48",
    "urbanscene": "/root/results/monogs/results/datasets_urbanscene/2026-04-27-13-50-06",
}

EVAL_INTERVAL = 5


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_dataset(name, result_dir):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Results: {result_dir}")

    config_path = os.path.join(result_dir, "config.yml")
    ply_path    = os.path.join(result_dir, "point_cloud/final/point_cloud.ply")

    if not os.path.isfile(ply_path):
        print(f"  [SKIP] point_cloud.ply not found: {ply_path}")
        return None

    config = load_config(config_path)
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])

    # Dataset
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    print(f"  Frames: {dataset.num_imgs}  (eval every {EVAL_INTERVAL}th)")

    # Gaussian model
    sh_degree = 3 if config["Training"]["spherical_harmonics"] else 0
    gaussians = GaussianModel(sh_degree, config=config)
    gaussians.load_ply(ply_path)
    print(f"  Gaussians loaded: {gaussians.get_xyz.shape[0]:,} points")

    # Rendering setup
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy,
        cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height,
    ).transpose(0, 1)

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    psnr_list, ssim_list, lpips_list = [], [], []

    indices = list(range(0, dataset.num_imgs, EVAL_INTERVAL))
    for count, idx in enumerate(indices):
        gt_image, _, gt_pose = dataset[idx]

        cam = Camera.init_from_dataset(dataset, idx, projection_matrix)
        # Render with GT pose (W2C matrix from dataset)
        cam.update_RT(gt_pose[:3, :3], gt_pose[:3, 3])

        with torch.no_grad():
            result = render(cam, gaussians, pipeline_params, background)

        if result is None:
            continue

        image = torch.clamp(result["render"], 0.0, 1.0)
        mask = gt_image > 0

        psnr_val  = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_val  = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        lpips_val = cal_lpips(image.unsqueeze(0), gt_image.unsqueeze(0))

        psnr_list.append(psnr_val.item())
        ssim_list.append(ssim_val.item())
        lpips_list.append(lpips_val.item())

        if (count + 1) % 10 == 0:
            print(f"  [{count+1}/{len(indices)}] "
                  f"PSNR={np.mean(psnr_list):.2f}  "
                  f"SSIM={np.mean(ssim_list):.3f}  "
                  f"LPIPS={np.mean(lpips_list):.3f}")

    output = {
        "mean_psnr":  float(np.mean(psnr_list)),
        "mean_ssim":  float(np.mean(ssim_list)),
        "mean_lpips": float(np.mean(lpips_list)),
        "n_frames":   len(psnr_list),
        "note": "GT poses used for rendering (estimated poses not saved during SLAM run)",
    }

    out_dir = os.path.join(result_dir, "psnr/gt_poses")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "final_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n  --- Results ({len(psnr_list)} frames) ---")
    print(f"  PSNR:  {output['mean_psnr']:.4f}")
    print(f"  SSIM:  {output['mean_ssim']:.4f}")
    print(f"  LPIPS: {output['mean_lpips']:.4f}")
    print(f"  Saved: {out_path}")

    return output


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_results = {}
    for name, result_dir in RESULT_DIRS.items():
        result = eval_dataset(name, result_dir)
        if result:
            all_results[name] = result

    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"{'Dataset':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Frames':>8}")
    print("-" * 50)
    for name, r in all_results.items():
        print(f"{name:<15} {r['mean_psnr']:>8.3f} {r['mean_ssim']:>8.3f} {r['mean_lpips']:>8.3f} {r['n_frames']:>8}")

    summary_path = "/root/results/monogs/metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

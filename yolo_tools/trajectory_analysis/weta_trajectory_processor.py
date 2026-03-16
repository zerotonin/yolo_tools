#!/usr/bin/env python3
"""
weta_trajectory_processor.py
============================
Matches YOLO trajectory .npy files to arena background_annotation .json files,
applies a perspective (homography) transform to convert normalised YOLO bbox
coordinates into real-world mm coordinates, interpolates missing detections,
and optionally filters the trajectories.

Arena real-world coordinate convention
--------------------------------------
    TL (0, 80) -------- TR (555, 80)
        |                    |
        |      arena         |
        |                    |
    BL (0,  0) -------- BR (555,  0)

Usage
-----
    python weta_trajectory_processor.py \
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \
        --fps 25 \
        --filter

Author: BRH Geurten 
"""

import numpy as np
import json
import os
import re
import glob
import argparse
from pathlib import Path
from scipy import signal

# ──────────────────────────────────────────────────────────────────────────────
# Real-world arena corners in mm  (TL, TR, BR, BL order matching JSON labels)
# ──────────────────────────────────────────────────────────────────────────────
REAL_WORLD_CORNERS = {
    "TL": np.array([0.0, 80.0]),
    "TR": np.array([555.0, 80.0]),
    "BR": np.array([555.0, 0.0]),
    "BL": np.array([0.0, 0.0]),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract the trial identifier from a filename
# e.g.  "hcrass1f_trial_redo_20250203_120147_trajectories.npy"
#    →  "hcrass1f_trial_redo_20250203_120147"
# ──────────────────────────────────────────────────────────────────────────────
def extract_trial_id(filename: str) -> str:
    """Strip the trailing suffix (_trajectories.npy or _background_annotation.json)
    and return the common trial identifier."""
    name = os.path.basename(filename)
    # Remove known suffixes
    for suffix in ["_trajectories.npy", "_background_annotation.json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# ──────────────────────────────────────────────────────────────────────────────
# File discovery and matching
# ──────────────────────────────────────────────────────────────────────────────
def discover_files(base_dir: str):
    """Walk the directory tree and return dicts mapping trial_id → filepath
    for both trajectory .npy files and background_annotation .json files."""

    traj_dir = os.path.join(base_dir, "trajectories")
    traj_files = glob.glob(os.path.join(traj_dir, "*_trajectories.npy"))
    traj_map = {extract_trial_id(f): f for f in traj_files}

    # Annotations live in the *_backgrounds folders
    annot_files = glob.glob(
        os.path.join(base_dir, "*_backgrounds", "*_background_annotation.json")
    )
    annot_map = {extract_trial_id(f): f for f in annot_files}

    return traj_map, annot_map


def match_files(traj_map: dict, annot_map: dict):
    """Return a list of (trial_id, traj_path, annot_path) for every matched pair.
    Also warns about unmatched files."""
    matched = []
    traj_only = set(traj_map) - set(annot_map)
    annot_only = set(annot_map) - set(traj_map)
    common = set(traj_map) & set(annot_map)

    for tid in sorted(common):
        matched.append((tid, traj_map[tid], annot_map[tid]))

    if traj_only:
        print(f"[WARN] {len(traj_only)} trajectory files with no matching annotation:")
        for t in sorted(traj_only):
            print(f"       - {t}")
    if annot_only:
        print(f"[WARN] {len(annot_only)} annotation files with no matching trajectory:")
        for a in sorted(annot_only):
            print(f"       - {a}")

    print(f"[INFO] Matched {len(matched)} trajectory–annotation pairs.\n")
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# Perspective transform (homography)
# ──────────────────────────────────────────────────────────────────────────────
def compute_homography(annotation: dict) -> np.ndarray:
    """Compute the 3×3 homography matrix that maps image‐pixel coordinates
    to real‐world mm coordinates using the four labelled arena corners.

    Uses a Direct Linear Transform (DLT) so we don't need OpenCV.
    """
    labels = annotation["labels"]
    vertices = annotation["vertices"]

    # Build source (pixel) and destination (mm) point arrays in matching order
    src_pts = []  # pixel coords
    dst_pts = []  # real-world mm
    for label, vertex in zip(labels, vertices):
        src_pts.append(vertex)  # [x_pixel, y_pixel]
        dst_pts.append(REAL_WORLD_CORNERS[label].tolist())

    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    # Solve for 3×3 homography H such that dst ~ H @ src (in homogeneous coords)
    # Using the standard DLT with 4 point correspondences
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    # Normalise so H[2,2] = 1
    H /= H[2, 2]
    return H


def apply_homography(H: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Apply homography H to an (N, 2) array of pixel coordinates.
    Returns (N, 2) array of real-world mm coordinates.
    NaN rows are preserved."""
    out = np.full_like(points_px, np.nan)
    valid = ~np.isnan(points_px).any(axis=1)
    if valid.sum() == 0:
        return out

    pts = points_px[valid]
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # (N, 3)
    transformed = (H @ pts_h.T).T  # (N, 3)
    transformed /= transformed[:, 2:3]  # dehomogenise
    out[valid] = transformed[:, :2]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Interpolation (borrowed from trajectoryAnalyser)
# ──────────────────────────────────────────────────────────────────────────────
def interpolate_1d(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps; forward/backward fill at edges."""
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) == 0:
        return arr
    valid_vals = arr[valid_idx]
    full_idx = np.arange(arr.size)
    interpolated = np.interp(full_idx, valid_idx, valid_vals)
    if valid_idx[0] > 0:
        interpolated[: valid_idx[0]] = valid_vals[0]
    if valid_idx[-1] < arr.size - 1:
        interpolated[valid_idx[-1] + 1 :] = valid_vals[-1]
    return interpolated


def interpolate_2d(matrix: np.ndarray) -> np.ndarray:
    """Interpolate each column of an (N, 2) array independently."""
    out = matrix.copy()
    for col in range(out.shape[1]):
        out[:, col] = interpolate_1d(out[:, col])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Low-pass filter
# ──────────────────────────────────────────────────────────────────────────────
def lowpass_filter(trajectory: np.ndarray, fps: float = 25.0,
                   order: int = 3, cutoff_hz: float = 2.0) -> np.ndarray:
    """Apply a Butterworth low-pass filter (zero-phase) to each column."""
    b, a = signal.butter(N=order, Wn=cutoff_hz / (fps / 2), btype="low")
    out = trajectory.copy()
    for col in range(out.shape[1]):
        out[:, col] = signal.filtfilt(b, a, out[:, col])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Bounding box helpers
# ──────────────────────────────────────────────────────────────────────────────
def bbox_to_pixels(bbox_norm: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalised [x1, y1, x2, y2] to pixel coordinates."""
    out = bbox_norm.copy()
    out[:, 0] *= img_w
    out[:, 2] *= img_w
    out[:, 1] *= img_h
    out[:, 3] *= img_h
    return out


def bbox_center_px(bbox_px: np.ndarray) -> np.ndarray:
    """Return (N, 2) centre [cx, cy] from [x1, y1, x2, y2] pixel bbox."""
    cx = 0.5 * (bbox_px[:, 0] + bbox_px[:, 2])
    cy = 0.5 * (bbox_px[:, 1] + bbox_px[:, 3])
    return np.column_stack([cx, cy])


def bbox_size_mm(bbox_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Compute bounding box width and height in mm by transforming all 4
    corners through the homography and taking extents."""
    # Extract the 4 corners of each bbox
    x1, y1, x2, y2 = bbox_px[:, 0], bbox_px[:, 1], bbox_px[:, 2], bbox_px[:, 3]

    tl = np.column_stack([x1, y1])
    tr = np.column_stack([x2, y1])
    br = np.column_stack([x2, y2])
    bl = np.column_stack([x1, y2])

    tl_mm = apply_homography(H, tl)
    tr_mm = apply_homography(H, tr)
    br_mm = apply_homography(H, br)
    bl_mm = apply_homography(H, bl)

    # Width: horizontal extent; Height: vertical extent
    all_x = np.stack([tl_mm[:, 0], tr_mm[:, 0], br_mm[:, 0], bl_mm[:, 0]], axis=1)
    all_y = np.stack([tl_mm[:, 1], tr_mm[:, 1], br_mm[:, 1], bl_mm[:, 1]], axis=1)

    w_mm = np.nanmax(all_x, axis=1) - np.nanmin(all_x, axis=1)
    h_mm = np.nanmax(all_y, axis=1) - np.nanmin(all_y, axis=1)
    return np.column_stack([w_mm, h_mm])


# ──────────────────────────────────────────────────────────────────────────────
# Main processing per trial
# ──────────────────────────────────────────────────────────────────────────────
def process_trial(trial_id: str, traj_path: str, annot_path: str,
                  fps: float = 10.0, do_filter: bool = False) -> dict:
    """
    Full pipeline for one trial:
      1. Load trajectory (.npy) and annotation (.json)
      2. Convert normalised YOLO bbox → pixel coords
      3. Compute bbox centre in pixels
      4. Compute homography from arena corners (pixels → mm)
      5. Transform centre to mm
      6. Interpolate NaN gaps
      7. (Optional) low-pass filter
      8. Compute bbox size in mm
      9. Return results dict

    Returns
    -------
    dict with keys:
        trial_id, species, traj_path, annot_path,
        center_mm        (N, 2) – interpolated (and filtered) centre in mm,
        center_mm_raw    (N, 2) – centre in mm before interpolation,
        bbox_size_mm     (N, 2) – [width_mm, height_mm] per frame,
        bbox_norm        (N, 4) – original normalised YOLO bboxes,
        n_frames, n_valid_detections, detection_rate,
        homography       (3, 3)
    """
    # ── Load data ──
    bbox_norm = np.load(traj_path)  # (N, 4) normalised [x1, y1, x2, y2]
    with open(annot_path, "r") as f:
        annotation = json.load(f)

    img_w = annotation["image_width"]
    img_h = annotation["image_height"]
    n_frames = bbox_norm.shape[0]
    n_valid = int(np.sum(~np.isnan(bbox_norm[:, 0])))

    # ── Infer species from trial_id ──
    if trial_id.startswith("hcrass"):
        species = "H. crassidens"
    elif trial_id.startswith("hm"):
        species = "H. maori"
    elif trial_id.startswith("hthora"):
        species = "H. thoracica"
    else:
        species = "unknown"

    # ── Pixel coordinates ──
    bbox_px = bbox_to_pixels(bbox_norm, img_w, img_h)
    center_px = bbox_center_px(bbox_px)

    # ── Homography ──
    H = compute_homography(annotation)

    # ── Centre → mm (raw, with NaNs) ──
    center_mm_raw = apply_homography(H, center_px)

    # ── Bbox size in mm (raw) ──
    bsize_mm = bbox_size_mm(bbox_px, H)

    # ── Interpolate ──
    center_mm = interpolate_2d(center_mm_raw)

    # ── Filter ──
    if do_filter and n_valid > 30:
        center_mm = lowpass_filter(center_mm, fps=fps)

    return {
        "trial_id": trial_id,
        "species": species,
        "traj_path": traj_path,
        "annot_path": annot_path,
        "center_mm": center_mm,
        "center_mm_raw": center_mm_raw,
        "bbox_size_mm": bsize_mm,
        "bbox_norm": bbox_norm,
        "n_frames": n_frames,
        "n_valid_detections": n_valid,
        "detection_rate": n_valid / n_frames if n_frames > 0 else 0.0,
        "homography": H,
        "image_width": img_w,
        "image_height": img_h,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing & saving
# ──────────────────────────────────────────────────────────────────────────────
def run_batch(base_dir: str, output_dir: str = None, fps: float = 10.0,
              do_filter: bool = False):
    """Discover, match, process, and save all trials."""
    if output_dir is None:
        output_dir = os.path.join(base_dir, "processed_trajectories")
    os.makedirs(output_dir, exist_ok=True)

    traj_map, annot_map = discover_files(base_dir)
    matched = match_files(traj_map, annot_map)

    results = []
    for trial_id, traj_path, annot_path in matched:
        print(f"  Processing {trial_id} ... ", end="")
        try:
            res = process_trial(trial_id, traj_path, annot_path,
                                fps=fps, do_filter=do_filter)
            results.append(res)

            # Save per-trial outputs
            out_prefix = os.path.join(output_dir, trial_id)
            np.save(f"{out_prefix}_center_mm.npy", res["center_mm"])
            np.save(f"{out_prefix}_center_mm_raw.npy", res["center_mm_raw"])
            np.save(f"{out_prefix}_bbox_size_mm.npy", res["bbox_size_mm"])

            print(f"OK  ({res['n_valid_detections']}/{res['n_frames']} detections, "
                  f"{res['detection_rate']:.1%})")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Save summary JSON ──
    summary = []
    for r in results:
        summary.append({
            "trial_id": r["trial_id"],
            "species": r["species"],
            "n_frames": r["n_frames"],
            "n_valid_detections": r["n_valid_detections"],
            "detection_rate": round(r["detection_rate"], 4),
            "traj_path": r["traj_path"],
            "annot_path": r["annot_path"],
        })

    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Summary written to {summary_path}")
    print(f"[INFO] Per-trial .npy files written to {output_dir}/")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Match weta YOLO trajectories to arena annotations and "
                    "transform to real-world mm coordinates."
    )
    parser.add_argument(
        "--base_dir", type=str,
        default="/home/geuba03p/weta_project/weta_videos_cropped",
        help="Root directory containing trajectories/ and *_backgrounds/ folders.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <base_dir>/processed_trajectories).",
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Frames per second of the video (default: 10).",
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Apply Butterworth low-pass filter after interpolation.",
    )

    args = parser.parse_args()
    run_batch(args.base_dir, args.output_dir, args.fps, args.filter)


if __name__ == "__main__":
    main()
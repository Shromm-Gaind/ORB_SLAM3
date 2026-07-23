#!/usr/bin/env python3
"""
Compare Shi-Tomasi + KLT (Lucas-Kanade) vs ORB + descriptor matching
for feature tracking in turbid underwater / low-contrast environments.

Designed for inspecting rock walls, structural dolphins, sea hulls, biofouling
where ORB-SLAM tends to struggle with inconsistent lighting and texture.

Usage:
    python compare_trackers.py /path/to/image/sequence --output ./results

The script:
  1. Reads an image sequence in order.
  2. For each consecutive frame pair, detects features with both methods.
  3. Tracks features with KLT (for Shi-Tomasi) and descriptor matching (for ORB).
  4. Scores each tracker on:
       - feature count
       - forward-backward error (track-then-backtrack drift)
       - epipolar inlier ratio (RANSAC fundamental matrix)
       - bulk motion consistency (direction agreement of the flow field)
  5. Writes:
       - a side-by-side comparison video
       - per-frame metrics CSV
       - summary plots
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Config — exposed at top so you can tweak for your dataset without
# digging into the code. Defaults chosen for low-contrast/turbid scenes.
# ----------------------------------------------------------------------

# Shi-Tomasi (goodFeaturesToTrack)
SHI_TOMASI_PARAMS = dict(
    maxCorners=500,
    qualityLevel=0.01,   # low: turbid scenes have weak corners
    minDistance=7,
    blockSize=7,
)

# KLT (calcOpticalFlowPyrLK)
LK_PARAMS = dict(
    winSize=(21, 21),    # larger window helps with low texture
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# ORB
ORB_PARAMS = dict(
    nfeatures=500,       # match Shi-Tomasi maxCorners for fairness
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=10,    # lower than default (20) for low-contrast
)

# ORB-SLAM-style quadtree distribution. Mirrors the ORBextractor in
# ORB-SLAM2/3: detect FAST per cell over a pyramid, with a fallback to a
# lower threshold when the high threshold finds nothing in a cell, then
# spatially distribute keypoints via quadtree so they can't clump.
ORB_QT_CELL_PX = 30              # initial cell size in pixels (ORB-SLAM default)
ORB_QT_FAST_HIGH = 20            # primary FAST threshold per cell
ORB_QT_FAST_LOW = 7              # fallback when high finds nothing

# Quality thresholds
FB_ERROR_THRESHOLD = 1.0          # px — forward-backward error to count as "good"
RANSAC_REPROJ_THRESHOLD = 3.0     # px — epipolar inlier threshold
MIN_MATCHES_FOR_F = 8             # need >=8 points for fundamental matrix

# Open-water detection: when BOTH trackers can barely find anything AND what
# they do find moves incoherently, the camera is almost certainly looking at
# featureless water. We expect this — the script flags it rather than letting
# the metrics confuse the comparison.
OPEN_WATER_MAX_FEATURES = 50      # both trackers below this
OPEN_WATER_MAX_MOTION = 0.55      # AND motion consistency near coin-flip


# ----------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image_paths(folder: Path) -> list[Path]:
    """Sorted list of image paths in folder (natural order via filename)."""
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {folder}")
    print(f"Found {len(paths)} images in {folder}")
    return paths


def read_gray(path: Path) -> np.ndarray:
    """Read image as 8-bit grayscale. Optional mild CLAHE for low-contrast."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


# ----------------------------------------------------------------------
# Tracker 1: Shi-Tomasi + KLT (Lucas-Kanade)
# ----------------------------------------------------------------------

def detect_shi_tomasi(gray: np.ndarray) -> np.ndarray:
    """Return Nx1x2 array of corner points (float32), or empty array."""
    pts = cv2.goodFeaturesToTrack(gray, mask=None, **SHI_TOMASI_PARAMS)
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts


def track_klt(prev_gray: np.ndarray,
              curr_gray: np.ndarray,
              prev_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track points from prev_gray -> curr_gray using LK pyramidal optical flow.
    Returns (good_prev, good_curr, fb_error) where fb_error is per-point
    forward-backward distance in pixels.
    """
    if len(prev_pts) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    # Forward
    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
    )
    # Backward (curr -> prev)
    back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, curr_pts, None, **LK_PARAMS
    )

    # Forward-backward error
    fb_err = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
    good = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1)

    prev_good = prev_pts.reshape(-1, 2)[good]
    curr_good = curr_pts.reshape(-1, 2)[good]
    fb_good = fb_err[good]
    return prev_good, curr_good, fb_good


# ----------------------------------------------------------------------
# Tracker 2: ORB + BFMatcher
# ----------------------------------------------------------------------

def get_orb():
    return cv2.ORB_create(**ORB_PARAMS)


def detect_orb_gridded(orb, gray, grid=(4, 4), per_cell=None):
    """
    Detect ORB features in grid cells to prevent clumping on high-contrast spots.
    Returns (keypoints, descriptors). Falls back to plain detectAndCompute if
    grid mode finds nothing in any cell.
    """
    h, w = gray.shape
    gh, gw = grid
    if per_cell is None:
        per_cell = max(20, ORB_PARAMS["nfeatures"] // (gh * gw))

    all_kp = []
    cell_orb = cv2.ORB_create(**{**ORB_PARAMS, "nfeatures": per_cell})
    for i in range(gh):
        for j in range(gw):
            y0, y1 = i * h // gh, (i + 1) * h // gh
            x0, x1 = j * w // gw, (j + 1) * w // gw
            sub = gray[y0:y1, x0:x1]
            kps = cell_orb.detect(sub, None)
            for kp in kps:
                kp.pt = (kp.pt[0] + x0, kp.pt[1] + y0)
            all_kp.extend(kps)

    if not all_kp:
        return [], None
    # Compute descriptors on the full image at the gathered keypoints
    all_kp, des = orb.compute(gray, all_kp)
    return all_kp, des


# ----------------------------------------------------------------------
# ORB-SLAM-style quadtree feature distribution
# ----------------------------------------------------------------------
#
# This is a faithful-enough port of ORB-SLAM2/3's ORBextractor strategy
# (Mur-Artal & Tardós, 2017). The key idea: FAST tends to clump on
# high-contrast spots, so naive ORB on biofouling concentrates 80% of
# its keypoints on 10% of the image. ORB-SLAM fixes this with:
#
#   1. Per-cell FAST detection over a pyramid (high threshold, fallback
#      to a lower one if the cell came up empty).
#   2. A quadtree that recursively subdivides the image, keeping at most
#      one keypoint per leaf node — the one with the strongest response.
#
# Result: roughly uniform spatial coverage, which is what we actually
# want for stable epipolar geometry estimation.

class _QTNode:
    """A node in the keypoint distribution quadtree."""
    __slots__ = ("x0", "y0", "x1", "y1", "kps", "no_more")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.kps = []
        self.no_more = False  # leaf with exactly 1 kp — don't subdivide further

    def split(self):
        """Split into 4 quadrants. Distribute kps by location."""
        mx = (self.x0 + self.x1) // 2
        my = (self.y0 + self.y1) // 2
        children = [
            _QTNode(self.x0, self.y0, mx, my),
            _QTNode(mx,      self.y0, self.x1, my),
            _QTNode(self.x0, my,      mx, self.y1),
            _QTNode(mx,      my,      self.x1, self.y1),
        ]
        for kp in self.kps:
            x, y = kp.pt
            for c in children:
                if c.x0 <= x < c.x1 and c.y0 <= y < c.y1:
                    c.kps.append(kp)
                    break
        # Mark leaves that already can't split usefully
        live = []
        for c in children:
            if len(c.kps) == 0:
                continue
            if len(c.kps) == 1:
                c.no_more = True
            live.append(c)
        return live


def _distribute_quadtree(kps, img_w, img_h, target_n):
    """
    ORB-SLAM-style quadtree distribution.

    Starts with one root node covering the image (or several side-by-side
    when the image is much wider than it is tall, as ORB-SLAM does), then
    repeatedly splits the node containing the most keypoints until the
    total leaf count >= target_n. From each leaf, keep the keypoint with
    the highest FAST response.
    """
    if not kps:
        return []

    # Initialize: one root per ~square region across image width.
    # (ORB-SLAM does the same — wide images get multiple roots so the
    # initial aspect ratio is closer to 1:1.)
    n_roots = max(1, round(img_w / img_h))
    roots = []
    step = img_w / n_roots
    for i in range(n_roots):
        roots.append(_QTNode(int(i * step), 0, int((i + 1) * step), img_h))

    # Distribute keypoints into roots
    for kp in kps:
        x, y = kp.pt
        for r in roots:
            if r.x0 <= x < r.x1 and r.y0 <= y < r.y1:
                r.kps.append(kp)
                break

    nodes = [r for r in roots if r.kps]

    # Split until we have at least target_n leaves or nothing left to split.
    # Each iteration: pick the node with most kps (and not flagged no_more),
    # split it, replace it with its non-empty children.
    while len(nodes) < target_n:
        # Find node with most kps that can still split
        splittable = [n for n in nodes if not n.no_more and len(n.kps) > 1]
        if not splittable:
            break
        # Split all currently-splittable nodes at once (matches ORB-SLAM's
        # batched expansion — much faster than splitting one at a time).
        splittable.sort(key=lambda n: len(n.kps), reverse=True)
        new_nodes = [n for n in nodes if n not in splittable]
        for node in splittable:
            new_nodes.extend(node.split())
            if len(new_nodes) >= target_n:
                # If splitting this one already overshot, stop here.
                # Adding remaining unsplit nodes back keeps fairness.
                idx = splittable.index(node)
                new_nodes.extend([n for n in splittable[idx + 1:]
                                  if n not in new_nodes])
                break
        if len(new_nodes) == len(nodes):
            break  # no progress
        nodes = new_nodes

    # From each leaf, pick the keypoint with the highest FAST response.
    selected = []
    for n in nodes:
        if not n.kps:
            continue
        best = max(n.kps, key=lambda k: k.response)
        selected.append(best)
    return selected


def detect_orb_quadtree(orb, gray, target_n=None):
    """
    ORB-SLAM-style pyramid + per-cell FAST + quadtree distribution.

    Builds a Gaussian pyramid manually, runs FAST per cell on each level
    (with low-threshold fallback for cells that come up empty — exactly
    what ORB-SLAM does for low-contrast regions), then runs the quadtree
    distribution per level to enforce spatial spread. Descriptors are
    finally computed on the gathered keypoints.

    Returns (keypoints, descriptors).
    """
    if target_n is None:
        target_n = ORB_PARAMS["nfeatures"]

    nlevels = ORB_PARAMS["nlevels"]
    scale_factor = ORB_PARAMS["scaleFactor"]
    cell_px = ORB_QT_CELL_PX
    edge = ORB_PARAMS["edgeThreshold"]

    # Per-level target — ORB-SLAM allocates more features to lower (larger)
    # pyramid levels. Use geometric distribution that sums to target_n.
    factor = 1.0 / scale_factor
    n_desired = []
    n_desired_per_level_unscaled = target_n * (1 - factor) / (1 - factor ** nlevels)
    cum = 0
    for lvl in range(nlevels - 1):
        n = round(n_desired_per_level_unscaled * (factor ** lvl))
        n_desired.append(n)
        cum += n
    n_desired.append(max(1, target_n - cum))

    # Build pyramid
    pyramid = [gray]
    for _ in range(1, nlevels):
        prev = pyramid[-1]
        new_w = max(2 * edge + 2, int(round(prev.shape[1] / scale_factor)))
        new_h = max(2 * edge + 2, int(round(prev.shape[0] / scale_factor)))
        pyramid.append(cv2.resize(prev, (new_w, new_h),
                                  interpolation=cv2.INTER_LINEAR))

    all_kp = []
    fast_high = cv2.FastFeatureDetector_create(threshold=ORB_QT_FAST_HIGH,
                                               nonmaxSuppression=True)
    fast_low = cv2.FastFeatureDetector_create(threshold=ORB_QT_FAST_LOW,
                                              nonmaxSuppression=True)

    for lvl, img in enumerate(pyramid):
        h, w = img.shape
        # Inner region we're allowed to detect in (avoid ORB's edge band)
        x_min, y_min = edge, edge
        x_max, y_max = w - edge, h - edge
        if x_max <= x_min or y_max <= y_min:
            continue

        # Per-cell FAST with fallback
        cell_kps = []
        y = y_min
        while y < y_max:
            y2 = min(y + cell_px, y_max)
            x = x_min
            while x < x_max:
                x2 = min(x + cell_px, x_max)
                cell = img[y:y2, x:x2]
                kps = fast_high.detect(cell, None)
                if not kps:
                    kps = fast_low.detect(cell, None)
                for kp in kps:
                    # Translate from cell coords to level coords
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                cell_kps.extend(kps)
                x = x2
            y = y2

        if not cell_kps:
            continue

        # Distribute via quadtree at this level
        selected = _distribute_quadtree(cell_kps,
                                        x_max - x_min, y_max - y_min,
                                        n_desired[lvl])

        # Scale keypoints back to level-0 coordinates and stamp level/size
        scale = scale_factor ** lvl
        patch_size_at_level = 31 * scale
        for kp in selected:
            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
            kp.octave = lvl
            kp.size = patch_size_at_level
        all_kp.extend(selected)

    if not all_kp:
        return [], None

    # Compute descriptors at the gathered keypoints on the level-0 image.
    # OpenCV's ORB.compute respects kp.octave when computing BRIEF on the
    # appropriate pyramid level internally.
    all_kp, des = orb.compute(gray, all_kp)
    return all_kp, des


def detect_and_match_orb(orb, bf_knn, prev_gray, curr_gray,
                         ratio=0.75, displacement_mad_mult=5.0,
                         use_quadtree=False):
    """
    Detect ORB on both frames, match descriptors with Lowe's ratio
    test, then apply a spatial outlier filter on the displacement field.

    When use_quadtree=True, uses the ORB-SLAM-style detector
    (per-cell FAST over a pyramid + quadtree spatial distribution).
    Otherwise uses the simpler gridded detector.

    Returns (prev_pts, curr_pts, desc_dist, n_kp_prev, n_kp_curr).
    """
    if use_quadtree:
        kp1, des1 = detect_orb_quadtree(orb, prev_gray)
        kp2, des2 = detect_orb_quadtree(orb, curr_gray)
    else:
        kp1, des1 = detect_orb_gridded(orb, prev_gray)
        kp2, des2 = detect_orb_gridded(orb, curr_gray)

    empty = np.empty((0, 2), dtype=np.float32)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return empty, empty, np.empty((0,), dtype=np.float32), len(kp1), len(kp2)

    # Lowe's ratio test: a match survives only if its best partner is
    # meaningfully closer than its second-best. This is what kills the
    # "two patches look equally similar so the matcher picks the wrong one"
    # failure mode that produces the long-line outliers on textured rock.
    knn = bf_knn.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return empty, empty, np.empty((0,), dtype=np.float32), len(kp1), len(kp2)

    prev_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    curr_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    desc_dist = np.array([m.distance for m in good], dtype=np.float32)

    # Spatial outlier filter: reject matches whose displacement magnitude
    # is far from the median (robust MAD-based threshold). Catches the
    # remaining cross-image false matches that survive the ratio test.
    if len(prev_pts) >= 5:
        disp = np.linalg.norm(curr_pts - prev_pts, axis=1)
        med = np.median(disp)
        mad = np.median(np.abs(disp - med)) + 1e-6
        keep = np.abs(disp - med) < displacement_mad_mult * mad
        prev_pts = prev_pts[keep]
        curr_pts = curr_pts[keep]
        desc_dist = desc_dist[keep]

    return prev_pts, curr_pts, desc_dist, len(kp1), len(kp2)


# ----------------------------------------------------------------------
# Quality metrics
# ----------------------------------------------------------------------

def epipolar_inlier_ratio(p1: np.ndarray, p2: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Fit a fundamental matrix with RANSAC and return (inlier_ratio, inlier_mask).
    Returns (0.0, all-False mask) if not enough points or fit fails.

    Degeneracy guards (any of these → bail out, since findFundamentalMat will
    either crash or produce garbage):
      - fewer than MIN_MATCHES_FOR_F points
      - points spread less than ~5 px in either axis (near-collinear / clustered)
      - many duplicate points (after rounding)
    """
    n = len(p1)
    if n < MIN_MATCHES_FOR_F:
        return 0.0, np.zeros(n, dtype=bool)

    # Ensure contiguous float32, shape (N, 1, 2) — OpenCV is picky about this
    # and some of its internal asserts blow up on odd strides / shapes.
    p1f = np.ascontiguousarray(p1, dtype=np.float32).reshape(-1, 1, 2)
    p2f = np.ascontiguousarray(p2, dtype=np.float32).reshape(-1, 1, 2)

    # Degeneracy check 1: insufficient spatial spread.
    spread1 = np.ptp(p1f.reshape(-1, 2), axis=0)  # [dx, dy]
    spread2 = np.ptp(p2f.reshape(-1, 2), axis=0)
    if spread1.min() < 5.0 or spread2.min() < 5.0:
        return 0.0, np.zeros(n, dtype=bool)

    # Degeneracy check 2: too many duplicates (rounded to int pixel).
    uniq1 = len(np.unique(p1f.reshape(-1, 2).round().astype(np.int32), axis=0))
    if uniq1 < MIN_MATCHES_FOR_F:
        return 0.0, np.zeros(n, dtype=bool)

    try:
        F, mask = cv2.findFundamentalMat(
            p1f, p2f, cv2.FM_RANSAC, RANSAC_REPROJ_THRESHOLD, 0.99
        )
    except cv2.error:
        # RANSAC occasionally hits internal asserts on pathological inputs.
        # Treat as a failed fit rather than crashing the whole run.
        return 0.0, np.zeros(n, dtype=bool)

    if F is None or mask is None or F.shape != (3, 3):
        return 0.0, np.zeros(n, dtype=bool)
    mask = mask.flatten().astype(bool)
    return mask.sum() / n, mask


def motion_consistency(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Bulk motion consistency: fraction of flow vectors whose direction
    is within 45 degrees of the median flow direction. A coherent scene
    motion will be close to 1.0; noise approaches 0.5.
    """
    if len(p1) < 5:
        return 0.0
    flow = p2 - p1
    mags = np.linalg.norm(flow, axis=1)
    # Ignore near-zero flow (camera nearly still) — direction is meaningless.
    valid = mags > 0.5
    if valid.sum() < 5:
        return 0.0
    flow_v = flow[valid]
    # Median direction (use vector sum of unit vectors)
    units = flow_v / np.linalg.norm(flow_v, axis=1, keepdims=True)
    mean_dir = units.mean(axis=0)
    mean_dir /= (np.linalg.norm(mean_dir) + 1e-9)
    cos_angles = units @ mean_dir
    return float((cos_angles > np.cos(np.deg2rad(45))).mean())


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def draw_tracks(img_gray: np.ndarray,
                p_prev: np.ndarray,
                p_curr: np.ndarray,
                inlier_mask: np.ndarray,
                label: str,
                stats: dict) -> np.ndarray:
    """
    Render current frame with flow vectors.
    Green = epipolar inlier, red = outlier.
    Header turns red and shows "TRACKING UNRELIABLE" when stats['failed'] is True.
    """
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for i, (a, b) in enumerate(zip(p_prev, p_curr)):
        color = (0, 200, 0) if i < len(inlier_mask) and inlier_mask[i] else (0, 0, 200)
        a_i = tuple(np.round(a).astype(int))
        b_i = tuple(np.round(b).astype(int))
        cv2.line(vis, a_i, b_i, color, 1, cv2.LINE_AA)
        cv2.circle(vis, b_i, 2, color, -1, cv2.LINE_AA)

    h, w = vis.shape[:2]
    failed = stats.get("failed", False)
    bar_color = (0, 0, 90) if failed else (0, 0, 0)  # dark red bar when failed
    bar = np.full((60, w, 3), bar_color, dtype=np.uint8)

    txt1 = f"{label}  tracked={stats['tracked']}  inliers={stats['inlier_ratio']:.2f}"
    txt2 = f"motion_consistency={stats['motion']:.2f}"
    if failed:
        txt2 += "   [TRACKING UNRELIABLE]"
    text_color = (200, 200, 255) if failed else (255, 255, 255)
    cv2.putText(bar, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    cv2.putText(bar, txt2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

    # Also draw a red border around the frame when failed, so it's visible
    # at a glance when scrubbing the comparison video.
    if failed:
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)

    return np.vstack([bar, vis])


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Folder of image sequence")
    ap.add_argument("--output", type=Path, default=Path("./results"),
                    help="Output folder (default: ./results)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process at most N frames (for quick tests)")
    ap.add_argument("--clahe", action="store_true",
                    help="Apply CLAHE preprocessing (helps with turbid water)")
    ap.add_argument("--orb-quadtree", action="store_true",
                    help="Use ORB-SLAM-style quadtree feature distribution "
                         "(per-cell FAST + spatial quadtree). Slower but more "
                         "spatially uniform than plain gridded detection.")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Output video FPS (default: 10)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    paths = load_image_paths(args.folder)
    if args.max_frames is not None:
        paths = paths[: args.max_frames]

    # Setup
    orb = get_orb()
    # knnMatch needs crossCheck=False (the two are mutually exclusive in OpenCV).
    # We get the cross-check behavior more flexibly via Lowe's ratio test.
    bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    # Prime first frame
    prev_gray = read_gray(paths[0])
    if clahe is not None:
        prev_gray = clahe.apply(prev_gray)

    h, w = prev_gray.shape
    # Side-by-side: width = 2*w, height = h + 60 (header bar)
    video_path = args.output / "comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (2 * w, h + 60))
    if not writer.isOpened():
        sys.exit(f"Could not open video writer at {video_path}")

    # Metrics CSV
    csv_path = args.output / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame_idx", "filename", "open_water",
        # Shi-Tomasi + KLT
        "st_detected", "st_tracked", "st_fb_good", "st_inliers", "st_inlier_ratio",
        "st_motion", "st_mean_fb_err",
        # ORB
        "orb_kp_prev", "orb_kp_curr", "orb_matched", "orb_inliers", "orb_inlier_ratio",
        "orb_motion", "orb_mean_desc_dist",
    ])

    metrics_log = []

    for idx in range(1, len(paths)):
        curr_gray = read_gray(paths[idx])
        if clahe is not None:
            curr_gray = clahe.apply(curr_gray)

        # ----- Shi-Tomasi + KLT -----
        st_pts = detect_shi_tomasi(prev_gray)
        st_detected = len(st_pts)
        st_prev, st_curr, st_fb_err = track_klt(prev_gray, curr_gray, st_pts)
        st_tracked = len(st_prev)
        # Filter by forward-backward error
        st_fb_mask = st_fb_err < FB_ERROR_THRESHOLD
        st_prev_g = st_prev[st_fb_mask]
        st_curr_g = st_curr[st_fb_mask]
        st_fb_good = len(st_prev_g)
        st_ratio, st_inl_mask = epipolar_inlier_ratio(st_prev_g, st_curr_g)
        st_inliers = int(st_inl_mask.sum())
        st_motion = motion_consistency(st_prev_g, st_curr_g)
        st_mean_fb = float(st_fb_err.mean()) if len(st_fb_err) else 0.0

        # ----- ORB + BFMatcher -----
        orb_prev, orb_curr, orb_desc_dist, kp1_n, kp2_n = detect_and_match_orb(
            orb, bf_knn, prev_gray, curr_gray, use_quadtree=args.orb_quadtree
        )
        orb_matched = len(orb_prev)
        orb_ratio, orb_inl_mask = epipolar_inlier_ratio(orb_prev, orb_curr)
        orb_inliers = int(orb_inl_mask.sum())
        orb_motion = motion_consistency(orb_prev, orb_curr)
        orb_mean_dd = float(orb_desc_dist.mean()) if len(orb_desc_dist) else 0.0

        # ----- Open-water classification -----
        # Heuristic: both trackers struggling AND incoherent motion = no scene.
        open_water = (
            st_fb_good < OPEN_WATER_MAX_FEATURES
            and orb_matched < OPEN_WATER_MAX_FEATURES
            and st_motion < OPEN_WATER_MAX_MOTION
            and orb_motion < OPEN_WATER_MAX_MOTION
        )

        # ----- Log CSV -----
        csv_w.writerow([
            idx, paths[idx].name, int(open_water),
            st_detected, st_tracked, st_fb_good, st_inliers, f"{st_ratio:.4f}",
            f"{st_motion:.4f}", f"{st_mean_fb:.4f}",
            kp1_n, kp2_n, orb_matched, orb_inliers, f"{orb_ratio:.4f}",
            f"{orb_motion:.4f}", f"{orb_mean_dd:.4f}",
        ])
        metrics_log.append(dict(
            idx=idx, open_water=open_water,
            st_tracked=st_fb_good, st_inliers=st_inliers, st_ratio=st_ratio, st_motion=st_motion,
            orb_matched=orb_matched, orb_inliers=orb_inliers, orb_ratio=orb_ratio, orb_motion=orb_motion,
        ))

        # ----- Visualize -----
        st_vis = draw_tracks(
            curr_gray, st_prev_g, st_curr_g, st_inl_mask,
            "Shi-Tomasi + KLT",
            dict(tracked=st_fb_good, inlier_ratio=st_ratio, motion=st_motion,
                 failed=open_water),
        )
        orb_label = "ORB (quadtree)" if args.orb_quadtree else "ORB (gridded)"
        orb_vis = draw_tracks(
            curr_gray, orb_prev, orb_curr, orb_inl_mask,
            orb_label,
            dict(tracked=orb_matched, inlier_ratio=orb_ratio, motion=orb_motion,
                 failed=open_water),
        )
        combined = np.hstack([st_vis, orb_vis])
        writer.write(combined)

        if idx % 10 == 0 or idx == len(paths) - 1:
            print(f"  frame {idx}/{len(paths)-1}  "
                  f"KLT(tracked={st_fb_good}, inl={st_ratio:.2f}, mot={st_motion:.2f})  "
                  f"ORB(matched={orb_matched}, inl={orb_ratio:.2f}, mot={orb_motion:.2f})")

        prev_gray = curr_gray

    writer.release()
    csv_f.close()

    # ----- Summary plots -----
    if metrics_log:
        idxs = [m["idx"] for m in metrics_log]
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(idxs, [m["st_tracked"] for m in metrics_log], label="KLT tracked (post-FB)", color="C0")
        axes[0].plot(idxs, [m["orb_matched"] for m in metrics_log], label="ORB matched", color="C1")
        axes[0].set_ylabel("# features")
        axes[0].set_title("Feature count per frame")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(idxs, [m["st_ratio"] for m in metrics_log], label="KLT inlier ratio", color="C0")
        axes[1].plot(idxs, [m["orb_ratio"] for m in metrics_log], label="ORB inlier ratio", color="C1")
        axes[1].set_ylabel("RANSAC inlier ratio")
        axes[1].set_title("Geometric (epipolar) consistency")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(idxs, [m["st_motion"] for m in metrics_log], label="KLT motion consistency", color="C0")
        axes[2].plot(idxs, [m["orb_motion"] for m in metrics_log], label="ORB motion consistency", color="C1")
        axes[2].set_ylabel("Direction agreement")
        axes[2].set_xlabel("frame index")
        axes[2].set_title("Bulk motion consistency (fraction within 45° of median flow)")
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        # Shade open-water segments across all three subplots so it's
        # visually obvious where tracking is unreliable and shouldn't be
        # included in the head-to-head comparison.
        ow_flags = np.array([m["open_water"] for m in metrics_log], dtype=bool)
        if ow_flags.any():
            # Find contiguous True runs
            edges = np.diff(ow_flags.astype(int))
            starts = np.where(edges == 1)[0] + 1
            ends = np.where(edges == -1)[0] + 1
            if ow_flags[0]:
                starts = np.insert(starts, 0, 0)
            if ow_flags[-1]:
                ends = np.append(ends, len(ow_flags))
            for ax in axes:
                for s, e in zip(starts, ends):
                    ax.axvspan(idxs[s], idxs[min(e, len(idxs)-1)],
                               color="red", alpha=0.10,
                               label="_nolegend_")
            # Add one labeled patch to the top legend
            from matplotlib.patches import Patch
            handles, labels = axes[0].get_legend_handles_labels()
            handles.append(Patch(color="red", alpha=0.10, label="Open water (flagged)"))
            axes[0].legend(handles=handles)

        fig.tight_layout()
        plot_path = args.output / "summary.png"
        fig.savefig(plot_path, dpi=120)
        print(f"\nSaved plot: {plot_path}")

        # Summary table — also report stats excluding open-water frames,
        # since those shouldn't be held against either tracker.
        def avg(key, mask=None):
            vals = np.array([m[key] for m in metrics_log])
            if mask is not None:
                vals = vals[mask]
            return vals.mean() if len(vals) else 0.0

        ow_frac = ow_flags.mean() if len(ow_flags) else 0.0
        valid = ~ow_flags

        print("\n=== Summary (mean across ALL frames) ===")
        print(f"  Shi-Tomasi + KLT:  tracked={avg('st_tracked'):.1f}  "
              f"inlier_ratio={avg('st_ratio'):.3f}  motion_consistency={avg('st_motion'):.3f}")
        print(f"  ORB + BFMatcher:   matched={avg('orb_matched'):.1f}  "
              f"inlier_ratio={avg('orb_ratio'):.3f}  motion_consistency={avg('orb_motion'):.3f}")
        print(f"\n  Open-water frames: {ow_flags.sum()} / {len(ow_flags)} "
              f"({100*ow_frac:.1f}%)")
        if valid.any():
            print("\n=== Summary (excluding open-water frames) ===")
            print(f"  Shi-Tomasi + KLT:  tracked={avg('st_tracked', valid):.1f}  "
                  f"inlier_ratio={avg('st_ratio', valid):.3f}  motion_consistency={avg('st_motion', valid):.3f}")
            print(f"  ORB + BFMatcher:   matched={avg('orb_matched', valid):.1f}  "
                  f"inlier_ratio={avg('orb_ratio', valid):.3f}  motion_consistency={avg('orb_motion', valid):.3f}")

    print(f"\nSaved video:   {video_path}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
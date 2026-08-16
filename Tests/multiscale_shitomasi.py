"""
multiscale_shitomasi.py

Shi-Tomasi detection across an ORB-SLAM3-style image pyramid, with an
explicit octave label per keypoint — the missing piece for porting the
hybrid frontend into ORB-SLAM3 (§9.1).

WHY OCTAVES MATTER (and why level-0-only detection is not enough):
  - ORB-SLAM3 computes BRIEF on the pyramid level given by kp.octave.
    Detect everything at level 0 and every descriptor comes from one
    scale, while DBoW2's vocabulary was trained on multi-scale
    descriptors — a plausible source of vocabulary mismatch, and a
    direct problem for revisits from a different standoff distance.
  - TrackLocalMap's search radius is r_TLM(l) = 2 * 1.2^l.
  - Bundle adjustment weights observations by invLevelSigma2[octave]:
    features found at coarse levels are less precisely localised and
    must be trusted less.

TWO PYRAMIDS, DELIBERATELY DIFFERENT (see also §4.2 vs §8):
  This module builds the DESCRIPTOR pyramid at scale factor 1.2, matching
  ORB-SLAM3. That is NOT required to match the KLT tracking pyramid
  (factor 2.0). They serve unrelated purposes: cv2.calcOpticalFlowPyrLK
  consumes and returns level-0 coordinates, using its pyramid purely as
  internal coarse-to-fine scratch space, whereas the octave label
  describes a feature's characteristic scale for descriptors, search
  radii, and BA weighting. Forcing KLT onto a 1.2 pyramid would need ~8
  levels to reach the displacement coverage 3 levels of a factor-2
  pyramid gives (level 3 at 1.2 is only 1.7x downsampled), which would
  badly weaken large-motion tracking for no benefit.

THRESHOLD RULES (the open question this module is built to settle):
  'absolute'  one fixed lambda2 threshold for every level. Fails if
              responses really do decay with level.
  'relative'  per-level threshold = quality * max(lambda2) at that
              level. OpenCV's goodFeaturesToTrack convention, applied
              per octave. Adapts to decay, but normalises by a
              SINGLE-PIXEL outlier: one specular glint sets the
              threshold for the whole octave.
  'rank'      per-level target count (ORB-SLAM3's geometric allocation)
              + quadtree selection by lambda2 rank, with a low absolute
              floor only to cull noise. This is what ORB-SLAM3 actually
              does for FAST, and it is insensitive to the absolute
              scale of the responses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from feature_comparison import _distribute_quadtree  # noqa: E402

DEFAULT_SCALE_FACTOR = 1.2
DEFAULT_NLEVELS = 8
DEFAULT_EDGE = 15
DEFAULT_BLOCK_SIZE = 7
DEFAULT_PATCH_SIZE = 31


def build_pyramid(gray: np.ndarray,
                  scale_factor: float = DEFAULT_SCALE_FACTOR,
                  nlevels: int = DEFAULT_NLEVELS,
                  edge: int = DEFAULT_EDGE) -> list[np.ndarray]:
    """ORB-SLAM3-style pyramid: successive resize by 1/scale_factor.

    Uses INTER_LINEAR without an explicit Gaussian pre-blur, matching
    ORB-SLAM3's ComputePyramid (cv::resize already low-pass filters).
    """
    pyr = [gray]
    for _ in range(1, nlevels):
        prev = pyr[-1]
        w = max(2 * edge + 2, int(round(prev.shape[1] / scale_factor)))
        h = max(2 * edge + 2, int(round(prev.shape[0] / scale_factor)))
        pyr.append(cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR))
    return pyr


def lambda2_map(img: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE):
    """Minimum eigenvalue of the structure tensor at every pixel.

    This IS the Shi-Tomasi response ("Good Features to Track", §4.5.2).
    """
    return cv2.cornerMinEigenVal(img, block_size)


def level_target_counts(target_n: int, nlevels: int,
                        scale_factor: float) -> list[int]:
    """ORB-SLAM3's geometric per-level feature allocation.

    Lower (larger) levels get proportionally more features; the counts
    sum to target_n.
    """
    factor = 1.0 / scale_factor
    unscaled = target_n * (1 - factor) / (1 - factor ** nlevels)
    counts, cum = [], 0
    for lvl in range(nlevels - 1):
        n = int(round(unscaled * (factor ** lvl)))
        counts.append(n)
        cum += n
    counts.append(max(1, target_n - cum))
    return counts


def detect_multiscale_shi_tomasi(
        gray: np.ndarray,
        target_n: int = 1000,
        rule: str = "rank",
        quality: float = 0.01,
        absolute_threshold: float | None = None,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        nlevels: int = DEFAULT_NLEVELS,
        edge: int = DEFAULT_EDGE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        nms_radius: int = 3,
        return_stats: bool = False,
):
    """Detect Shi-Tomasi corners across the pyramid, one octave label each.

    Returns a list of cv2.KeyPoint in LEVEL-0 coordinates with .octave
    set, .size set to the patch size scaled to the level (so ORB.compute
    describes at the right scale), and .response = lambda2.

    With return_stats=True also returns a per-level diagnostic dict.
    """
    if rule not in ("absolute", "relative", "rank"):
        raise ValueError(f"unknown rule {rule!r}")

    pyr = build_pyramid(gray, scale_factor, nlevels, edge)
    targets = level_target_counts(target_n, nlevels, scale_factor)
    all_kps: list[cv2.KeyPoint] = []
    stats = []

    for lvl, img in enumerate(pyr):
        h, w = img.shape
        if w <= 2 * edge or h <= 2 * edge:
            stats.append({"level": lvl, "n_candidates": 0, "n_kept": 0,
                          "lam_max": 0.0, "lam_p99": 0.0, "lam_median": 0.0,
                          "threshold": 0.0})
            continue

        lam = lambda2_map(img, block_size)
        inner = lam[edge:h - edge, edge:w - edge]
        lam_max = float(inner.max()) if inner.size else 0.0
        lam_p99 = float(np.percentile(inner, 99)) if inner.size else 0.0
        lam_med = float(np.median(inner)) if inner.size else 0.0

        # Non-maximum suppression so a single blob doesn't yield hundreds
        # of adjacent "corners".
        k = 2 * nms_radius + 1
        local_max = cv2.dilate(lam, np.ones((k, k), np.uint8))
        is_peak = (lam >= local_max)

        if rule == "absolute":
            # One threshold for every level. If responses decay with
            # level, high octaves starve — that is exactly the failure
            # mode this rule exists to expose.
            thr = (absolute_threshold if absolute_threshold is not None
                   else quality * float(lambda2_map(pyr[0], block_size).max()))
        elif rule == "relative":
            # Per-level, normalised by that level's single largest
            # response. Adapts to decay but is outlier-sensitive.
            thr = quality * lam_max
        else:  # rank
            # Low floor only; the per-level target + quadtree does the
            # real selection, so absolute response scale is irrelevant.
            thr = 0.01 * quality * lam_max

        mask = is_peak & (lam > thr)
        mask[:edge, :] = False; mask[h - edge:, :] = False
        mask[:, :edge] = False; mask[:, w - edge:] = False
        ys, xs = np.nonzero(mask)
        n_cand = len(xs)

        if n_cand:
            resp = lam[ys, xs]
            kps = [cv2.KeyPoint(x=float(x), y=float(y), size=7,
                                response=float(r))
                   for x, y, r in zip(xs, ys, resp)]
            if rule == "rank" and len(kps) > targets[lvl]:
                # ORB-SLAM3's spatial distribution, ranked by lambda2.
                kps = _distribute_quadtree(kps, w - 2 * edge, h - 2 * edge,
                                           targets[lvl])
            elif len(kps) > targets[lvl]:
                # Even for the threshold rules, cap per level so the
                # comparison between rules is about WHERE features come
                # from, not how many.
                kps.sort(key=lambda k_: -k_.response)
                kps = kps[:targets[lvl]]
        else:
            kps = []

        # Lift to level-0 coordinates and stamp octave + patch size.
        s = scale_factor ** lvl
        for kp in kps:
            kp.pt = (kp.pt[0] * s, kp.pt[1] * s)
            kp.octave = lvl
            kp.size = DEFAULT_PATCH_SIZE * s
        all_kps.extend(kps)

        stats.append({"level": lvl, "n_candidates": n_cand,
                      "n_kept": len(kps), "lam_max": lam_max,
                      "lam_p99": lam_p99, "lam_median": lam_med,
                      "threshold": float(thr)})

    if return_stats:
        return all_kps, stats
    return all_kps


def describe(gray: np.ndarray, kps, orb=None):
    """Compute steered BRIEF at the given keypoints.

    ORB.compute honours kp.octave and kp.size, so descriptors are taken
    at the scale the feature was detected at — which is the entire point
    of carrying an octave label.
    """
    if orb is None:
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=DEFAULT_SCALE_FACTOR,
                             nlevels=DEFAULT_NLEVELS,
                             edgeThreshold=DEFAULT_EDGE)
    if not kps:
        return [], None
    kept, desc = orb.compute(gray, list(kps))
    return list(kept), desc
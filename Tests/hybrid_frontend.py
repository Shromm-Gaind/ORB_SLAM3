"""
hybrid_frontend.py

Per-frame pipeline implementing Steps 1-5 of the hybrid frontend
(design doc §4). Built on top of the existing feature_comparison.py
infrastructure for Shi-Tomasi detection and the quadtree distribution,
plus the standalone DormantTrackBuffer and SpatialDescriptorMatch
primitives.

POC scope (per §9.2 / "Step 2"):
  - monocular only (no stereo depth — §4.5.6 skipped)
  - no backend, hence no Step 5b TrackLocalMap and no map points
  - all tracks remain "infants" (no triangulation), which means every
    KLT failure enters the dormant buffer and is eligible for Step 5
    re-ID. This is the right thing for the precision/recall test.

Per-frame state:
  active_tracks: dict[int, ActiveTrack]   # id -> track
  dormant_buffer: DormantTrackBuffer
  frame_index: int
  next_track_id: int
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reuse the user's existing quadtree distribution (faithful port of
# ORB-SLAM3's spatial distribution). Importing rather than copying so
# any future fixes to feature_comparison.py propagate here.
# BUGFIX: this used to insert a hard-coded "/mnt/user-data/uploads"
# ahead of everything, which silently shadowed the sibling modules
# (dormant_buffer, spatial_descriptor_matcher, feature_comparison) with
# stale copies from a different directory. Import siblings from THIS
# file's directory.
sys.path.insert(0, str(Path(__file__).parent))
from feature_comparison import _distribute_quadtree  # noqa: E402

from dormant_buffer import DormantTrack, DormantTrackBuffer  # noqa: E402
from spatial_descriptor_matcher import (  # noqa: E402
    MatchOptions,
    PixelCandidate,
    PixelQuery,
    spatial_descriptor_match,
)


# --------------------------------------------------------------------
# Config — defaults match §8 of the design doc.
# --------------------------------------------------------------------

@dataclass
class HybridConfig:
    # Step 1 — KLT
    klt_window: tuple[int, int] = (21, 21)
    klt_pyramid_levels: int = 3
    klt_max_iter: int = 30
    klt_eps: float = 0.01

    # Step 2 — Forward-backward consistency
    fb_threshold_px: float = 1.0

    # Step 3 — RANSAC fundamental matrix
    ransac_reproj_px: float = 3.0
    min_matches_for_F: int = 8

    # Step 4 — Shi-Tomasi top-up
    target_active_tracks: int = 1000      # N_target
    shi_tomasi_quality: float = 0.01      # θ_ST proxy (OpenCV scale)
    shi_tomasi_min_distance: int = 7
    shi_tomasi_block_size: int = 7
    occupancy_mask_radius: int = 10       # r_mask, anti-clumping
    descriptor_patch_size: int = 31       # for ORB.compute steered BRIEF

    # Step 5 — Re-ID against dormant buffer
    dormant_horizon_frames: int = 30      # Δ_dormant
    reid_radius_px: float = 20.0          # r_reid
    reid_hamming_threshold: int = 32      # θ_reid
    # Distinctiveness gate: best match must beat the second-best spatial
    # candidate by this many Hamming bits (0 disables). See
    # MatchOptions.second_best_margin.
    reid_second_best_margin: int = 10
    # Motion-compensate dormant predicted positions each frame using the
    # median KLT flow of RANSAC-inlier survivors (§4.6 "optionally
    # propagated by the current motion model"). Lets r_reid be tightened.
    motion_compensate_dormant: bool = True
    # A naturally-dying track must have survived at least this many
    # frames to be worth buffering for re-ID. Short-lived tracks are
    # mostly detector noise; buffering them (especially in bulk during
    # low-quality sections) floods the buffer with impostors. Does NOT
    # apply to force_kill(), which buffers unconditionally (the forced-
    # failure harness filters its kill sample by age instead).
    dormant_min_track_age: int = 3
    # Bias Step 4's corner selection toward dormant predicted positions:
    # detected corners falling within reid_radius_px of a dormant entry
    # are kept preferentially (up to the deficit) before quadtree
    # distribution fills the rest. Without this, a small deficit spread
    # over the whole image rarely re-detects a recently-died landmark's
    # spot, so Step 5 never gets a candidate to match ("missed" for
    # detection reasons, not matching reasons).
    seed_corners_near_dormant: bool = True

    # Tracking-lost threshold (§8 N_min)
    min_active_tracks: int = 50

    # Diagnostics — gather per-query spatial candidate counts, per-match
    # Hamming distances, and resurrected-entry ages. Cost is dominated by
    # numpy per-frame work, modest even with large dormant buffers.
    collect_diagnostics: bool = True


# --------------------------------------------------------------------
# Active track state
# --------------------------------------------------------------------

@dataclass
class ActiveTrack:
    id: int
    x: float
    y: float
    birth_descriptor: np.ndarray   # uint8 (32,)
    octave: int = 0
    age: int = 0
    map_point: object = None       # always None in this POC (no backend)
    # Diagnostics only (collect_diagnostics=True): the fresh descriptor
    # computed at this track's position on the previous frame, used to
    # measure 1-frame same-track drift (the distribution θ_reid should
    # be calibrated against, now that dormant entries store death-time
    # descriptors and dormancy gaps are 1-2 frames).
    diag_prev_descriptor: Optional[np.ndarray] = None


# --------------------------------------------------------------------
# Per-frame result struct
# --------------------------------------------------------------------

@dataclass
class FrameResult:
    """What happened on this frame. Returned by process_frame()."""
    frame_index: int
    # Step 1-3 outcomes
    tracks_in: int                 # active tracks at start of frame
    tracks_after_klt: int          # survived KLT
    tracks_after_fb: int           # survived FB filter
    tracks_after_ransac: int       # survived geometric filter
    # Step 4 outcomes
    new_corners_detected: int      # Shi-Tomasi candidates after mask+quadtree
    # Step 5 outcomes
    reids_attempted: int           # candidates searched against dormant buffer
    reids_succeeded: int           # candidates that matched something
    # Bookkeeping
    tracks_out: int                # active tracks at end of frame
    dormant_buffer_size: int

    # Optional: which tracks died this frame (used by the forced-failure
    # test to know what *should* be resurrectable). List of (id, last_x, last_y).
    died_this_frame: list[tuple[int, float, float]] = field(default_factory=list)
    # Optional: which tracks got resurrected this frame, list of resurrected ids.
    resurrected_ids: list[int] = field(default_factory=list)

    # ---- Step 5 diagnostics (populated only when verbose stats are on) ----
    # How many dormant entries fall inside each query's spatial gate.
    # One entry per Step 5 query (= per new corner). Tells us whether the
    # spatial filter is doing real work or whether the matcher is being
    # asked to discriminate among many candidates per query.
    spatial_candidates_per_query: list[int] = field(default_factory=list)
    # Hamming distance of each accepted match. Mostly low (0-15) = good,
    # mostly near threshold = matcher is sneaking through noise.
    accepted_hamming_distances: list[int] = field(default_factory=list)
    # Age in frames of resurrected dormant entries. Mostly 1-3 = strong
    # signal, mostly 25-30 = probably stale-matching noise.
    resurrected_ages: list[int] = field(default_factory=list)
    # Hamming distance between a surviving track's birth descriptor and
    # its descriptor recomputed at the new (KLT-tracked, FB+RANSAC-verified)
    # position. This is the descriptor's intra-feature stability — a
    # ground-truth distribution for what "Hamming distance for the same
    # physical point" looks like on this footage. The matcher's accepted-
    # Hamming distribution should ideally lie at or below this one.
    same_track_hamming_distances: list[int] = field(default_factory=list)
    # Same idea, but between the descriptors recomputed on consecutive
    # frames (1-frame gap) for the same surviving track. Since dormant
    # entries now store the death-time descriptor and dormancy gaps are
    # ~1-2 frames, THIS is the distribution θ_reid should be calibrated
    # against (e.g. p90 plus a small allowance).
    same_track_gap1_hamming_distances: list[int] = field(default_factory=list)

    # Dominant image motion this frame: median (dx, dy) of RANSAC-inlier
    # KLT flow. (0, 0) when too few survivors. Used for motion
    # compensation of dormant predictions, and by the forced-failure
    # harness to propagate expected resurrection locations.
    median_flow: tuple[float, float] = (0.0, 0.0)
    # Pixel locations of the new Shi-Tomasi corners detected in Step 4
    # (post mask + quadtree, pre re-ID). Lets the harness distinguish
    # "missed because no corner ever appeared there" from "missed
    # because the matcher rejected it".
    new_corner_positions: list[tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------
# Step 4 helpers — Shi-Tomasi + quadtree + steered BRIEF
# --------------------------------------------------------------------

def _build_occupancy_mask(img_shape: tuple[int, int],
                          active_tracks: dict[int, ActiveTrack],
                          radius: int) -> np.ndarray:
    """Build a uint8 mask: 0 in a square of `radius` around each existing
    track, 255 elsewhere. This is the M_k from §4.5.1.

    Note: OpenCV's goodFeaturesToTrack uses a *non-zero* convention for
    the mask (i.e. detect where mask != 0), so we return 255 for "free
    to detect" and 0 for "occupied".
    """
    h, w = img_shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for t in active_tracks.values():
        x0 = max(0, int(round(t.x)) - radius)
        x1 = min(w, int(round(t.x)) + radius + 1)
        y0 = max(0, int(round(t.y)) - radius)
        y1 = min(h, int(round(t.y)) + radius + 1)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = 0
    return mask


def _detect_shi_tomasi_with_quadtree(
        gray: np.ndarray, mask: np.ndarray, target_n: int,
        quality: float, min_distance: int, block_size: int,
        priority_map: Optional[np.ndarray] = None,
        pool_n: Optional[int] = None,
) -> list[cv2.KeyPoint]:
    """Detect Shi-Tomasi corners constrained by `mask`, then enforce
    spatial spread with the ORB-SLAM3 quadtree.

    The quadtree implementation in feature_comparison.py works on
    cv2.KeyPoint objects (which is what we want anyway, to feed to
    ORB.compute() in the next step).
    """
    # goodFeaturesToTrack keeps only the globally strongest maxCorners.
    # With a small deficit that means ~2*deficit corners image-wide, and
    # a killed landmark's (often mediocre-quality) corner rarely makes
    # the cut — so dormant seeding has nothing near the dormant entries
    # to prioritize. `pool_n` lets the caller widen the candidate pool
    # when dormant entries exist; total output is still capped at
    # target_n by the priority/quadtree selection below.
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=(pool_n if pool_n is not None else target_n * 2),
        qualityLevel=quality, minDistance=min_distance,
        blockSize=block_size, mask=mask,
    )
    if pts is None:
        return []
    h, w = gray.shape
    # goodFeaturesToTrack doesn't return a response, so use the
    # eigenvalue at each point. The Harris response would be slightly
    # different, but the eigenvalue (Shi-Tomasi's "goodness" criterion)
    # is the correct one for KLT trackability per §3.1.
    # cv2.cornerMinEigenVal gives lambda_2 at every pixel.
    eig = cv2.cornerMinEigenVal(gray, block_size)
    kps = []
    for p in pts.reshape(-1, 2):
        x, y = float(p[0]), float(p[1])
        # Sample lambda_2 at the (rounded) corner location.
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        kps.append(cv2.KeyPoint(x=x, y=y, size=7, response=float(eig[yi, xi])))
    # Priority selection: corners falling inside `priority_map` (True =
    # near a dormant predicted position) are kept first, best-response
    # first, before the quadtree fills the remainder from the rest.
    # Total is still capped at target_n, so the dark-section case (huge
    # dormant buffer covering the whole image) cannot blow up the count.
    if priority_map is not None and len(kps) > target_n:
        near, far = [], []
        for kp in kps:
            xi = int(np.clip(round(kp.pt[0]), 0, w - 1))
            yi = int(np.clip(round(kp.pt[1]), 0, h - 1))
            (near if priority_map[yi, xi] else far).append(kp)
        near.sort(key=lambda k: -k.response)
        keep = near[:target_n]
        remaining = target_n - len(keep)
        if remaining > 0:
            if len(far) > remaining:
                keep += _distribute_quadtree(far, w, h, remaining)
            else:
                keep += far
        return keep
    # Quadtree distribution (uses kp.pt and kp.response).
    if len(kps) > target_n:
        kps = _distribute_quadtree(kps, w, h, target_n)
    return kps


def _descriptors_at_positions(
        gray: np.ndarray, xy: list[tuple[float, float]],
        orb: cv2.ORB, patch_size: int,
) -> dict[int, np.ndarray]:
    """Compute steered-BRIEF descriptors at caller-specified positions.

    Returns {input_index: descriptor uint8 (32,)}. Positions whose patch
    falls off the image border are silently absent from the result (ORB
    drops them); callers must handle missing keys with a fallback.

    Alignment between input positions and returned rows uses the
    kp.class_id tag (ORB.compute preserves it through drops).
    """
    if not xy:
        return {}
    kps = []
    for i, (x, y) in enumerate(xy):
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=float(patch_size))
        kp.class_id = i
        kps.append(kp)
    kept, desc = orb.compute(gray, kps)
    out: dict[int, np.ndarray] = {}
    if desc is None:
        return out
    for row_idx, kp in enumerate(kept):
        if 0 <= kp.class_id < len(xy):
            out[kp.class_id] = desc[row_idx]
    return out


def _stamp_squares(shape: tuple[int, int],
                   centers_xy: np.ndarray, radius: float) -> np.ndarray:
    """Boolean map with True inside an L-infinity square of `radius`
    around each center. Used to tag detected corners that fall near a
    dormant predicted position (Step 4 seeding)."""
    h, w = shape
    hit = np.zeros((h, w), dtype=bool)
    r = int(round(radius))
    for cx, cy in centers_xy:
        x0 = max(0, int(round(cx)) - r); x1 = min(w, int(round(cx)) + r + 1)
        y0 = max(0, int(round(cy)) - r); y1 = min(h, int(round(cy)) + r + 1)
        if x0 < x1 and y0 < y1:
            hit[y0:y1, x0:x1] = True
    return hit


def _compute_steered_brief(
        gray: np.ndarray, kps: list[cv2.KeyPoint],
        orb: cv2.ORB, patch_size: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Compute ORB's steered BRIEF on caller-provided keypoints.

    Returns (kept_kps, descriptors uint8 (N, 32)).

    ORB may drop keypoints whose `size`-radius patch falls off the image
    border. The returned kept_kps is aligned with the descriptor rows.
    """
    if not kps:
        return [], np.empty((0, 32), dtype=np.uint8)
    # ORB.compute uses kp.size as the patch diameter for orientation /
    # sampling, so set it explicitly to the requested patch size.
    for kp in kps:
        kp.size = float(patch_size)
    kept, desc = orb.compute(gray, kps)
    if desc is None:
        return [], np.empty((0, 32), dtype=np.uint8)
    return list(kept), desc


# --------------------------------------------------------------------
# Geometric outlier rejection (Step 3) — Sampson distance via
# findFundamentalMat with RANSAC.
# --------------------------------------------------------------------

def _ransac_inlier_mask(prev_pts: np.ndarray, curr_pts: np.ndarray,
                        ransac_px: float, min_n: int) -> np.ndarray:
    """Return a boolean inlier mask of shape (N,)."""
    n = len(prev_pts)
    if n < min_n:
        return np.zeros(n, dtype=bool)
    # Same degeneracy guards as feature_comparison.epipolar_inlier_ratio,
    # but we keep it inline rather than depending on that function — the
    # POC may want to tweak the thresholds independently.
    p1 = np.ascontiguousarray(prev_pts, np.float32).reshape(-1, 1, 2)
    p2 = np.ascontiguousarray(curr_pts, np.float32).reshape(-1, 1, 2)
    spread1 = np.ptp(p1.reshape(-1, 2), axis=0)
    spread2 = np.ptp(p2.reshape(-1, 2), axis=0)
    if spread1.min() < 5.0 or spread2.min() < 5.0:
        return np.zeros(n, dtype=bool)
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.99)
    except cv2.error:
        return np.zeros(n, dtype=bool)
    if F is None or mask is None or F.shape != (3, 3):
        return np.zeros(n, dtype=bool)
    return mask.flatten().astype(bool)


# --------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------

class HybridFrontend:
    """Stateful per-frame pipeline implementing Steps 1-5."""

    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.cfg = config or HybridConfig()
        self.active_tracks: dict[int, ActiveTrack] = {}
        self.dormant_buffer = DormantTrackBuffer(self.cfg.dormant_horizon_frames)
        self.frame_index: int = -1
        self._next_id: int = 1
        self._prev_gray: Optional[np.ndarray] = None
        # ORB instance used solely for computing steered BRIEF on
        # caller-provided keypoints. fastThreshold is irrelevant because
        # we never call detect() on it.
        self._orb = cv2.ORB_create(
            nfeatures=4 * self.cfg.target_active_tracks,
            patchSize=self.cfg.descriptor_patch_size,
            edgeThreshold=self.cfg.descriptor_patch_size // 2,
        )

    # ---- public API -------------------------------------------------

    def initialize(self, first_gray: np.ndarray) -> None:
        """Bootstrap on the first frame: detect corners, compute
        descriptors, populate active_tracks. No KLT-tracking yet."""
        self.frame_index = 0
        self.active_tracks.clear()
        self.dormant_buffer.clear()
        # Detect Shi-Tomasi on the full image (no occupancy mask — there's
        # nothing tracked yet).
        h, w = first_gray.shape
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        kps = _detect_shi_tomasi_with_quadtree(
            first_gray, full_mask, self.cfg.target_active_tracks,
            self.cfg.shi_tomasi_quality, self.cfg.shi_tomasi_min_distance,
            self.cfg.shi_tomasi_block_size,
        )
        kept, desc = _compute_steered_brief(
            first_gray, kps, self._orb, self.cfg.descriptor_patch_size,
        )
        for kp, d in zip(kept, desc):
            self._spawn_track(kp.pt[0], kp.pt[1], d, octave=0)
        self._prev_gray = first_gray

    def process_frame(self, curr_gray: np.ndarray) -> FrameResult:
        """Run Steps 1-5 for a new incoming frame."""
        assert self._prev_gray is not None, "Call initialize() first"
        self.frame_index += 1
        cfg = self.cfg

        # Purge stale dormant entries every frame. Doing this in Step 4
        # only — as a prior version did — leaves the buffer growing
        # unboundedly on frames where the active set is at/above target
        # and Step 4 doesn't fire. The horizon Δ_dormant is supposed to
        # cap buffer size at roughly (horizon × per-frame spawn rate).
        self.dormant_buffer.purge_older_than(self.frame_index)

        tracks_in = len(self.active_tracks)
        died_this_frame: list[tuple[int, float, float]] = []
        resurrected_ids: list[int] = []
        # Step 5 diagnostic accumulators (populated only when enabled)
        spatial_candidates_per_query: list[int] = []
        accepted_hamming: list[int] = []
        resurrected_ages: list[int] = []
        same_track_hamming: list[int] = []
        same_track_gap1: list[int] = []
        new_corner_positions: list[tuple[float, float]] = []
        flow_dx, flow_dy = 0.0, 0.0

        # ============== Step 1: KLT track active features ==============
        if self.active_tracks:
            ids = list(self.active_tracks.keys())
            prev_pts = np.array(
                [[self.active_tracks[i].x, self.active_tracks[i].y] for i in ids],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray, prev_pts, None,
                winSize=cfg.klt_window, maxLevel=cfg.klt_pyramid_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          cfg.klt_max_iter, cfg.klt_eps),
            )
            # =============== Step 2: FB consistency filter ==============
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, self._prev_gray, curr_pts, None,
                winSize=cfg.klt_window, maxLevel=cfg.klt_pyramid_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          cfg.klt_max_iter, cfg.klt_eps),
            )
            fb_err = np.linalg.norm(
                prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1,
                )
            klt_ok = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1)
            fb_ok = fb_err < cfg.fb_threshold_px

            tracks_after_klt = int(klt_ok.sum())
            tracks_after_fb = int((klt_ok & fb_ok).sum())

            # =============== Step 3: RANSAC outlier rejection ===========
            surv_mask = klt_ok & fb_ok
            surv_ids = [ids[i] for i in range(len(ids)) if surv_mask[i]]
            surv_prev = prev_pts.reshape(-1, 2)[surv_mask]
            surv_curr = curr_pts.reshape(-1, 2)[surv_mask]

            if len(surv_ids) >= cfg.min_matches_for_F:
                inlier_mask = _ransac_inlier_mask(
                    surv_prev, surv_curr, cfg.ransac_reproj_px, cfg.min_matches_for_F,
                )
            else:
                inlier_mask = np.ones(len(surv_ids), dtype=bool)  # too few to test

            tracks_after_ransac = int(inlier_mask.sum())

            # Dominant image motion this frame: median flow over the
            # RANSAC-inlier survivors. Robust to independent outliers
            # (fish, particulates) as long as most of the scene moves
            # with the camera.
            if inlier_mask.any():
                d = surv_curr[inlier_mask] - surv_prev[inlier_mask]
                flow_dx = float(np.median(d[:, 0]))
                flow_dy = float(np.median(d[:, 1]))

            # Motion-compensate existing dormant predictions BEFORE this
            # frame's deaths are added (deaths are added already
            # flow-adjusted below), so that every entry's (last_x, last_y)
            # is its predicted location in the CURRENT frame.
            if cfg.motion_compensate_dormant:
                self.dormant_buffer.translate_all(flow_dx, flow_dy)

            # Kill any track that didn't survive Steps 1-3. Move to dormant
            # buffer (since all our tracks are infants in this POC).
            kept_ids: set[int] = set()
            for i, tid in enumerate(surv_ids):
                if inlier_mask[i]:
                    t = self.active_tracks[tid]
                    t.x = float(surv_curr[i, 0])
                    t.y = float(surv_curr[i, 1])
                    t.age += 1
                    kept_ids.add(tid)

            # ---- Diagnostic: descriptor stability for surviving tracks ----
            # For each track that survived KLT+FB+RANSAC, the new position
            # is by construction the same physical point (KLT tracked it,
            # FB+RANSAC verified it). Recompute the BRIEF descriptor at
            # the new location and Hamming-compare to the birth descriptor.
            # The histogram of these distances tells us what "Hamming for
            # the same physical feature" actually looks like on this
            # footage — i.e., a ground truth for the matcher's threshold.
            if cfg.collect_diagnostics and kept_ids:
                surv_tracks = [self.active_tracks[tid]
                               for i, tid in enumerate(surv_ids)
                               if inlier_mask[i]]
                fresh_map = _descriptors_at_positions(
                    curr_gray, [(t.x, t.y) for t in surv_tracks],
                    self._orb, cfg.descriptor_patch_size,
                )
                for src_idx, fresh in fresh_map.items():
                    t = surv_tracks[src_idx]
                    # (a) birth drift: how far the descriptor has walked
                    # since the track was born. Explains why storing the
                    # BIRTH descriptor in the dormant buffer was fatal.
                    same_track_hamming.append(
                        int(np.unpackbits(
                            np.bitwise_xor(t.birth_descriptor, fresh)).sum())
                    )
                    # (b) 1-frame drift: same physical point, descriptors
                    # recomputed on consecutive frames. This is the
                    # distribution θ_reid should be calibrated against
                    # now that dormant entries store death-time
                    # descriptors and dormancy gaps are ~1-2 frames.
                    if t.diag_prev_descriptor is not None:
                        same_track_gap1.append(
                            int(np.unpackbits(
                                np.bitwise_xor(t.diag_prev_descriptor,
                                               fresh)).sum())
                        )
                    t.diag_prev_descriptor = fresh

            # Everything in `ids` not in `kept_ids` dies.
            #
            # THE FIX for the stale-descriptor bug: the dormant entry
            # stores the descriptor recomputed at the LAST SUCCESSFUL
            # OBSERVATION — (t.x, t.y) on the previous frame, which is
            # where KLT last verified this track — not the birth
            # descriptor. On long-lived tracks the birth descriptor has
            # drifted by ~50 bits (measured: same-track median 47 vs a
            # θ_reid of 50), which capped forced-fail recall at ~47%.
            # The death-time descriptor is at most 1 KLT step stale.
            dying_ids = [tid for tid in ids if tid not in kept_ids]
            if dying_ids:
                death_desc = _descriptors_at_positions(
                    self._prev_gray,
                    [(self.active_tracks[tid].x, self.active_tracks[tid].y)
                     for tid in dying_ids],
                    self._orb, cfg.descriptor_patch_size,
                )
                for k_i, tid in enumerate(dying_ids):
                    t = self.active_tracks.pop(tid)
                    died_this_frame.append((tid, t.x, t.y))
                    # Age gate: short-lived tracks are mostly detector
                    # noise; buffering them in bulk (dark sections)
                    # floods the buffer with impostors for Δ_dormant
                    # frames. Their IDs are simply retired.
                    if t.age < cfg.dormant_min_track_age:
                        continue
                    # Fall back to the birth descriptor only when the
                    # death-time patch fell off the image border (those
                    # tracks are usually unrecoverable anyway).
                    stored = death_desc.get(k_i, t.birth_descriptor)
                    # The last position is in previous-frame coordinates;
                    # apply this frame's flow so the entry is stored as a
                    # current-frame prediction like everything else.
                    px, py = t.x, t.y
                    if cfg.motion_compensate_dormant:
                        px += flow_dx
                        py += flow_dy
                    self.dormant_buffer.add(DormantTrack(
                        id=tid, last_x=px, last_y=py,
                        descriptor=np.asarray(stored, dtype=np.uint8).copy(),
                        frame_died=self.frame_index, octave=t.octave,
                        age_at_death=t.age,
                    ))
        else:
            tracks_after_klt = 0
            tracks_after_fb = 0
            tracks_after_ransac = 0

        # ============== Step 4: top up via Shi-Tomasi + BRIEF ==============
        deficit = cfg.target_active_tracks - len(self.active_tracks)
        new_corners_detected = 0
        reids_attempted = 0
        reids_succeeded = 0
        if deficit > 0:
            mask = _build_occupancy_mask(
                curr_gray.shape, self.active_tracks, cfg.occupancy_mask_radius,
            )
            # Dormant seeding: prefer detected corners near a dormant
            # predicted position, so a recently-died landmark's spot
            # actually receives a candidate for Step 5 to match. With a
            # small deficit and quadtree spread this otherwise almost
            # never happens (the dominant "missed" mode at low kill
            # fractions in the forced-failure test).
            priority_map = None
            pool_n = deficit * 2
            if cfg.seed_corners_near_dormant and not self.dormant_buffer.empty():
                dorm_xy = np.array(
                    [[e.last_x, e.last_y]
                     for e in self.dormant_buffer.all_entries()],
                    dtype=np.float32,
                )
                priority_map = _stamp_squares(
                    curr_gray.shape, dorm_xy, cfg.reid_radius_px,
                )
                # Widen the detection pool so corners near dormant
                # entries can enter it at all (capped to keep the dark-
                # section worst case bounded).
                pool_n = deficit * 2 + min(len(self.dormant_buffer), 500)
            kps = _detect_shi_tomasi_with_quadtree(
                curr_gray, mask, deficit,
                cfg.shi_tomasi_quality, cfg.shi_tomasi_min_distance,
                cfg.shi_tomasi_block_size,
                priority_map=priority_map,
                pool_n=pool_n,
            )
            kept_kps, desc = _compute_steered_brief(
                curr_gray, kps, self._orb, cfg.descriptor_patch_size,
            )
            new_corners_detected = len(kept_kps)
            new_corner_positions = [(float(kp.pt[0]), float(kp.pt[1]))
                                    for kp in kept_kps]

            # ============== Step 5: re-ID against dormant buffer ==========
            if kept_kps and not self.dormant_buffer.empty():
                # For each candidate, look up the dormant tracks within
                # r_reid of its location. We could do this in two ways:
                # (a) loop over candidates and call query_within per candidate
                # (b) build a single list of all dormant entries within ANY
                #     candidate window and call spatial_descriptor_match once
                #
                # (b) is cleaner and matches the §4.8 shared-primitive
                # interface: candidates are the new Shi-Tomasi corners,
                # queries are the dormant entries. But thinking about who
                # is "query" vs "candidate":
                #   - Step 5 asks: for each NEW CORNER, is there a matching
                #     dormant track? So new corners are the queries; dormant
                #     entries are the candidates.
                # That mapping makes the gates work correctly: the spatial
                # gate is "candidate near query", which means "dormant track
                # near new corner". Same thing, but cleaner this way.
                queries = [
                    PixelQuery(
                        x=float(kp.pt[0]), y=float(kp.pt[1]),
                        descriptor=desc[i],
                    )
                    for i, kp in enumerate(kept_kps)
                ]
                all_dormant = list(self.dormant_buffer.all_entries())
                candidates = [
                    PixelCandidate(x=e.last_x, y=e.last_y, descriptor=e.descriptor)
                    for e in all_dormant
                ]
                opts = MatchOptions(
                    default_radius=cfg.reid_radius_px,
                    hamming_threshold=cfg.reid_hamming_threshold,
                    unique_candidates=True,  # one dormant track resurrects to
                    # at most one new corner
                    second_best_margin=cfg.reid_second_best_margin,
                )

                # ---- Diagnostic: spatial candidate density per query ----
                # Cheap: vectorise dormant positions and count per query
                # how many fall inside the L∞ radius. This is what tells
                # us whether the spatial filter is doing useful work.
                if cfg.collect_diagnostics and all_dormant:
                    dorm_xy = np.array([[e.last_x, e.last_y] for e in all_dormant],
                                       dtype=np.float32)
                    r = cfg.reid_radius_px
                    for q in queries:
                        dx = np.abs(dorm_xy[:, 0] - q.x)
                        dy = np.abs(dorm_xy[:, 1] - q.y)
                        spatial_candidates_per_query.append(
                            int(((dx <= r) & (dy <= r)).sum())
                        )

                matches = spatial_descriptor_match(queries, candidates, opts)
                reids_attempted = len(queries)

                # Apply matches: candidates with a match resurrect the
                # dormant track ID; others get a fresh ID.
                used_dormant_ids: set[int] = set()
                for i, m in enumerate(matches):
                    kp = kept_kps[i]
                    if m is not None:
                        dormant = all_dormant[m.candidate_index]
                        assert dormant.id not in used_dormant_ids, (
                            "unique_candidates should have prevented this"
                        )
                        used_dormant_ids.add(dormant.id)
                        # ---- Diagnostic: per-match Hamming + age ----
                        if cfg.collect_diagnostics:
                            accepted_hamming.append(m.hamming_distance)
                            resurrected_ages.append(
                                self.frame_index - dormant.frame_died
                            )
                        # Reuse the original ID. The new descriptor is the
                        # current corner's descriptor (a fresh birth
                        # descriptor for the resurrected infant — per the
                        # design doc the dormant track's stored descriptor
                        # was its single-shot birth one and we now have a
                        # newer observation, so use that).
                        self._spawn_track(
                            kp.pt[0], kp.pt[1], desc[i], octave=0,
                            id=dormant.id, age=dormant.age_at_death,
                        )
                        self.dormant_buffer.remove(dormant.id)
                        resurrected_ids.append(dormant.id)
                        reids_succeeded += 1
                    else:
                        self._spawn_track(kp.pt[0], kp.pt[1], desc[i], octave=0)
            else:
                # Either no candidates or empty dormant buffer: every new
                # corner gets a fresh ID.
                for i, kp in enumerate(kept_kps):
                    self._spawn_track(kp.pt[0], kp.pt[1], desc[i], octave=0)

        # ============== End-of-frame bookkeeping ==============
        self._prev_gray = curr_gray
        return FrameResult(
            frame_index=self.frame_index,
            tracks_in=tracks_in,
            tracks_after_klt=tracks_after_klt,
            tracks_after_fb=tracks_after_fb,
            tracks_after_ransac=tracks_after_ransac,
            new_corners_detected=new_corners_detected,
            reids_attempted=reids_attempted,
            reids_succeeded=reids_succeeded,
            tracks_out=len(self.active_tracks),
            dormant_buffer_size=len(self.dormant_buffer),
            died_this_frame=died_this_frame,
            resurrected_ids=resurrected_ids,
            spatial_candidates_per_query=spatial_candidates_per_query,
            accepted_hamming_distances=accepted_hamming,
            resurrected_ages=resurrected_ages,
            same_track_hamming_distances=same_track_hamming,
            same_track_gap1_hamming_distances=same_track_gap1,
            median_flow=(flow_dx, flow_dy),
            new_corner_positions=new_corner_positions,
        )

    def force_kill(self, track_ids: list[int]) -> list[tuple[int, float, float]]:
        """Force-kill the given tracks (move them into the dormant buffer
        as if KLT had just failed on them). Used by the synthetic
        re-ID precision/recall test (§10 item 2).

        Returns a list of (id, last_x, last_y) for the killed tracks.
        Should be called AFTER process_frame() so that the killed tracks
        carry their just-tracked positions — which also means
        self._prev_gray is already the CURRENT frame, so the death-time
        descriptor is computed at the exact kill appearance.

        Unlike natural deaths, force_kill buffers unconditionally
        (no dormant_min_track_age gate): the caller explicitly asked for
        these tracks to be resurrectable. The forced-failure harness is
        expected to filter its kill sample by age instead.
        """
        victims = [(tid, self.active_tracks[tid]) for tid in track_ids
                   if tid in self.active_tracks]
        death_desc = _descriptors_at_positions(
            self._prev_gray, [(t.x, t.y) for _, t in victims],
            self._orb, self.cfg.descriptor_patch_size,
        )
        killed = []
        for k_i, (tid, t) in enumerate(victims):
            self.active_tracks.pop(tid)
            killed.append((tid, t.x, t.y))
            stored = death_desc.get(k_i, t.birth_descriptor)
            self.dormant_buffer.add(DormantTrack(
                id=tid, last_x=t.x, last_y=t.y,
                descriptor=np.asarray(stored, dtype=np.uint8).copy(),
                frame_died=self.frame_index, octave=t.octave,
                age_at_death=t.age,
            ))
        return killed

    # ---- internal helpers --------------------------------------------

    def _spawn_track(self, x: float, y: float, descriptor: np.ndarray,
                     octave: int = 0, id: Optional[int] = None,
                     age: int = 0) -> int:
        """Create an active track and return its id.

        `age` is nonzero only for resurrections, where the dormant
        entry's age_at_death is carried over so an established landmark
        stays established for age-gated logic.
        """
        if id is None:
            id = self._next_id
            self._next_id += 1
        else:
            # Defensive: don't let an externally-supplied id collide with
            # future fresh ids. Bump _next_id past it.
            self._next_id = max(self._next_id, id + 1)
        assert id not in self.active_tracks, f"Duplicate active track id {id}"
        self.active_tracks[id] = ActiveTrack(
            id=id, x=float(x), y=float(y),
            birth_descriptor=descriptor.astype(np.uint8).copy(),
            octave=octave, age=age,
        )
        return id
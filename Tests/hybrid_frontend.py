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
    # θ_reid BASE: acceptance ceiling for a 1-frame dormancy gap.
    # Calibrate from the "same-track Hamming (1-frame gap)" diagnostic
    # (p95 + margin). On the turbid coral footage that's ~32.
    reid_hamming_threshold: int = 32
    # Gap scaling: a candidate that died g frames ago is accepted up to
    #   min(cap, base + slope * g)
    # so short gaps stay strict (most matches; keeps hijacks near zero)
    # while long gaps get the headroom their accumulated drift needs.
    reid_hamming_slope_per_frame: float = 1.0
    reid_hamming_cap: int = 55
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

    # ---- Representative descriptor (ORB-SLAM3-style) ----
    # Instead of storing one snapshot of the descriptor, accumulate
    # observations along the track's life and store the one with the
    # MINIMUM MEDIAN Hamming distance to all the others. That is
    # ORB-SLAM3's MapPoint::ComputeDistinctiveDescriptors, and it is far
    # more robust than any single observation: an unlucky snapshot (motion
    # blur, a flicker in illumination, a bad orientation estimate) can sit
    # tens of bits away from the landmark's typical appearance, whereas
    # the medoid is by construction central to it. Cheap here because KLT
    # already verifies a fresh observation every frame.
    use_representative_descriptor: bool = True
    # Observations are sampled every `stride` frames so the set spans the
    # track's life rather than a burst of near-identical adjacent frames.
    representative_max_observations: int = 8
    representative_sample_stride: int = 3

    # ---- Local detection inside dormant windows ----
    # Step 4's global detector only has `deficit` corners to spend and
    # spreads them over the whole image, so a killed landmark's spot
    # often receives no candidate at all and Step 5 never gets to try
    # (the "no corner ever detected there" bucket). This adds a targeted
    # local search: within each dormant prediction window, take the best
    # Shi-Tomasi response above a RELAXED threshold. Analogous to
    # ORB-SLAM3's search-by-projection, which likewise looks where the
    # map says a landmark should be instead of relying on global
    # detection to find it.
    #
    # These corners are re-ID CANDIDATES ONLY: if a local corner fails to
    # match a dormant track it is discarded, never spawned as a new
    # track. So this cannot inflate the active set past N_target, and it
    # is budget-independent of `deficit`.
    local_detect_in_dormant_windows: bool = True
    # Quality threshold for local corners, as a fraction of the global
    # Shi-Tomasi threshold. < 1 means "accept weaker corners here,
    # because we have strong positional evidence a landmark exists".
    local_detect_quality_scale: float = 0.3
    # Cap on windows searched per frame (bounds the dark-section cost).
    local_detect_max_windows: int = 400

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
    # Sampled descriptor observations over the track's life, and the
    # medoid computed from them (see use_representative_descriptor).
    descriptor_history: list = field(default_factory=list)
    representative_descriptor: Optional[np.ndarray] = None


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
    # Descriptors of those corners, aligned row-for-row with
    # new_corner_positions. Lets the harness measure the true
    # "stored dormant descriptor vs freshly-detected corner descriptor"
    # distance — the exact comparison Step 5 performs — without
    # recomputing anything.
    new_corner_descriptors: Optional[np.ndarray] = None


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


def _representative_descriptor(observations: list) -> np.ndarray:
    """Return the medoid of a set of descriptors: the observation with
    the minimum MEDIAN Hamming distance to all the others.

    This is ORB-SLAM3's ComputeDistinctiveDescriptors. The median (not
    the mean) is what makes it robust — a couple of badly corrupted
    observations cannot drag the choice away from the landmark's typical
    appearance. Returns the single observation when there is only one.
    """
    n = len(observations)
    if n == 1:
        return observations[0]
    obs = np.stack(observations, axis=0)                    # (n, 32)
    xor = np.bitwise_xor(obs[:, None, :], obs[None, :, :])  # (n, n, 32)
    dists = np.unpackbits(xor, axis=2).sum(axis=2)          # (n, n)
    medians = np.median(dists, axis=1)
    return observations[int(np.argmin(medians))]


def _local_corners_in_windows(
        gray: np.ndarray, mask: np.ndarray, centers_xy: np.ndarray,
        radius: float, block_size: int, quality: float,
        quality_scale: float, max_windows: int,
        min_separation: float, existing_xy: list,
) -> list[tuple[float, float]]:
    """Best Shi-Tomasi response inside each window, above a relaxed
    threshold. One shared cornerMinEigenVal pass over the image, then an
    argmax per window — far cheaper than a goodFeaturesToTrack call per
    window, and equivalent for "give me the single best corner here".

    Respects the occupancy mask (never returns a corner on top of a live
    track) and enforces `min_separation` from corners already selected.
    """
    if centers_xy.size == 0:
        return []
    h, w = gray.shape
    eig = cv2.cornerMinEigenVal(gray, block_size)
    # Relaxed absolute threshold, mirroring goodFeaturesToTrack's
    # qualityLevel * max(eig) convention.
    thresh = float(eig.max()) * quality * quality_scale
    r = max(1, int(round(radius)))
    chosen: list[tuple[float, float]] = []
    taken = list(existing_xy)
    sep2 = min_separation * min_separation
    for cx, cy in centers_xy[:max_windows]:
        x0 = max(0, int(round(cx)) - r); x1 = min(w, int(round(cx)) + r + 1)
        y0 = max(0, int(round(cy)) - r); y1 = min(h, int(round(cy)) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        sub = eig[y0:y1, x0:x1]
        sub_mask = mask[y0:y1, x0:x1]
        cand = np.where(sub_mask > 0, sub, -1.0)
        idx = int(np.argmax(cand))
        val = float(cand.flat[idx])
        if val < thresh:
            continue
        yy, xx = divmod(idx, cand.shape[1])
        px, py = float(x0 + xx), float(y0 + yy)
        if any((px - ex) ** 2 + (py - ey) ** 2 < sep2 for ex, ey in taken):
            continue
        chosen.append((px, py))
        taken.append((px, py))
    return chosen


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
        new_corner_descriptors = None
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
            if (cfg.collect_diagnostics or cfg.use_representative_descriptor) \
                    and kept_ids:
                surv_tracks = [self.active_tracks[tid]
                               for i, tid in enumerate(surv_ids)
                               if inlier_mask[i]]
                fresh_map = _descriptors_at_positions(
                    curr_gray, [(t.x, t.y) for t in surv_tracks],
                    self._orb, cfg.descriptor_patch_size,
                )
                for src_idx, fresh in fresh_map.items():
                    t = surv_tracks[src_idx]
                    # Accumulate a sampled observation and refresh the
                    # medoid. Sampling by stride spreads the set over the
                    # track's life instead of collecting a burst of
                    # near-identical adjacent frames; the cap keeps the
                    # pairwise medoid computation trivial.
                    if cfg.use_representative_descriptor:
                        if (t.age % max(1, cfg.representative_sample_stride)) == 0:
                            t.descriptor_history.append(fresh.copy())
                            if len(t.descriptor_history) > cfg.representative_max_observations:
                                # Drop the OLDEST non-birth observation so
                                # the set keeps spanning the track's life
                                # (index 0 is the birth appearance, worth
                                # retaining as one anchor).
                                del t.descriptor_history[1]
                            t.representative_descriptor = _representative_descriptor(
                                t.descriptor_history)
                    if not cfg.collect_diagnostics:
                        continue
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
                    # Prefer the medoid over the death-time snapshot: a
                    # single observation at the moment KLT failed is
                    # exactly the observation most likely to be corrupted
                    # (that is often WHY it failed).
                    stored = death_desc.get(k_i, t.birth_descriptor)
                    if (cfg.use_representative_descriptor
                            and t.representative_descriptor is not None):
                        stored = t.representative_descriptor
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
            new_corner_descriptors = desc if len(kept_kps) else None

            # ============== Step 5: re-ID against dormant buffer ==========
            # Local detection can create re-ID candidates even when the
            # global detector produced none, so the gate is on the
            # dormant buffer rather than on kept_kps.
            if not self.dormant_buffer.empty():
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
                # ---- Targeted local detection in dormant windows ----
                # Positions/descriptors of the global Step 4 corners, then
                # any extra corners found only inside dormant windows.
                # `n_global` marks the boundary: queries at or beyond it
                # are local-only and are DISCARDED if unmatched, so they
                # can never inflate the active set.
                q_xy = [(float(kp.pt[0]), float(kp.pt[1])) for kp in kept_kps]
                q_desc = [desc[i] for i in range(len(kept_kps))]
                n_global = len(q_xy)

                if cfg.local_detect_in_dormant_windows:
                    dorm_all = list(self.dormant_buffer.all_entries())
                    dorm_pos = np.array([[e.last_x, e.last_y] for e in dorm_all],
                                        dtype=np.float32)
                    local_mask = _build_occupancy_mask(
                        curr_gray.shape, self.active_tracks,
                        cfg.occupancy_mask_radius,
                    )
                    local_xy = _local_corners_in_windows(
                        curr_gray, local_mask, dorm_pos,
                        radius=cfg.reid_radius_px,
                        block_size=cfg.shi_tomasi_block_size,
                        quality=cfg.shi_tomasi_quality,
                        quality_scale=cfg.local_detect_quality_scale,
                        max_windows=cfg.local_detect_max_windows,
                        min_separation=float(cfg.shi_tomasi_min_distance),
                        existing_xy=q_xy,
                    )
                    if local_xy:
                        l_kps = [cv2.KeyPoint(x=x, y=y,
                                              size=float(cfg.descriptor_patch_size))
                                 for x, y in local_xy]
                        l_kept, l_desc = _compute_steered_brief(
                            curr_gray, l_kps, self._orb,
                            cfg.descriptor_patch_size,
                        )
                        for j, kp in enumerate(l_kept):
                            q_xy.append((float(kp.pt[0]), float(kp.pt[1])))
                            q_desc.append(l_desc[j])

                queries = [
                    PixelQuery(x=x, y=y, descriptor=q_desc[i])
                    for i, (x, y) in enumerate(q_xy)
                ]
                # Report ALL Step 5 candidates (global + local) so the
                # positions and descriptors handed to the harness stay
                # aligned and its "was a corner available here?"
                # diagnostic reflects what the matcher actually saw.
                new_corner_positions = list(q_xy)
                new_corner_descriptors = (
                    np.stack(q_desc, axis=0) if q_desc else None
                )
                all_dormant = list(self.dormant_buffer.all_entries())
                candidates = [
                    PixelCandidate(
                        x=e.last_x, y=e.last_y, descriptor=e.descriptor,
                        # Gap-scaled acceptance: descriptor drift grows
                        # with the dormancy gap (measured: 1-frame drift
                        # p90≈22 vs long-gap drift approaching the
                        # birth-drift regime), so each candidate's
                        # threshold covers ITS OWN gap instead of one
                        # fixed number covering neither regime well.
                        hamming_threshold=min(
                            cfg.reid_hamming_cap,
                            int(round(
                                cfg.reid_hamming_threshold +
                                cfg.reid_hamming_slope_per_frame *
                                max(0, self.frame_index - e.frame_died)
                            )),
                        ),
                    )
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
                # Apply matches under a strict budget: total additions
                # to the active set this frame (resurrections + fresh
                # spawns) may not exceed `deficit`, so N_target is
                # honoured exactly. Resurrections are applied FIRST
                # because recovering a landmark's identity is worth more
                # than adding an anonymous new corner — a fresh corner
                # can be spawned next frame, whereas a dormant entry
                # expires and its identity is lost for good.
                used_dormant_ids: set[int] = set()
                budget = deficit

                for i, m in enumerate(matches):
                    if m is None or budget <= 0:
                        continue
                    dormant = all_dormant[m.candidate_index]
                    assert dormant.id not in used_dormant_ids, (
                        "unique_candidates should have prevented this"
                    )
                    used_dormant_ids.add(dormant.id)
                    if cfg.collect_diagnostics:
                        accepted_hamming.append(m.hamming_distance)
                        resurrected_ages.append(
                            self.frame_index - dormant.frame_died
                        )
                    qx, qy = q_xy[i]
                    # The resurrected track is seeded with BOTH the fresh
                    # observation and the dormant entry's representative
                    # descriptor, so the landmark keeps its accumulated
                    # appearance evidence across the gap.
                    self._spawn_track(
                        qx, qy, q_desc[i], octave=0,
                        id=dormant.id, age=dormant.age_at_death,
                        seed_history=[dormant.descriptor],
                    )
                    self.dormant_buffer.remove(dormant.id)
                    resurrected_ids.append(dormant.id)
                    reids_succeeded += 1
                    budget -= 1

                for i, m in enumerate(matches):
                    # Unmatched GLOBAL corners become new tracks. Unmatched
                    # LOCAL-only corners are discarded: they were detected
                    # purely as re-ID evidence under a relaxed threshold
                    # and outside the deficit budget, so promoting them
                    # would pollute the map with weak corners.
                    if m is not None or i >= n_global or budget <= 0:
                        continue
                    qx, qy = q_xy[i]
                    self._spawn_track(qx, qy, q_desc[i], octave=0)
                    budget -= 1
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
            new_corner_descriptors=new_corner_descriptors,
        )

    def descriptors_at_current(
            self, xy: list[tuple[float, float]]) -> dict[int, np.ndarray]:
        """Compute steered-BRIEF descriptors at arbitrary positions on the
        MOST RECENTLY PROCESSED frame.

        Diagnostic hook for the forced-failure harness: it lets the
        caller ask "what would the descriptor look like right here, right
        now?" at a dormant track's predicted location, so descriptor
        drift can be measured against the dormant entry's stored
        descriptor for the population that actually matters (tracks that
        DIED), rather than for KLT survivors.

        Returns {input_index: descriptor}; positions whose patch falls
        off the image border are absent.
        """
        return _descriptors_at_positions(
            self._prev_gray, xy, self._orb, self.cfg.descriptor_patch_size,
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
            if (self.cfg.use_representative_descriptor
                    and t.representative_descriptor is not None):
                stored = t.representative_descriptor
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
                     age: int = 0, seed_history: Optional[list] = None) -> int:
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
        d0 = descriptor.astype(np.uint8).copy()
        t = ActiveTrack(
            id=id, x=float(x), y=float(y),
            birth_descriptor=d0, octave=octave, age=age,
        )
        if self.cfg.use_representative_descriptor:
            # Seed the observation set. On a resurrection, `seed_history`
            # carries the dormant entry's representative across the gap so
            # the landmark's accumulated appearance evidence is not thrown
            # away and rebuilt from one fresh corner.
            t.descriptor_history = [d0]
            if seed_history:
                for d in seed_history:
                    t.descriptor_history.append(np.asarray(d, np.uint8).copy())
            t.representative_descriptor = _representative_descriptor(
                t.descriptor_history)
        self.active_tracks[id] = t
        return id
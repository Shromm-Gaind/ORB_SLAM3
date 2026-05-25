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
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
# The user's file is feature_comparison.py at the uploads root.
sys.path.insert(0, "/mnt/user-data/uploads")
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
    reid_hamming_threshold: int = 50      # θ_reid

    # Tracking-lost threshold (§8 N_min)
    min_active_tracks: int = 50


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
) -> list[cv2.KeyPoint]:
    """Detect Shi-Tomasi corners constrained by `mask`, then enforce
    spatial spread with the ORB-SLAM3 quadtree.

    The quadtree implementation in feature_comparison.py works on
    cv2.KeyPoint objects (which is what we want anyway, to feed to
    ORB.compute() in the next step).
    """
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=target_n * 2,   # over-detect; quadtree will thin
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
    # Quadtree distribution (uses kp.pt and kp.response).
    if len(kps) > target_n:
        kps = _distribute_quadtree(kps, w, h, target_n)
    return kps


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

        tracks_in = len(self.active_tracks)
        died_this_frame: list[tuple[int, float, float]] = []
        resurrected_ids: list[int] = []

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
            # Everything in `ids` not in `kept_ids` dies.
            for tid in ids:
                if tid not in kept_ids:
                    t = self.active_tracks.pop(tid)
                    died_this_frame.append((tid, t.x, t.y))
                    # POC: every dying track is an infant (map_point is None).
                    # Add to dormant buffer for possible re-ID later.
                    self.dormant_buffer.add(DormantTrack(
                        id=tid, last_x=t.x, last_y=t.y,
                        descriptor=t.birth_descriptor.copy(),
                        frame_died=self.frame_index, octave=t.octave,
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
            kps = _detect_shi_tomasi_with_quadtree(
                curr_gray, mask, deficit,
                cfg.shi_tomasi_quality, cfg.shi_tomasi_min_distance,
                cfg.shi_tomasi_block_size,
            )
            kept_kps, desc = _compute_steered_brief(
                curr_gray, kps, self._orb, cfg.descriptor_patch_size,
            )
            new_corners_detected = len(kept_kps)

            # ============== Step 5: re-ID against dormant buffer ==========
            # Purge stale entries first.
            self.dormant_buffer.purge_older_than(self.frame_index)

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
                        # Reuse the original ID. The new descriptor is the
                        # current corner's descriptor (a fresh birth
                        # descriptor for the resurrected infant — per the
                        # design doc the dormant track's stored descriptor
                        # was its single-shot birth one and we now have a
                        # newer observation, so use that).
                        self._spawn_track(
                            kp.pt[0], kp.pt[1], desc[i], octave=0,
                            id=dormant.id,
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
        )

    def force_kill(self, track_ids: list[int]) -> list[tuple[int, float, float]]:
        """Force-kill the given tracks (move them into the dormant buffer
        as if KLT had just failed on them). Used by the synthetic
        re-ID precision/recall test (§10 item 2).

        Returns a list of (id, last_x, last_y) for the killed tracks.
        Should be called AFTER process_frame() so that the killed tracks
        carry their just-tracked positions.
        """
        killed = []
        for tid in track_ids:
            t = self.active_tracks.pop(tid, None)
            if t is None:
                continue
            killed.append((tid, t.x, t.y))
            self.dormant_buffer.add(DormantTrack(
                id=tid, last_x=t.x, last_y=t.y,
                descriptor=t.birth_descriptor.copy(),
                frame_died=self.frame_index, octave=t.octave,
            ))
        return killed

    # ---- internal helpers --------------------------------------------

    def _spawn_track(self, x: float, y: float, descriptor: np.ndarray,
                     octave: int = 0, id: Optional[int] = None) -> int:
        """Create an active track and return its id."""
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
            octave=octave, age=0,
        )
        return id

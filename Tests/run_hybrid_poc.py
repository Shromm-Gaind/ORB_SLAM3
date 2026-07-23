#!/usr/bin/env python3
"""
run_hybrid_poc.py

Driver for the hybrid-frontend Python proof-of-concept.

Two modes:
  --mode normal      Process the sequence end-to-end, producing a
                     comparison video, per-frame metrics CSV, and
                     summary plots. This is the everyday "is the
                     pipeline doing the right thing on real data" mode.
  --mode forced-fail Run the sequence with synthetic forced KLT failures
                     (~10% of active tracks per frame) and measure the
                     re-ID precision/recall. This validates §4.6 (Step 5)
                     against the §10 item-2 target: > 80% correct
                     resurrection, < 1% incorrect resurrection.

Usage:
  python run_hybrid_poc.py /path/to/sequence --output ./poc_results
  python run_hybrid_poc.py /path/to/sequence --output ./poc_results \\
      --mode forced-fail --kill-fraction 0.10 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_frontend import (
    FrameResult,
    HybridConfig,
    HybridFrontend,
)


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image_paths(folder: Path) -> list[Path]:
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not paths:
        sys.exit(f"No images found in {folder}")
    print(f"Found {len(paths)} images in {folder}")
    return paths


def read_gray(path: Path, clahe: cv2.CLAHE | None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    if clahe is not None:
        img = clahe.apply(img)
    return img


# --------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------

def draw_frame(curr_gray: np.ndarray,
               frontend: HybridFrontend,
               result: FrameResult,
               kill_ids: set[int] | None = None) -> np.ndarray:
    """Render the current frame with track points colour-coded.
       Green = active, blue = resurrected this frame, red = killed this frame."""
    vis = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
    resurrected = set(result.resurrected_ids)
    died = {tid for tid, _, _ in result.died_this_frame}
    for t in frontend.active_tracks.values():
        if t.id in resurrected:
            colour = (255, 100, 0)  # blue-ish (BGR) for resurrected
        else:
            colour = (0, 200, 0)    # green for normal active
        cv2.circle(vis, (int(round(t.x)), int(round(t.y))), 2, colour, -1, cv2.LINE_AA)
    # Show death locations from this frame (already removed from active).
    for tid, dx, dy in result.died_this_frame:
        cv2.drawMarker(vis, (int(round(dx)), int(round(dy))),
                       (0, 0, 200), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

    h, w = vis.shape[:2]
    bar = np.zeros((60, w, 3), dtype=np.uint8)
    line1 = (f"frame={result.frame_index}  active={result.tracks_out}  "
             f"reid={result.reids_succeeded}/{result.reids_attempted}  "
             f"dormant={result.dormant_buffer_size}")
    line2 = (f"KLT_in={result.tracks_in} -> klt={result.tracks_after_klt} "
             f"-> fb={result.tracks_after_fb} -> ransac={result.tracks_after_ransac}  "
             f"new={result.new_corners_detected}")
    cv2.putText(bar, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


# --------------------------------------------------------------------
# Normal mode
# --------------------------------------------------------------------

def _percentiles(xs: list[int]) -> tuple[float, float, float, float, float]:
    """Return (n, mean, median, p90, max) for a list. Zeros if empty."""
    if not xs:
        return (0, 0.0, 0.0, 0.0, 0.0)
    a = np.asarray(xs)
    return (len(a), float(a.mean()), float(np.median(a)),
            float(np.percentile(a, 90)), float(a.max()))


def run_normal(paths: list[Path], front: HybridFrontend, output: Path,
               clahe: cv2.CLAHE | None, fps: float, write_video: bool):
    """Process the sequence and emit metrics + (optionally) a video."""
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame_idx", "filename",
        "tracks_in", "tracks_after_klt", "tracks_after_fb", "tracks_after_ransac",
        "new_corners_detected", "reids_attempted", "reids_succeeded",
        "tracks_out", "dormant_buffer_size",
        # ---- Diagnostics: spatial filter density ----
        "spatial_cand_mean", "spatial_cand_median", "spatial_cand_p90", "spatial_cand_max",
        # ---- Diagnostics: descriptor quality of accepted matches ----
        "accepted_hamming_mean", "accepted_hamming_median", "accepted_hamming_p90",
        # ---- Diagnostics: how stale are matched dormant entries ----
        "resurrected_age_mean", "resurrected_age_median", "resurrected_age_p90", "resurrected_age_max",
        # ---- Diagnostics: descriptor stability of the same physical point ----
        "same_track_hamming_mean", "same_track_hamming_median", "same_track_hamming_p90", "same_track_hamming_max",
    ])

    first = read_gray(paths[0], clahe)
    front.initialize(first)
    h, w = first.shape
    writer = None
    if write_video:
        video_path = output / "hybrid.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h + 60))
        if not writer.isOpened():
            print(f"Warning: could not open video writer at {video_path}")
            writer = None

    metrics_log: list[FrameResult] = []
    for idx in range(1, len(paths)):
        curr = read_gray(paths[idx], clahe)
        res = front.process_frame(curr)
        metrics_log.append(res)
        sc = _percentiles(res.spatial_candidates_per_query)
        hd = _percentiles(res.accepted_hamming_distances)
        ag = _percentiles(res.resurrected_ages)
        st = _percentiles(res.same_track_hamming_distances)
        csv_w.writerow([
            res.frame_index, paths[idx].name,
            res.tracks_in, res.tracks_after_klt,
            res.tracks_after_fb, res.tracks_after_ransac,
            res.new_corners_detected, res.reids_attempted, res.reids_succeeded,
            res.tracks_out, res.dormant_buffer_size,
            f"{sc[1]:.2f}", f"{sc[2]:.2f}", f"{sc[3]:.2f}", int(sc[4]),
            f"{hd[1]:.2f}", f"{hd[2]:.2f}", f"{hd[3]:.2f}",
            f"{ag[1]:.2f}", f"{ag[2]:.2f}", f"{ag[3]:.2f}", int(ag[4]),
            f"{st[1]:.2f}", f"{st[2]:.2f}", f"{st[3]:.2f}", int(st[4]),
        ])
        if writer is not None:
            vis = draw_frame(curr, front, res)
            writer.write(vis)
        if idx % 25 == 0 or idx == len(paths) - 1:
            print(f"  frame {idx}/{len(paths)-1}  "
                  f"active={res.tracks_out}  "
                  f"reid={res.reids_succeeded}/{res.reids_attempted}  "
                  f"dormant={res.dormant_buffer_size}  "
                  f"spatial_cand[mean={sc[1]:.1f},p90={sc[3]:.0f},max={int(sc[4])}]  "
                  f"acc_ham[mean={hd[1]:.1f},p90={hd[3]:.0f}]  "
                  f"same_track_ham[mean={st[1]:.1f},p90={st[3]:.0f}]  "
                  f"age[mean={ag[1]:.1f},p90={ag[3]:.0f}]")

    csv_f.close()
    if writer is not None:
        writer.release()
        print(f"Saved video:   {output / 'hybrid.mp4'}")
    print(f"Saved metrics: {csv_path}")

    if metrics_log:
        _print_aggregate_diagnostics(metrics_log,
                                     current_threshold=front.cfg.reid_hamming_threshold)
        plot_normal_summary(metrics_log, output / "summary.png")


def _print_aggregate_diagnostics(metrics_log: list[FrameResult],
                                 current_threshold: int | None = None):
    """End-of-run summary that pools per-frame diagnostic arrays and
    reports the distribution shape across the whole sequence.

    current_threshold lets us print "Suggested θ_reid = X (currently Y)"
    rather than a bare suggestion.
    """
    all_spatial = []
    all_hamming = []
    all_ages = []
    all_same_track = []
    all_gap1 = []
    for r in metrics_log:
        all_spatial.extend(r.spatial_candidates_per_query)
        all_hamming.extend(r.accepted_hamming_distances)
        all_ages.extend(r.resurrected_ages)
        all_same_track.extend(r.same_track_hamming_distances)
        all_gap1.extend(r.same_track_gap1_hamming_distances)

    print()
    print("=" * 70)
    print("RE-ID DIAGNOSTICS — aggregate over sequence")
    print("=" * 70)

    def row(name: str, xs: list[int]):
        if not xs:
            print(f"  {name:<42s} (no data)")
            return
        a = np.asarray(xs)
        print(f"  {name:<42s} n={len(a):>8d}  "
              f"mean={a.mean():>6.2f}  median={np.median(a):>5.1f}  "
              f"p90={np.percentile(a, 90):>5.1f}  max={a.max():>5d}")

    row("spatial candidates per query", all_spatial)
    row("accepted-match Hamming distance", all_hamming)
    row("same-track Hamming (birth vs current)", all_same_track)
    row("same-track Hamming (1-frame gap)", all_gap1)
    row("resurrected dormant entry age (frames)", all_ages)

    # θ_reid calibration: dormant entries now store the death-time
    # descriptor and dormancy gaps are ~1-2 frames, so the acceptance
    # threshold should cover the 1-frame same-track drift distribution
    # (with a small allowance for the extra gap frames), NOT the birth
    # drift. p95 of the 1-frame drift plus a few bits is a sane pick.
    if all_gap1:
        p95 = float(np.percentile(np.asarray(all_gap1), 95))
        suggested = int(round(p95 + 5))
        cur = (f" (currently {current_threshold})"
               if current_threshold is not None else "")
        print(f"\n  Suggested θ_reid from 1-frame drift p95+5: "
              f"{suggested}{cur}")

    # ---- Compare accepted-match Hamming to same-track Hamming. ----
    # The same-track distribution is the ground truth for "this is the
    # same physical point" on this footage; the accepted-match
    # distribution should look broadly similar if Step 5 is matching
    # correctly. A large gap (accepted >> same-track) means Step 5 is
    # accepting matches that are descriptively worse than what we get
    # when we KNOW two observations are the same point — i.e., noise.
    #
    # Important caveat: the global same-track baseline is contaminated
    # by frames where the camera isn't looking at stable geometry
    # (open water, heavy turbidity, motion blur). In those frames KLT
    # may "survive" but the patches aren't really the same physical
    # point, so the baseline is inflated by noise that isn't relevant
    # to Step 5's success conditions. We therefore segment the data by
    # RANSAC survival rate as a proxy for "do we have real geometry"
    # and report stats on stable frames separately.
    if all_hamming and all_same_track:
        # Per-frame stable mask: tracks_after_ransac >= 60% of tracks_in
        # is a generous "the geometry test is firing cleanly" threshold.
        stable_acc = []
        stable_st = []
        for r in metrics_log:
            if r.tracks_in > 0 and r.tracks_after_ransac / r.tracks_in >= 0.5:
                stable_acc.extend(r.accepted_hamming_distances)
                stable_st.extend(r.same_track_hamming_distances)

        acc_p90 = float(np.percentile(all_hamming, 90))
        st_p90 = float(np.percentile(all_same_track, 90))
        acc_med = float(np.median(all_hamming))
        st_med = float(np.median(all_same_track))
        print()
        print(f"  Global:   accepted Hamming med={acc_med:.1f} p90={acc_p90:.1f}  "
              f"vs same-track med={st_med:.1f} p90={st_p90:.1f}")
        if stable_acc and stable_st:
            sa_p90 = float(np.percentile(stable_acc, 90))
            ss_p90 = float(np.percentile(stable_st, 90))
            sa_med = float(np.median(stable_acc))
            ss_med = float(np.median(stable_st))
            stable_frames = sum(
                1 for r in metrics_log
                if r.tracks_in > 0 and r.tracks_after_ransac / r.tracks_in >= 0.5
            )
            print(f"  Stable:   accepted Hamming med={sa_med:.1f} p90={sa_p90:.1f}  "
                  f"vs same-track med={ss_med:.1f} p90={ss_p90:.1f}  "
                  f"({stable_frames}/{len(metrics_log)} frames)")
            # The stable-frame comparison is the one to act on.
            gap = sa_med - ss_med
            curr = f"currently {current_threshold}" if current_threshold else "current threshold unknown"
            print()
            if gap > 8:
                target = int(ss_med + 5)
                print(f"  → STABLE segment shows {gap:.1f}-bit gap: matcher accepts")
                print(f"    matches descriptively worse than the same-point baseline.")
                print(f"    Suggested θ_reid = {target} ({curr}).")
            elif gap > 3:
                target = int(ss_med + 5)
                print(f"  → Moderate gap of {gap:.1f} bits in stable segment; threshold")
                print(f"    is in the right ballpark but could tighten to ~{target} ({curr}).")
            else:
                print(f"  → Accepted matches in stable segment closely track the")
                print(f"    same-point baseline ({gap:.1f}-bit gap). Threshold is appropriate.")

    # ---- Sanity check: do same-track Hamming spikes correlate with
    # bad geometry frames (low RANSAC survival)? If yes, the bad-segment
    # interpretation is supported; if no, BRIEF noise is segment-
    # independent and we have a more fundamental problem.
    if all_same_track:
        per_frame_st = []
        per_frame_surv_rate = []
        for r in metrics_log:
            if r.tracks_in > 0 and r.same_track_hamming_distances:
                per_frame_st.append(np.mean(r.same_track_hamming_distances))
                per_frame_surv_rate.append(r.tracks_after_ransac / r.tracks_in)
        if len(per_frame_st) > 10:
            corr = float(np.corrcoef(per_frame_st, per_frame_surv_rate)[0, 1])
            print()
            print(f"  corr(same-track Hamming, RANSAC survival rate) = {corr:+.2f}")
            if corr < -0.3:
                print(f"    → strong negative correlation: high Hamming concentrates")
                print(f"      in bad-geometry frames. Stable-segment stats are the")
                print(f"      right reading; threshold tuning will help.")
            elif corr < -0.1:
                print(f"    → mild negative correlation: BRIEF noise is partly geometry-")
                print(f"      dependent, partly intrinsic.")
            else:
                print(f"    → no clear correlation: BRIEF noise is roughly uniform")
                print(f"      across the sequence; threshold tuning has limited headroom.")

    # ---- Individual warnings ----
    if all_spatial and np.mean(all_spatial) > 10:
        print()
        print("  ! spatial filter is overloaded: many candidates compete per query")
        print("    → tighten r_reid or shorten Δ_dormant")
    if all_ages and np.percentile(all_ages, 90) > 15:
        print()
        print("  ! many resurrections are from old dormant entries (p90 > 15)")
        print("    → could indicate stale-matching; consider shorter Δ_dormant")
    print()


def plot_normal_summary(metrics_log: list[FrameResult], path: Path):
    idxs = [r.frame_index for r in metrics_log]
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(idxs, [r.tracks_out for r in metrics_log], label="active tracks", color="C0")
    axes[0].plot(idxs, [r.dormant_buffer_size for r in metrics_log], label="dormant buffer", color="C2")
    axes[0].set_ylabel("# tracks")
    axes[0].set_title("Active vs dormant track populations")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(idxs, [r.tracks_after_ransac for r in metrics_log], label="survived KLT+FB+RANSAC", color="C0")
    axes[1].plot(idxs, [r.new_corners_detected for r in metrics_log], label="new Shi-Tomasi corners (Step 4)", color="C1")
    axes[1].set_ylabel("# features")
    axes[1].set_title("Frame-to-frame data association")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    reid_rate = [r.reids_succeeded / max(1, r.reids_attempted) for r in metrics_log]
    axes[2].plot(idxs, reid_rate, label="re-ID success rate", color="C3")
    axes[2].plot(idxs, [r.reids_succeeded for r in metrics_log],
                 label="re-ID count", color="C4", alpha=0.6)
    axes[2].set_ylabel("rate / count")
    axes[2].set_title("Step 5 (re-ID) activity")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    # Diagnostic signals: per-frame mean and p90 of (a) spatial candidate
    # density, (b) accepted-match Hamming, (c) resurrected entry age.
    # These let us see *where* in the sequence Step 5 is straining.
    def per_frame(key: str, stat: str):
        out = []
        for r in metrics_log:
            xs = getattr(r, key)
            if not xs:
                out.append(np.nan)
            elif stat == "mean":
                out.append(float(np.mean(xs)))
            else:  # p90
                out.append(float(np.percentile(xs, 90)))
        return out

    ax = axes[3]
    ax.plot(idxs, per_frame("spatial_candidates_per_query", "mean"),
            label="spatial candidates/query (mean)", color="C0", alpha=0.7)
    ax.plot(idxs, per_frame("spatial_candidates_per_query", "p90"),
            label="spatial candidates/query (p90)", color="C0", linestyle="--", alpha=0.5)
    ax.plot(idxs, per_frame("accepted_hamming_distances", "mean"),
            label="accepted Hamming (mean)", color="C3", alpha=0.8)
    ax.plot(idxs, per_frame("same_track_hamming_distances", "mean"),
            label="same-track Hamming (mean) [baseline]", color="C5", alpha=0.8)
    ax.plot(idxs, per_frame("resurrected_ages", "mean"),
            label="resurrected age (mean frames)", color="C2", alpha=0.7)
    ax.set_ylabel("value")
    ax.set_xlabel("frame index")
    ax.set_title("Step 5 diagnostics — lower means cleaner re-ID; same-track Hamming is the descriptor-stability baseline")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot:    {path}")


# --------------------------------------------------------------------
# Forced-failure mode (the §10 item-2 test)
# --------------------------------------------------------------------

def run_forced_fail(paths: list[Path], front: HybridFrontend, output: Path,
                    clahe: cv2.CLAHE | None,
                    kill_fraction: float, seed: int,
                    spatial_tolerance_px: float):
    """Run the sequence with synthetic forced KLT failures and measure
    re-ID precision/recall on Step 5.

    Protocol per frame k:
      1. process_frame(k) runs normally → updates positions, kills natural
         KLT failures, spawns new corners, attempts Step 5 re-ID.
      2. We pick ~kill_fraction of the *current* active tracks at random.
      3. For each, record (id, x, y) at the moment of kill. Call these
         "ground-truth kills" with expected resurrection location (x, y).
      4. force_kill() moves them to the dormant buffer with frame_died=k.
      5. On frames k+1, k+2, ... within the dormant horizon, watch for
         a resurrected ID matching the killed one. Also watch for
         "incorrect" resurrections in the same spatial neighbourhood
         (a different ID resurrected within `spatial_tolerance_px` of
         the killed position).

    A "ground-truth kill" event resolves into one of three outcomes:
      CORRECT_REID:    the original id was resurrected at any point
                       within the dormant horizon, within tolerance.
      INCORRECT_REID:  a *different* dormant id was resurrected within
                       tolerance of the killed location (an error of
                       commission — the Step 5 matcher confused two
                       physical features).
      MISSED:          neither happened before the horizon expired.

    These outcomes are mutually exclusive: as soon as one resolves the
    event, it is removed from the watchlist. CORRECT_REID wins ties
    against INCORRECT_REID (i.e. if both happen in the same frame, we
    score correct — the matcher did find the right one even if it also
    incorrectly matched a different candidate).

    Important subtlety: a track may be force-killed AND naturally
    resurrected, then later die naturally and be force-killed AGAIN with
    the same id — actually no, IDs are globally unique and never reused
    once a track is born. So each force-kill event creates a unique
    (id, frame) pair that we can track independently.
    """
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    first = read_gray(paths[0], clahe)
    front.initialize(first)

    # Outcomes per ground-truth kill event.
    @dataclass
    class KillEvent:
        kill_id: int
        kill_x: float
        kill_y: float
        kill_frame: int
        # Expected resurrection location in the CURRENT frame: the kill
        # location propagated forward by the per-frame median flow. The
        # scene moves under the camera, so comparing resurrections
        # against the raw kill pixel silently mis-scores any kill whose
        # gap spans real motion.
        exp_x: float = 0.0
        exp_y: float = 0.0
        # outcome: "correct" / "incorrect" / "missed_no_corner" /
        #          "missed_matcher"
        outcome: str | None = None
        outcome_frame: int | None = None
        incorrect_with: int | None = None  # the other id, if outcome=="incorrect"
        # A foreign id was resurrected within tolerance of the expected
        # location while this event was open (potential ID hijack). The
        # event stays open — the true id can still come back later — and
        # only resolves as "incorrect" if it expires without a correct
        # resurrection.
        hijacked_by: int | None = None
        hijack_frame: int | None = None
        # Whether Step 4 ever produced a new corner within tolerance of
        # the expected location while the event was open. Splits "missed"
        # into a detection failure (no corner ever appeared: Step 4's
        # problem) vs a matcher failure (a corner appeared but Step 5
        # rejected or mis-assigned it).
        corner_seen: bool = False
        # Distance between the correct resurrection and the expected
        # location (diagnostic; only set for outcome=="correct" or the
        # same-id-far case).
        resurrection_dist: float | None = None

    open_events: list[KillEvent] = []
    resolved_events: list[KillEvent] = []
    horizon = front.cfg.dormant_horizon_frames
    image_h, image_w = first.shape
    # Track resurrections per frame so we can compute the expected
    # coincidence rate at the end.
    resurrections_per_frame: list[int] = []

    # Per-frame stats CSV.
    csv_f = open(output / "forced_fail.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame", "active_in", "killed_this_frame",
        "resurrected_total", "open_events",
        "correct_so_far", "incorrect_so_far", "missed_so_far",
    ])

    correct = 0
    incorrect = 0
    missed = 0
    # Sub-counts for interpretation.
    incorrect_hijack = 0        # foreign id resurrected at the kill spot
    incorrect_same_id_far = 0   # right id, but > tolerance from expected
    missed_no_corner = 0        # Step 4 never put a corner there
    missed_matcher = 0          # a corner appeared; Step 5 didn't link it

    for idx in range(1, len(paths)):
        curr = read_gray(paths[idx], clahe)
        active_before = list(front.active_tracks.keys())
        res = front.process_frame(curr)

        # Evaluate every still-open event against this frame's resurrections.
        # `res.resurrected_ids` are the ids that came back in this frame.
        # We need to know their RESURRECTION locations — those are the
        # current positions of the corresponding active tracks (Step 5
        # spawned them as fresh active tracks).
        resurrection_locations = {
            tid: (front.active_tracks[tid].x, front.active_tracks[tid].y)
            for tid in res.resurrected_ids
            if tid in front.active_tracks
        }
        resurrections_per_frame.append(len(resurrection_locations))

        flow_dx, flow_dy = res.median_flow
        still_open: list[KillEvent] = []
        for ev in open_events:
            # Propagate the expected resurrection location by this
            # frame's dominant motion (mirrors the frontend's dormant-
            # buffer motion compensation, so scoring geometry matches
            # matcher geometry).
            ev.exp_x += flow_dx
            ev.exp_y += flow_dy

            # Expiry: more than `horizon` frames since the kill. The
            # dormant entry is gone from the buffer, so the outcome is
            # final: incorrect if the spot was hijacked by a foreign id,
            # otherwise a miss — split by whether Step 4 ever produced a
            # corner there at all.
            if idx - ev.kill_frame > horizon:
                if ev.hijacked_by is not None:
                    ev.outcome = "incorrect"
                    ev.outcome_frame = ev.hijack_frame
                    ev.incorrect_with = ev.hijacked_by
                    incorrect += 1
                    incorrect_hijack += 1
                elif ev.corner_seen:
                    ev.outcome = "missed_matcher"
                    ev.outcome_frame = idx
                    missed += 1
                    missed_matcher += 1
                else:
                    ev.outcome = "missed_no_corner"
                    ev.outcome_frame = idx
                    missed += 1
                    missed_no_corner += 1
                resolved_events.append(ev)
                continue

            # Correct resurrection: the killed id came back. Require it
            # near the (motion-propagated) expected location — a same-id
            # resurrection far away means a different physical corner
            # stole the dead id, which is an error of commission.
            if ev.kill_id in resurrection_locations:
                rx, ry = resurrection_locations[ev.kill_id]
                dist = max(abs(rx - ev.exp_x), abs(ry - ev.exp_y))
                ev.resurrection_dist = dist
                ev.outcome_frame = idx
                if dist <= spatial_tolerance_px:
                    ev.outcome = "correct"
                    correct += 1
                else:
                    ev.outcome = "incorrect"
                    ev.incorrect_with = ev.kill_id
                    incorrect += 1
                    incorrect_same_id_far += 1
                resolved_events.append(ev)
                continue

            # Foreign resurrection within tolerance of the expected
            # location: record as a potential hijack but DO NOT close the
            # event — the true id can still be resurrected on a later
            # frame (the old close-on-first-coincidence logic both
            # inflated 'incorrect' and stole would-be 'correct's).
            if ev.hijacked_by is None:
                for other_id, (rx, ry) in resurrection_locations.items():
                    if (abs(rx - ev.exp_x) <= spatial_tolerance_px and
                            abs(ry - ev.exp_y) <= spatial_tolerance_px):
                        ev.hijacked_by = other_id
                        ev.hijack_frame = idx
                        break

            # Did Step 4 place any new corner near the expected location
            # this frame? (Purely diagnostic; used to split misses.)
            if not ev.corner_seen:
                for cx, cy in res.new_corner_positions:
                    if (abs(cx - ev.exp_x) <= spatial_tolerance_px and
                            abs(cy - ev.exp_y) <= spatial_tolerance_px):
                        ev.corner_seen = True
                        break

            still_open.append(ev)
        open_events = still_open

        # Pick this frame's forced kills from the current active set
        # (after process_frame has updated it).
        # Sample kills only from tracks old enough to have been buffered
        # had they died naturally (mirrors cfg.dormant_min_track_age).
        # This also makes the measured population the meaningful one:
        # established tracks whose identity is worth preserving.
        min_age = front.cfg.dormant_min_track_age
        eligible = [tid for tid, t in front.active_tracks.items()
                    if t.age >= min_age]
        n_kill = int(round(kill_fraction * len(front.active_tracks)))
        kill_ids = rng.sample(eligible, k=min(n_kill, len(eligible)))
        killed_locations = front.force_kill(kill_ids)
        for tid, kx, ky in killed_locations:
            open_events.append(KillEvent(
                kill_id=tid, kill_x=kx, kill_y=ky, kill_frame=idx,
                exp_x=kx, exp_y=ky,
            ))

        csv_w.writerow([
            idx, len(active_before), len(kill_ids),
            len(res.resurrected_ids), len(open_events),
            correct, incorrect, missed,
        ])
        if idx % 25 == 0 or idx == len(paths) - 1:
            print(f"  frame {idx}/{len(paths)-1}  "
                  f"active={res.tracks_out}  killed_now={len(kill_ids)}  "
                  f"open={len(open_events)}  "
                  f"correct={correct}  incorrect={incorrect}  missed={missed}")

    # Anything still open at the end of the sequence resolves with the
    # same taxonomy (these events had less than a full horizon of
    # exposure, but there are at most `horizon` frames' worth of them).
    for ev in open_events:
        if ev.hijacked_by is not None:
            ev.outcome = "incorrect"
            ev.outcome_frame = ev.hijack_frame
            ev.incorrect_with = ev.hijacked_by
            incorrect += 1
            incorrect_hijack += 1
        elif ev.corner_seen:
            ev.outcome = "missed_matcher"
            missed += 1
            missed_matcher += 1
        else:
            ev.outcome = "missed_no_corner"
            missed += 1
            missed_no_corner += 1
        resolved_events.append(ev)
    csv_f.close()

    # ---- Summary ----
    total = correct + incorrect + missed
    if total == 0:
        print("No force-kill events — sequence too short or kill_fraction too low.")
        return
    recall = correct / total
    incorrect_rate = incorrect / total
    precision = correct / max(1, correct + incorrect)
    miss_rate = missed / total

    print()
    print("=" * 60)
    print("FORCED-FAILURE TEST RESULTS")
    print("=" * 60)
    print(f"  Total force-kill events:   {total}")
    print(f"  Correct re-ID (recall):    {correct:6d} ({100*recall:.2f}%)")
    print(f"  Incorrect re-ID:           {incorrect:6d} ({100*incorrect_rate:.2f}%)")
    print(f"      hijacked by foreign id:  {incorrect_hijack:6d}")
    print(f"      same id, > tolerance:    {incorrect_same_id_far:6d}")
    print(f"  Missed (no resurrection):  {missed:6d} ({100*miss_rate:.2f}%)")
    print(f"      no corner ever detected there (Step 4): {missed_no_corner:6d}")
    print(f"      corner appeared, matcher rejected (Step 5): {missed_matcher:6d}")
    print(f"  Precision (of attempts):   {100*precision:.2f}%")

    # ---- Coincidence baseline ----
    # An "incorrect" event is recorded when ANY resurrected track lands
    # within spatial_tolerance_px of a kill location. With many
    # resurrections per frame and a finite image, some of those are
    # geometric coincidences — an unrelated track legitimately re-IDed
    # near where we happened to kill something. Compute the expected
    # rate of such coincidences under a uniform-density model.
    if resurrections_per_frame:
        # Per kill window, the probability that a uniformly-placed
        # resurrection lands inside is roughly tol_area / image_area
        # (we use L∞ window, so tol_area = (2*tol)^2).
        tol_area = (2.0 * spatial_tolerance_px) ** 2
        image_area = float(image_h * image_w)
        p_one_coincidence = min(1.0, tol_area / image_area)
        # Each kill is open for up to `horizon` frames. Sum over the
        # horizon of P(any coincidence in that frame) for an average kill.
        avg_resurrections = float(np.mean(resurrections_per_frame))
        # P(at least one of ~k resurrections lands in window) ≈ k * p
        # (for small k*p; we're well in that regime).
        p_coincidence_per_open_frame = min(1.0, avg_resurrections * p_one_coincidence)
        # An average kill is open for `min(horizon, frames_remaining)` frames.
        # Approximate: assume horizon frames of exposure per kill.
        expected_coincidences = total * (1 - (1 - p_coincidence_per_open_frame) ** horizon)
        # Cap at total (defensive).
        expected_coincidences = min(expected_coincidences, float(total))
        corrected_incorrect = max(0, incorrect - expected_coincidences)
        corrected_incorrect_rate = corrected_incorrect / total
        print()
        print(f"  Coincidence baseline:")
        print(f"    Expected incorrects from chance overlap: {expected_coincidences:6.0f}  "
              f"({100*expected_coincidences/total:.2f}%)")
        print(f"    (kill window {2*spatial_tolerance_px:.0f}px on {image_w}x{image_h} image, "
              f"~{avg_resurrections:.0f} resurrections/frame over {horizon}-frame horizon)")
        print(f"    Coincidence-corrected incorrect:         {corrected_incorrect:6.0f}  "
              f"({100*corrected_incorrect_rate:.2f}%)")

    print()
    print(f"  §10 targets:  recall > 80%   correct: {'PASS' if recall > 0.80 else 'FAIL'}")
    print(f"                incorrect < 1%  incorrect: {'PASS' if incorrect_rate < 0.01 else 'FAIL'}")
    print()

    # Write event log.
    with open(output / "forced_fail_events.csv", "w", newline="") as f:
        ew = csv.writer(f)
        ew.writerow(["kill_id", "kill_x", "kill_y", "kill_frame",
                     "outcome", "outcome_frame", "incorrect_with",
                     "corner_seen", "resurrection_dist"])
        for ev in resolved_events:
            ew.writerow([ev.kill_id, ev.kill_x, ev.kill_y, ev.kill_frame,
                         ev.outcome, ev.outcome_frame, ev.incorrect_with,
                         int(ev.corner_seen),
                         "" if ev.resurrection_dist is None
                         else f"{ev.resurrection_dist:.2f}"])

    plot_forced_fail_summary(
        resolved_events, output / "forced_fail_summary.png",
        correct=correct, incorrect=incorrect, missed=missed,
                         )


def plot_forced_fail_summary(events, path: Path,
                             correct: int, incorrect: int, missed: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Outcome breakdown.
    ax1.bar(["correct", "incorrect", "missed"],
            [correct, incorrect, missed],
            color=["#2ca02c", "#d62728", "#7f7f7f"])
    ax1.set_ylabel("# events")
    ax1.set_title(f"Outcomes of {correct+incorrect+missed} forced-kill events")
    for i, v in enumerate([correct, incorrect, missed]):
        ax1.text(i, v, f" {v}", va="bottom", ha="center")

    # Resolution latency for correct re-IDs.
    latencies = [(ev.outcome_frame - ev.kill_frame)
                 for ev in events
                 if ev.outcome == "correct" and ev.outcome_frame is not None]
    if latencies:
        ax2.hist(latencies, bins=max(5, min(30, max(latencies))),
                 color="#2ca02c", edgecolor="black")
        ax2.set_xlabel("frames between kill and resurrection")
        ax2.set_ylabel("# correct re-IDs")
        ax2.set_title("Latency of correct re-IDs")
    else:
        ax2.text(0.5, 0.5, "no correct re-IDs", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot:    {path}")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--output", type=Path, default=Path("./poc_results"))
    ap.add_argument("--mode", choices=["normal", "forced-fail"], default="normal")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--clahe", action="store_true",
                    help="Apply CLAHE preprocessing (helps with turbid water)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true",
                    help="Skip writing the video (faster)")
    # forced-fail params
    ap.add_argument("--kill-fraction", type=float, default=0.10,
                    help="Fraction of active tracks to force-kill per frame "
                         "(forced-fail mode only). §10 item 2 calls for 10%.")
    ap.add_argument("--spatial-tolerance", type=float, default=5.0,
                    help="Px tolerance for matching a resurrection to a kill "
                         "location (forced-fail mode only). Default 5 px sits "
                         "well below typical inter-corner spacing (~30 px at "
                         "1000 tracks on 720p), so coincidental nearby re-IDs "
                         "of unrelated tracks don't get counted as 'incorrect'. "
                         "Old default of 15 was too loose and overcounted FPs.")
    ap.add_argument("--seed", type=int, default=0)
    # Hybrid config overrides (just the ones likely to want tuning)
    ap.add_argument("--target-tracks", type=int, default=1000)
    ap.add_argument("--reid-radius", type=float, default=20.0,
                    help="r_reid. With motion compensation on, try 8-10.")
    ap.add_argument("--reid-hamming", type=int, default=50,
                    help="θ_reid. With death-time descriptors, calibrate "
                         "from the '1-frame gap' diagnostic (suggested "
                         "value printed at end of a normal run).")
    ap.add_argument("--reid-margin", type=int, default=10,
                    help="Distinctiveness gate: best match must beat the "
                         "second-best spatial candidate by this many "
                         "Hamming bits. 0 disables.")
    ap.add_argument("--dormant-horizon", type=int, default=30)
    ap.add_argument("--min-track-age", type=int, default=3,
                    help="Naturally-dying tracks younger than this are not "
                         "buffered for re-ID (noise suppression). The "
                         "forced-fail kill sample is filtered to tracks at "
                         "least this old.")
    ap.add_argument("--no-motion-comp", action="store_true",
                    help="Disable motion compensation of dormant "
                         "predicted positions.")
    ap.add_argument("--no-dormant-seeding", action="store_true",
                    help="Disable Step 4's preference for corners near "
                         "dormant predicted positions.")
    args = ap.parse_args()

    paths = load_image_paths(args.folder)
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    cfg = HybridConfig(
        target_active_tracks=args.target_tracks,
        reid_radius_px=args.reid_radius,
        reid_hamming_threshold=args.reid_hamming,
        reid_second_best_margin=args.reid_margin,
        dormant_horizon_frames=args.dormant_horizon,
        dormant_min_track_age=args.min_track_age,
        motion_compensate_dormant=not args.no_motion_comp,
        seed_corners_near_dormant=not args.no_dormant_seeding,
    )
    front = HybridFrontend(cfg)

    if args.mode == "normal":
        run_normal(paths, front, args.output, clahe, args.fps,
                   write_video=not args.no_video)
    else:
        run_forced_fail(paths, front, args.output, clahe,
                        kill_fraction=args.kill_fraction,
                        seed=args.seed,
                        spatial_tolerance_px=args.spatial_tolerance)


if __name__ == "__main__":
    main()
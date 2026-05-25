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
        csv_w.writerow([
            res.frame_index, paths[idx].name,
            res.tracks_in, res.tracks_after_klt,
            res.tracks_after_fb, res.tracks_after_ransac,
            res.new_corners_detected, res.reids_attempted, res.reids_succeeded,
            res.tracks_out, res.dormant_buffer_size,
        ])
        if writer is not None:
            vis = draw_frame(curr, front, res)
            writer.write(vis)
        if idx % 25 == 0 or idx == len(paths) - 1:
            print(f"  frame {idx}/{len(paths)-1}  active={res.tracks_out}  "
                  f"reid={res.reids_succeeded}/{res.reids_attempted}  "
                  f"dormant={res.dormant_buffer_size}")

    csv_f.close()
    if writer is not None:
        writer.release()
        print(f"Saved video:   {output / 'hybrid.mp4'}")
    print(f"Saved metrics: {csv_path}")

    if metrics_log:
        plot_normal_summary(metrics_log, output / "summary.png")


def plot_normal_summary(metrics_log: list[FrameResult], path: Path):
    idxs = [r.frame_index for r in metrics_log]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

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

    # Re-ID rate per frame.
    reid_rate = [r.reids_succeeded / max(1, r.reids_attempted) for r in metrics_log]
    axes[2].plot(idxs, reid_rate, label="re-ID success rate", color="C3")
    axes[2].plot(idxs, [r.reids_succeeded for r in metrics_log],
                 label="re-ID count", color="C4", alpha=0.6)
    axes[2].set_ylabel("rate / count")
    axes[2].set_xlabel("frame index")
    axes[2].set_title("Step 5 (re-ID) activity")
    axes[2].legend(); axes[2].grid(alpha=0.3)

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
        outcome: str | None = None      # "correct" / "incorrect" / "missed"
        outcome_frame: int | None = None
        incorrect_with: int | None = None  # the other id, if outcome=="incorrect"

    open_events: list[KillEvent] = []
    resolved_events: list[KillEvent] = []
    horizon = front.cfg.dormant_horizon_frames

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

        still_open: list[KillEvent] = []
        for ev in open_events:
            # Check expiry first: if more than `horizon` frames have
            # passed since the kill, the event is missed.
            if idx - ev.kill_frame > horizon:
                ev.outcome = "missed"
                ev.outcome_frame = idx
                resolved_events.append(ev)
                missed += 1
                continue
            # Check for correct resurrection: id matches AND location is
            # near the killed location.
            if ev.kill_id in resurrection_locations:
                rx, ry = resurrection_locations[ev.kill_id]
                if (abs(rx - ev.kill_x) <= spatial_tolerance_px and
                        abs(ry - ev.kill_y) <= spatial_tolerance_px):
                    ev.outcome = "correct"
                    ev.outcome_frame = idx
                    resolved_events.append(ev)
                    correct += 1
                    continue
                # Otherwise: id matched but the location is way off, which
                # would be very surprising (we don't expect to see this
                # because the matcher's spatial gate is much tighter than
                # the tolerance). Treat it as incorrect.
                ev.outcome = "incorrect"
                ev.outcome_frame = idx
                ev.incorrect_with = ev.kill_id  # same id, wrong place — shouldn't happen, but log it
                resolved_events.append(ev)
                incorrect += 1
                continue
            # Check for incorrect resurrection: any other resurrected id
            # whose location is within tolerance of the killed location.
            incorrect_match = None
            for other_id, (rx, ry) in resurrection_locations.items():
                if (abs(rx - ev.kill_x) <= spatial_tolerance_px and
                        abs(ry - ev.kill_y) <= spatial_tolerance_px):
                    incorrect_match = other_id
                    break
            if incorrect_match is not None:
                ev.outcome = "incorrect"
                ev.outcome_frame = idx
                ev.incorrect_with = incorrect_match
                resolved_events.append(ev)
                incorrect += 1
                continue
            # Not yet resolved.
            still_open.append(ev)
        open_events = still_open

        # Pick this frame's forced kills from the current active set
        # (after process_frame has updated it).
        active_after = list(front.active_tracks.keys())
        n_kill = int(round(kill_fraction * len(active_after)))
        kill_ids = rng.sample(active_after, k=min(n_kill, len(active_after)))
        killed_locations = front.force_kill(kill_ids)
        for tid, kx, ky in killed_locations:
            open_events.append(KillEvent(
                kill_id=tid, kill_x=kx, kill_y=ky, kill_frame=idx,
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

    # Anything still open at the end of the sequence is missed.
    for ev in open_events:
        ev.outcome = "missed"
        ev.outcome_frame = None
        resolved_events.append(ev)
        missed += 1
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
    print(f"  Missed (no resurrection):  {missed:6d} ({100*miss_rate:.2f}%)")
    print(f"  Precision (of attempts):   {100*precision:.2f}%")
    print()
    print(f"  §10 targets:  recall > 80%   correct: {'PASS' if recall > 0.80 else 'FAIL'}")
    print(f"                incorrect < 1%  incorrect: {'PASS' if incorrect_rate < 0.01 else 'FAIL'}")
    print()

    # Write event log.
    with open(output / "forced_fail_events.csv", "w", newline="") as f:
        ew = csv.writer(f)
        ew.writerow(["kill_id", "kill_x", "kill_y", "kill_frame",
                     "outcome", "outcome_frame", "incorrect_with"])
        for ev in resolved_events:
            ew.writerow([ev.kill_id, ev.kill_x, ev.kill_y, ev.kill_frame,
                         ev.outcome, ev.outcome_frame, ev.incorrect_with])

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
    ap.add_argument("--spatial-tolerance", type=float, default=15.0,
                    help="Px tolerance for matching a resurrection to a kill "
                         "location (forced-fail mode only)")
    ap.add_argument("--seed", type=int, default=0)
    # Hybrid config overrides (just the ones likely to want tuning)
    ap.add_argument("--target-tracks", type=int, default=1000)
    ap.add_argument("--reid-radius", type=float, default=20.0)
    ap.add_argument("--reid-hamming", type=int, default=50)
    ap.add_argument("--dormant-horizon", type=int, default=30)
    args = ap.parse_args()

    paths = load_image_paths(args.folder)
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    cfg = HybridConfig(
        target_active_tracks=args.target_tracks,
        reid_radius_px=args.reid_radius,
        reid_hamming_threshold=args.reid_hamming,
        dormant_horizon_frames=args.dormant_horizon,
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

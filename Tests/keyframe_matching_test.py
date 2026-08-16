"""
keyframe_matching_test.py

Design doc §10 item 1: does ORB descriptor matching work across
KEYFRAME-scale and LOOP-scale baselines on this footage?

This is a completely different question from Step 5 re-ID. Step 5
bridges 1-2 frame dropouts and is validated by run_hybrid_poc.py
--mode forced-fail. The questions here are:

  (a) separation sweep — how does match quality decay as the baseline
      grows from 1 frame to ~120 frames? This is the regime that
      TrackLocalMap (§4.7) and keyframe-to-keyframe matching live in,
      and it bounds how far apart keyframes can be placed.

  (b) loop test — when the vehicle returns to the start of the survey
      (~2000 frames later, same corals, opposite traverse direction),
      can descriptor matching recognise the place at all? This bounds
      whether loop closure is feasible, and NOTHING in the frontend can
      substitute for it.

Both are reported against a CHANCE FLOOR measured on this sequence:
pairs of frames far enough apart in time that they cannot overlap. A
loop candidate is only meaningful if it scores clearly above the floor.
That is the same permutation-control logic used in the forced-failure
harness, and it matters here because RANSAC will happily return a
handful of "inliers" for two completely unrelated images.

Usage:
    python keyframe_matching_test.py <image_dir> --mode separation
    python keyframe_matching_test.py <image_dir> --mode loop
    python keyframe_matching_test.py <image_dir> --mode both --output results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from feature_comparison import (  # noqa: E402
    detect_orb_quadtree,
    get_orb,
    load_image_paths,
    read_gray,
)

# ORB-SLAM3's loop closure requires a geometric verification to succeed
# with at least this many inliers before it will accept a candidate.
# (LoopClosing: nInliers >= 20 for the Sim3/SE3 solver to be trusted.)
ORBSLAM3_LOOP_INLIER_MIN = 20
# Descriptor distance ceiling used throughout ORB-SLAM3's matchers.
ORBSLAM3_TH_HIGH = 100
ORBSLAM3_TH_LOW = 50


class FrameStore:
    """Lazily detects and caches ORB keypoints/descriptors per frame."""

    def __init__(self, paths, orb, clahe=None, target_n=1000):
        self.paths = paths
        self.orb = orb
        self.clahe = clahe
        self.target_n = target_n
        self._cache: dict[int, tuple] = {}

    def get(self, idx: int):
        if idx not in self._cache:
            gray = read_gray(self.paths[idx])
            if self.clahe is not None:
                gray = self.clahe.apply(gray)
            kps, des = detect_orb_quadtree(self.orb, gray, target_n=self.target_n)
            self._cache[idx] = (kps, des)
        return self._cache[idx]

    def drop(self, idx: int):
        self._cache.pop(idx, None)


def match_pair(store: FrameStore, i: int, j: int, ratio: float = 0.75):
    """Match two frames the way a place-recognition front-end would.

    Deliberately does NOT use the MAD displacement filter from
    feature_comparison.detect_and_match_orb: that filter assumes small,
    coherent frame-to-frame motion, which is exactly what a wide-baseline
    or loop-closure pair does not have. Using it here would throw away
    the true matches.

    Returns dict with match counts, geometric inliers, and the Hamming
    distance distribution of the verified matches.
    """
    kp1, des1 = store.get(i)
    kp2, des2 = store.get(j)
    out = {
        "frame_a": i, "frame_b": j, "separation": abs(j - i),
        "n_kp_a": len(kp1), "n_kp_b": len(kp2),
        "n_ratio": 0, "n_inliers": 0, "inlier_ratio": 0.0,
        "median_hamming": float("nan"), "median_hamming_inliers": float("nan"),
        "frac_under_th_low": float("nan"),
    }
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return out

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    out["n_ratio"] = len(good)
    if len(good) < 8:
        return out

    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    dists = np.array([m.distance for m in good], dtype=np.float32)
    out["median_hamming"] = float(np.median(dists))
    out["frac_under_th_low"] = float((dists <= ORBSLAM3_TH_LOW).mean())

    try:
        F, mask = cv2.findFundamentalMat(
            p1.reshape(-1, 1, 2), p2.reshape(-1, 1, 2),
            cv2.FM_RANSAC, 3.0, 0.99,
        )
    except cv2.error:
        return out
    if F is None or mask is None or F.shape != (3, 3):
        return out
    mask = mask.flatten().astype(bool)
    out["n_inliers"] = int(mask.sum())
    out["inlier_ratio"] = float(mask.sum() / len(good))
    if mask.any():
        out["median_hamming_inliers"] = float(np.median(dists[mask]))
    return out


def chance_floor(store: FrameStore, n_frames: int, n_samples: int,
                 rng: np.random.Generator, min_separation: int):
    """Match pairs of frames too far apart in time to overlap.

    Whatever inlier counts come back are what RANSAC produces from two
    unrelated images of the same KIND of scene (coral, same camera, same
    turbidity). A real loop candidate must beat this distribution.
    """
    results = []
    for _ in range(n_samples):
        i = int(rng.integers(0, n_frames))
        lo, hi = i + min_separation, n_frames
        if lo >= hi:
            hi_alt = i - min_separation
            if hi_alt <= 0:
                continue
            j = int(rng.integers(0, hi_alt))
        else:
            j = int(rng.integers(lo, hi))
        results.append(match_pair(store, i, j))
        store.drop(j)
    return results


def run_separation(store: FrameStore, n_frames: int, output: Path,
                   separations, n_anchors: int, rng):
    print("=" * 70)
    print("SEPARATION SWEEP — how match quality decays with baseline")
    print("=" * 70)

    max_sep = max(separations)
    anchors = np.linspace(0, max(0, n_frames - max_sep - 1),
                          num=n_anchors, dtype=int)

    rows = []
    print(f"\n  {'sep':>5s} {'n':>5s} {'kp_a':>6s} {'ratio':>7s} "
          f"{'inliers':>8s} {'inl_ratio':>10s} {'medHam(inl)':>12s} "
          f"{'%>=20inl':>9s}")
    for sep in separations:
        res = []
        for a in anchors:
            b = a + sep
            if b >= n_frames:
                continue
            res.append(match_pair(store, int(a), int(b)))
        if not res:
            continue
        inl = np.array([r["n_inliers"] for r in res], dtype=float)
        rat = np.array([r["n_ratio"] for r in res], dtype=float)
        ir = np.array([r["inlier_ratio"] for r in res], dtype=float)
        mh = np.array([r["median_hamming_inliers"] for r in res], dtype=float)
        kp = np.array([r["n_kp_a"] for r in res], dtype=float)
        ok = 100.0 * (inl >= ORBSLAM3_LOOP_INLIER_MIN).mean()
        print(f"  {sep:>5d} {len(res):>5d} {kp.mean():>6.0f} "
              f"{np.median(rat):>7.0f} {np.median(inl):>8.0f} "
              f"{np.mean(ir):>10.2f} {np.nanmedian(mh):>12.1f} {ok:>8.0f}%")
        rows.extend(res)
        # Cache would grow without bound over a long sweep.
        for a in anchors:
            store.drop(int(a) + sep)

    _write_csv(output / "keyframe_separation.csv", rows)

    # Plot: median inliers and median Hamming vs separation.
    seps = sorted({r["separation"] for r in rows})
    med_inl = [np.median([r["n_inliers"] for r in rows if r["separation"] == s])
               for s in seps]
    med_ham = [np.nanmedian([r["median_hamming_inliers"] for r in rows
                             if r["separation"] == s]) for s in seps]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(seps, med_inl, "-o", color="#2ca02c", label="median inliers")
    ax1.axhline(ORBSLAM3_LOOP_INLIER_MIN, color="#2ca02c", ls=":",
                label=f"ORB-SLAM3 loop minimum ({ORBSLAM3_LOOP_INLIER_MIN})")
    ax1.set_xlabel("frame separation")
    ax1.set_ylabel("geometrically verified inliers", color="#2ca02c")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(seps, med_ham, "-s", color="#d62728",
             label="median Hamming (inliers)")
    ax2.axhline(ORBSLAM3_TH_LOW, color="#d62728", ls=":",
                label=f"ORB-SLAM3 TH_LOW ({ORBSLAM3_TH_LOW})")
    ax2.set_ylabel("Hamming distance", color="#d62728")
    ax1.set_title("ORB matching vs frame separation")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "keyframe_separation.png", dpi=110)
    plt.close(fig)
    print(f"\n  Saved: {output/'keyframe_separation.png'}")


def run_loop(store: FrameStore, n_frames: int, output: Path,
             start_frac: float, end_frac: float, n_anchors: int,
             candidate_stride: int, rng, floor_samples: int):
    print()
    print("=" * 70)
    print("LOOP TEST — can the revisit be recognised by descriptors alone?")
    print("=" * 70)

    # Chance floor first, so the loop scores can be read against it.
    # "Unrelated" means far enough apart in time that the views cannot
    # overlap; on a short sequence that separation may not exist, in
    # which case there is no floor to measure and the loop verdict falls
    # back to ORB-SLAM3's inlier minimum alone.
    min_sep = min(max(200, n_frames // 4), max(1, n_frames // 2))
    floor = chance_floor(store, n_frames, floor_samples, rng,
                         min_separation=min_sep)
    f_inl = np.array([r["n_inliers"] for r in floor], dtype=float)
    if f_inl.size >= 10:
        f_p99 = float(np.percentile(f_inl, 99))
        print(f"\n  Chance floor from {len(floor)} unrelated pairs "
              f"(>= {min_sep} frames apart):")
        print(f"    inliers  median={np.median(f_inl):.0f}  "
              f"p90={np.percentile(f_inl, 90):.0f}  p99={f_p99:.0f}  "
              f"max={f_inl.max():.0f}")
        print(f"    A loop candidate must clear p99 ({f_p99:.0f} inliers) to be")
        print(f"    distinguishable from two unrelated frames of coral.")
    else:
        f_p99 = float("-inf")
        print(f"\n  Chance floor: only {f_inl.size} unrelated pairs available "
              f"(sequence too short);")
        print(f"    falling back to the ORB-SLAM3 inlier minimum alone. "
              f"Treat verdicts as optimistic.")

    anchors = np.linspace(0, int(n_frames * start_frac), num=n_anchors,
                          dtype=int)
    cand_lo = int(n_frames * end_frac)
    candidates = list(range(cand_lo, n_frames, candidate_stride))
    print(f"\n  Anchors (start of survey): {list(anchors)}")
    print(f"  Candidates (end of survey): {cand_lo}..{n_frames-1} "
          f"step {candidate_stride}  ({len(candidates)} frames)")

    rows = []
    print(f"\n  {'anchor':>7s} {'best_cand':>10s} {'inliers':>8s} "
          f"{'ratio_m':>8s} {'medHam':>7s} {'verdict':>22s}")
    best_overall = []
    for a in anchors:
        best = None
        for c in candidates:
            r = match_pair(store, int(a), int(c))
            rows.append(r)
            if best is None or r["n_inliers"] > best["n_inliers"]:
                best = r
            store.drop(c)
        if best is None:
            continue
        best_overall.append(best)
        clears_floor = best["n_inliers"] > f_p99
        clears_min = best["n_inliers"] >= ORBSLAM3_LOOP_INLIER_MIN
        if clears_min and clears_floor:
            verdict = "LOOP DETECTABLE"
        elif clears_floor:
            verdict = "above chance, below min"
        else:
            verdict = "indistinguishable from chance"
        print(f"  {a:>7d} {best['frame_b']:>10d} {best['n_inliers']:>8d} "
              f"{best['n_ratio']:>8d} {best['median_hamming_inliers']:>7.1f} "
              f"{verdict:>22s}")

    _write_csv(output / "keyframe_loop.csv", rows)

    n_detect = sum(1 for b in best_overall
                   if b["n_inliers"] >= ORBSLAM3_LOOP_INLIER_MIN
                   and b["n_inliers"] > f_p99)
    print(f"\n  {n_detect}/{len(best_overall)} anchors produced a loop "
          f"candidate that clears BOTH the chance floor and ORB-SLAM3's "
          f"{ORBSLAM3_LOOP_INLIER_MIN}-inlier minimum.")
    if n_detect == 0:
        print("  => Descriptor-only loop closure is NOT viable on this")
        print("     sequence as configured. The revisit is not recognisable")
        print("     from ORB descriptors, so DBoW2 would not fire either.")
    elif n_detect < len(best_overall) / 2:
        print("  => Marginal: some revisit geometry is recoverable, but")
        print("     loop closure would be unreliable.")
    else:
        print("  => Descriptor-based loop closure looks viable.")

    # Plot: inliers vs candidate frame, per anchor, against the floor.
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in anchors:
        sub = [r for r in rows if r["frame_a"] == a]
        if not sub:
            continue
        ax.plot([r["frame_b"] for r in sub], [r["n_inliers"] for r in sub],
                lw=1, label=f"anchor {a}")
    if np.isfinite(f_p99):
        ax.axhline(f_p99, color="k", ls="--",
                   label=f"chance floor p99 ({f_p99:.0f})")
    ax.axhline(ORBSLAM3_LOOP_INLIER_MIN, color="r", ls=":",
               label=f"ORB-SLAM3 loop min ({ORBSLAM3_LOOP_INLIER_MIN})")
    ax.set_xlabel("candidate frame (end of survey)")
    ax.set_ylabel("geometrically verified inliers")
    ax.set_title("Loop-closure candidate search: start frames vs end frames")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "keyframe_loop.png", dpi=110)
    plt.close(fig)
    print(f"  Saved: {output/'keyframe_loop.png'}")


def _write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("keyframe_results"))
    ap.add_argument("--mode", choices=["separation", "loop", "both"],
                    default="both")
    ap.add_argument("--target-features", type=int, default=1000)
    ap.add_argument("--clahe", action="store_true",
                    help="Apply CLAHE before detection (turbid footage).")
    ap.add_argument("--separations", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20, 30, 60, 120])
    ap.add_argument("--anchors", type=int, default=25,
                    help="Anchor frames per separation.")
    ap.add_argument("--loop-anchors", type=int, default=5)
    ap.add_argument("--loop-start-frac", type=float, default=0.05,
                    help="Anchors are drawn from the first this-fraction "
                         "of the sequence.")
    ap.add_argument("--loop-end-frac", type=float, default=0.85,
                    help="Candidates are drawn from this fraction onward.")
    ap.add_argument("--candidate-stride", type=int, default=10)
    ap.add_argument("--floor-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = load_image_paths(args.image_dir)
    if not paths:
        print(f"No images in {args.image_dir}")
        return 1
    print(f"Found {len(paths)} images in {args.image_dir}")
    args.output.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
        if args.clahe else None
    store = FrameStore(paths, get_orb(), clahe=clahe,
                       target_n=args.target_features)
    rng = np.random.default_rng(args.seed)

    if args.mode in ("separation", "both"):
        run_separation(store, len(paths), args.output,
                       args.separations, args.anchors, rng)
    if args.mode in ("loop", "both"):
        run_loop(store, len(paths), args.output,
                 args.loop_start_frac, args.loop_end_frac,
                 args.loop_anchors, args.candidate_stride, rng,
                 args.floor_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
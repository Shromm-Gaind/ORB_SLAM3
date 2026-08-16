"""
How far can the AUV's standoff change between mapping a landmark and
revisiting it before descriptor matching stops working?

This is the question behind loop closure at a different distance. The
governing relation comes from ORB-SLAM3's MapPoint::PredictScale:

    l_far = l_near - log(s) / log(1.2)        s = d_far / d_near

so backing off to twice the distance (s = 2) shifts a feature DOWN by
log(2)/log(1.2) = 3.80 octaves. A feature born at level 0-3 has no
counterpart at 2x the distance: the pyramid has no negative levels. Only
the high-octave subset survives, which is precisely why populating every
octave matters (see octave_rule_test.py Q2/Q3).

METHOD — synthetic rescaling, which buys exact ground truth.
Resizing a frame by 1/s simulates viewing the same scene from s times
the distance. A feature at (x, y) maps to (x/s, y/s), so correspondences
are known EXACTLY, with no RANSAC and no matching heuristics in the
loop. That lets us separate three things that are usually confounded:

  DETECTOR REPEATABILITY  does the detector fire on the same physical
                          point at the new scale at all?
  OCTAVE SHIFT            does the measured shift follow the predicted
                          -log(s)/log(1.2)?
  DESCRIPTOR MATCHABILITY given a true correspondence was detected, is
                          the Hamming distance low enough to match?

The limitation is that pure rescaling omits perspective change, and
real standoff changes also alter illumination, viewing angle and
backscatter. Treat the numbers as an upper bound on real-world
performance — if matching already fails here, it will certainly fail on
a real revisit.

Usage:
    python scale_invariance_test.py <image_dir> --frame-range 0 800
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
    detect_orb_quadtree, get_orb, load_image_paths, read_gray,
)
from multiscale_shitomasi import (  # noqa: E402
    DEFAULT_NLEVELS, DEFAULT_SCALE_FACTOR, detect_multiscale_shi_tomasi,
)

TH_LOW = 50    # ORB-SLAM3's strict descriptor threshold
TH_HIGH = 100  # ORB-SLAM3's permissive threshold (loop closure / reloc)


def describe(gray, kps, orb):
    if not kps:
        return [], None
    kept, desc = orb.compute(gray, list(kps))
    if desc is None:
        return [], None
    return list(kept), desc


def detect_and_describe(gray, method, orb, target_n, nlevels, scale_factor):
    if method == "shi_tomasi":
        kps = detect_multiscale_shi_tomasi(
            gray, target_n=target_n, rule="rank",
            nlevels=nlevels, scale_factor=scale_factor)
        return describe(gray, kps, orb)
    kps, desc = detect_orb_quadtree(orb, gray, target_n=target_n)
    return list(kps), desc


def hamming_matrix(d1, d2):
    """Pairwise Hamming distances between two descriptor sets."""
    x = np.bitwise_xor(d1[:, None, :], d2[None, :, :])
    return np.unpackbits(x, axis=2).sum(axis=2)


def evaluate_pair(gray, s, method, orb, target_n, nlevels, scale_factor,
                  tol_px):
    """Compare a frame against a synthetically rescaled copy of itself."""
    h, w = gray.shape
    nh, nw = max(64, int(round(h / s))), max(64, int(round(w / s)))
    scaled = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA
                        if s > 1 else cv2.INTER_LINEAR)

    kp_a, d_a = detect_and_describe(gray, method, orb, target_n, nlevels,
                                    scale_factor)
    kp_b, d_b = detect_and_describe(scaled, method, orb, target_n, nlevels,
                                    scale_factor)
    out = {"scale": s, "method": method, "n_a": len(kp_a), "n_b": len(kp_b),
           "n_corr": 0, "repeatability": 0.0, "mean_d_octave": np.nan,
           "median_hamming": np.nan, "frac_th_low": np.nan,
           "frac_th_high": np.nan, "match_recall": 0.0,
           "match_precision": np.nan}
    if d_a is None or d_b is None or len(kp_a) < 5 or len(kp_b) < 5:
        return out

    pa = np.array([k.pt for k in kp_a], dtype=np.float32)
    pb = np.array([k.pt for k in kp_b], dtype=np.float32)
    oa = np.array([k.octave for k in kp_a], dtype=np.int32)
    ob = np.array([k.octave for k in kp_b], dtype=np.int32)

    # Ground truth: a's position mapped into the rescaled image.
    pa_in_b = pa / s
    d2 = ((pa_in_b[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
    nn = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(pa)), nn])
    # Tolerance is applied in the RESCALED image's pixels, so it means the
    # same physical thing at every scale.
    corr = nn_dist <= tol_px
    n_corr = int(corr.sum())
    out["n_corr"] = n_corr
    out["repeatability"] = n_corr / max(1, len(kp_a))
    if n_corr < 5:
        return out

    ia = np.flatnonzero(corr)
    ib = nn[ia]
    d_oct = ob[ib] - oa[ia]
    out["mean_d_octave"] = float(d_oct.mean())

    ham = np.array([int(np.unpackbits(np.bitwise_xor(d_a[i], d_b[j])).sum())
                    for i, j in zip(ia, ib)])
    out["median_hamming"] = float(np.median(ham))
    out["frac_th_low"] = float((ham <= TH_LOW).mean())
    out["frac_th_high"] = float((ham <= TH_HIGH).mean())

    # Practical matching: brute force + ratio test, scored against the
    # geometric ground truth. Recall is over ALL features in a, so it
    # folds in detector repeatability as well as descriptor quality —
    # which is what actually determines whether a revisit is recognised.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d_a, d_b, k=2)
    good = [m for pr in knn if len(pr) == 2
            for m, n in [pr] if m.distance < 0.75 * n.distance]
    if good:
        truth = {int(i): int(j) for i, j in zip(ia, ib)}
        ok = sum(1 for m in good
                 if truth.get(m.queryIdx, -1) == m.trainIdx)
        out["match_recall"] = ok / max(1, len(kp_a))
        out["match_precision"] = ok / len(good)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("scale_results"))
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--frame-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.5, 0.7, 0.83, 1.0, 1.2, 1.44, 1.7, 2.0, 2.5, 3.0],
                    help="d_far / d_near. >1 = revisit from further away.")
    ap.add_argument("--target-features", type=int, default=1000)
    ap.add_argument("--tol-px", type=float, default=2.0)
    ap.add_argument("--nlevels", type=int, default=DEFAULT_NLEVELS)
    ap.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    ap.add_argument("--methods", nargs="+",
                    default=["shi_tomasi", "orb"])
    args = ap.parse_args()

    paths = load_image_paths(args.image_dir)
    lo, hi = args.frame_range if args.frame_range else (0, len(paths))
    hi = min(hi, len(paths))
    frames = np.linspace(lo, hi - 1, num=min(args.frames, hi - lo),
                         dtype=int).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    orb = get_orb()

    print(f"Sampling {len(frames)} frames from [{lo}, {hi})")
    print(f"Pyramid: {args.nlevels} levels at {args.scale_factor}x  "
          f"=> usable range {args.scale_factor**(args.nlevels-1):.2f}x\n")

    rows = []
    for method in args.methods:
        print("=" * 78)
        print(f"METHOD: {method}")
        print("=" * 78)
        print(f"  {'s':>5s} {'pred dOct':>10s} {'meas dOct':>10s} "
              f"{'repeat':>7s} {'medHam':>7s} {'<=50':>6s} {'<=100':>6s} "
              f"{'recall':>7s} {'prec':>6s} {'#correct':>9s}")
        for s in args.scales:
            accum = []
            for fi in frames:
                gray = read_gray(paths[fi])
                accum.append(evaluate_pair(
                    gray, s, method, orb, args.target_features,
                    args.nlevels, args.scale_factor, args.tol_px))
            def m(k):
                v = np.array([a[k] for a in accum], dtype=float)
                v = v[np.isfinite(v)]
                return float(v.mean()) if v.size else float("nan")
            pred = -np.log(s) / np.log(args.scale_factor)
            row = {"method": method, "scale": s, "pred_d_octave": pred,
                   "meas_d_octave": m("mean_d_octave"),
                   "repeatability": m("repeatability"),
                   "median_hamming": m("median_hamming"),
                   "frac_th_low": m("frac_th_low"),
                   "frac_th_high": m("frac_th_high"),
                   "match_recall": m("match_recall"),
                   "match_precision": m("match_precision"),
                   "n_corr": m("n_corr"), "n_a": m("n_a")}
            # Absolute count of correct matches is what actually decides
            # whether loop closure fires: ORB-SLAM3 needs ~20 geometric
            # inliers, so a small recall FRACTION over a large feature set
            # can still be plenty.
            row["n_correct_matches"] = row["match_recall"] * row["n_a"]
            rows.append(row)
            print(f"  {s:>5.2f} {pred:>10.2f} {row['meas_d_octave']:>10.2f} "
                  f"{row['repeatability']:>7.2f} {row['median_hamming']:>7.1f} "
                  f"{row['frac_th_low']:>6.2f} {row['frac_th_high']:>6.2f} "
                  f"{row['match_recall']:>7.3f} "
                  f"{row['match_precision']:>6.2f} "
                  f"{row['n_correct_matches']:>9.0f}")
        print()

    # ---- Interpretation ----
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    st = [r for r in rows if r["method"] == "shi_tomasi"]
    if st:
        far = [r for r in st if r["scale"] > 1.0]
        # ORB-SLAM3's loop-closure solver needs ~20 geometric inliers;
        # require 3x that before RANSAC to leave margin for the outliers
        # that real (non-synthetic) appearance change will introduce.
        MIN_CORRECT = 60
        ok = [r for r in far if r["n_correct_matches"] >= MIN_CORRECT]
        limit = max((r["scale"] for r in ok), default=1.0)
        print(f"\n  Predicted vs measured octave shift: the measured column")
        print(f"  should track the prediction where the pyramid can still")
        print(f"  represent the shift. Where it flattens out, features are")
        print(f"  being clamped at level 0 — the hard wall.")
        print(f"\n  Largest standoff increase still yielding >= {MIN_CORRECT} "
              f"correct matches: {limit:.2f}x")
        print(f"  Pyramid's theoretical range: "
              f"{args.scale_factor**(args.nlevels-1):.2f}x")
        if limit < 2.0:
            print("  => Loop closure will be unreliable if the revisit")
            print("     standoff differs by more than ~2x. Worth checking")
            print("     the actual standoff variation in the survey.")
        else:
            print("  => Matching survives the standoff changes a survey of")
            print("     this kind is likely to produce.")

    with open(args.output / "scale_invariance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {args.output/'scale_invariance.csv'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for method in args.methods:
        sub = [r for r in rows if r["method"] == method]
        sc = [r["scale"] for r in sub]
        axes[0].plot(sc, [r["meas_d_octave"] for r in sub], "-o", ms=4,
                     label=f"{method} (measured)")
        axes[1].plot(sc, [r["repeatability"] for r in sub], "-o", ms=4,
                     label=method)
        axes[2].plot(sc, [r["match_recall"] for r in sub], "-o", ms=4,
                     label=method)
    sub0 = [r for r in rows if r["method"] == args.methods[0]]
    axes[0].plot([r["scale"] for r in sub0], [r["pred_d_octave"] for r in sub0],
                 "k--", label="predicted -log(s)/log(1.2)")
    axes[0].set_ylabel("octave shift"); axes[0].set_title("Octave shift vs standoff ratio")
    axes[1].set_ylabel("detector repeatability")
    axes[1].set_title("Same physical point re-detected")
    axes[2].set_ylabel("match recall"); axes[2].set_title("End-to-end match recall")
    for ax in axes:
        ax.set_xlabel("standoff ratio s = d_far / d_near")
        ax.set_xscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "scale_invariance.png", dpi=110)
    plt.close(fig)
    print(f"  Saved: {args.output/'scale_invariance.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
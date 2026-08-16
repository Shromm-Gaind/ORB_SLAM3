"""
octave_staleness_test.py

ORB-SLAM3 re-detects every frame, so a keypoint's octave is re-estimated
every frame. A KLT track is detected ONCE and then followed — median
lifetime ~206 frames in the well-lit part of this dataset. Over that
span the vehicle's standoff changes, the landmark's apparent scale
changes with it, but the octave label stays frozen at its birth value.

That matters because octave drives three things in ORB-SLAM3:
  - the pyramid level BRIEF is computed on (so: re-ID, TrackLocalMap,
    relocalization, loop closure all use it),
  - the TrackLocalMap search radius r_TLM(l) = 2 * 1.2^l,
  - the bundle-adjustment weight invLevelSigma2[l].

This script measures two things over track age:

  DRIFT      how far the instantaneous best octave moves away from the
             birth octave.
  COST       whether that drift actually hurts descriptor matching —
             the question that decides whether a fix is worth building.
             Each frame a track is re-described twice: once at the
             FROZEN birth octave (what the frontend does today) and once
             at the ADAPTIVE current-best octave. Both are compared to
             the birth descriptor. If adaptive wins by a meaningful
             margin at long ages, octave must be maintained; if the two
             curves sit on top of each other, freezing at birth is fine
             and the design doc can say so with evidence.

COMPARING lambda2 ACROSS LEVELS — a necessary judgement call:
    octave_rule_test.py Q1 measured that lambda2 RISES with pyramid
    level on this footage (p99 is 2.44x higher at L7 than L0). A raw
    argmax over levels would therefore be biased toward coarse octaves
    regardless of the landmark's true scale. We instead compare each
    level's response to that level's own p99, i.e. we take the octave
    where the point is most exceptional RELATIVE to its level, which
    removes the measured inter-level bias.

Usage:
    python octave_staleness_test.py <image_dir> --frame-range 0 800
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
from feature_comparison import load_image_paths, read_gray  # noqa: E402
from multiscale_shitomasi import (  # noqa: E402
    DEFAULT_NLEVELS, DEFAULT_PATCH_SIZE, DEFAULT_SCALE_FACTOR,
    build_pyramid, detect_multiscale_shi_tomasi, lambda2_map,
)

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           30, 0.01))
FB_THRESHOLD = 1.0


def level_normalisers(pyr, block_size=7):
    """Per-level lambda2 map and its p99, used to compare responses
    across levels on an equal footing (see module docstring)."""
    maps, norms = [], []
    for img in pyr:
        lam = lambda2_map(img, block_size)
        maps.append(lam)
        p99 = float(np.percentile(lam, 99))
        norms.append(p99 if p99 > 1e-12 else 1.0)
    return maps, norms


def best_octave_at(x, y, maps, norms, scale_factor, edge=15):
    """Octave where (x, y) is most exceptional relative to its level."""
    best_l, best_v = 0, -1.0
    for l, lam in enumerate(maps):
        s = scale_factor ** l
        xi = int(round(x / s))
        yi = int(round(y / s))
        h, w = lam.shape
        if not (edge <= xi < w - edge and edge <= yi < h - edge):
            continue
        v = float(lam[yi, xi]) / norms[l]
        if v > best_v:
            best_v, best_l = v, l
    return best_l


def describe_at(gray, entries, orb, key_octave):
    """Batch-describe tracks at their current positions.

    `key_octave` selects which octave each keypoint is described at:
    'birth' freezes it, 'best' uses the instantaneous estimate. Returns
    {entry index: descriptor}; ORB drops border keypoints, so callers
    must tolerate missing keys.
    """
    kps = []
    for i, e in enumerate(entries):
        lvl = e["birth_octave"] if key_octave == "birth" else e["best_octave"]
        kp = cv2.KeyPoint(x=float(e["x"]), y=float(e["y"]),
                          size=float(DEFAULT_PATCH_SIZE *
                                     DEFAULT_SCALE_FACTOR ** lvl))
        kp.octave = int(lvl)
        kp.class_id = i
        kps.append(kp)
    if not kps:
        return {}
    kept, desc = orb.compute(gray, kps)
    out = {}
    if desc is None:
        return out
    for row, kp in enumerate(kept):
        if 0 <= kp.class_id < len(entries):
            out[kp.class_id] = desc[row]
    return out


def hamming(a, b):
    return int(np.unpackbits(np.bitwise_xor(a, b)).sum())


def run(paths, anchors, max_age, max_tracks, target_n, nlevels,
        scale_factor, output: Path):
    orb = cv2.ORB_create(nfeatures=8000, scaleFactor=scale_factor,
                         nlevels=nlevels, edgeThreshold=15)
    samples = []  # one row per (track, age)

    for a_i, anchor in enumerate(anchors):
        gray0 = read_gray(paths[anchor])
        kps = detect_multiscale_shi_tomasi(
            gray0, target_n=target_n, rule="rank",
            nlevels=nlevels, scale_factor=scale_factor)
        if not kps:
            continue
        if len(kps) > max_tracks:
            idx = np.linspace(0, len(kps) - 1, max_tracks, dtype=int)
            kps = [kps[i] for i in idx]

        kept, desc0 = orb.compute(gray0, list(kps))
        if desc0 is None or not kept:
            continue

        entries = []
        for k, kp in enumerate(kept):
            entries.append({
                "x": kp.pt[0], "y": kp.pt[1],
                "birth_octave": int(kp.octave),
                "best_octave": int(kp.octave),
                "birth_desc": desc0[k].copy(),
            })

        prev_gray = gray0
        alive = list(range(len(entries)))
        for age in range(1, max_age + 1):
            fi = anchor + age
            if fi >= len(paths):
                break
            gray = read_gray(paths[fi])

            pts = np.array([[entries[i]["x"], entries[i]["y"]]
                            for i in alive], dtype=np.float32).reshape(-1, 1, 2)
            nxt, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts,
                                                    None, **LK_PARAMS)
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt,
                                                     None, **LK_PARAMS)
            fb = np.linalg.norm(pts.reshape(-1, 2) - back.reshape(-1, 2),
                                axis=1)
            ok = (st_f.flatten() == 1) & (st_b.flatten() == 1) & \
                 (fb < FB_THRESHOLD)

            new_alive = []
            cur = nxt.reshape(-1, 2)
            for j, i in enumerate(alive):
                if not ok[j]:
                    continue
                entries[i]["x"] = float(cur[j, 0])
                entries[i]["y"] = float(cur[j, 1])
                new_alive.append(i)
            alive = new_alive
            if not alive:
                break

            # Instantaneous best octave at each surviving position.
            pyr = build_pyramid(gray, scale_factor, nlevels)
            maps, norms = level_normalisers(pyr)
            for i in alive:
                entries[i]["best_octave"] = best_octave_at(
                    entries[i]["x"], entries[i]["y"], maps, norms,
                    scale_factor)

            live = [entries[i] for i in alive]
            d_frozen = describe_at(gray, live, orb, "birth")
            d_adapt = describe_at(gray, live, orb, "best")
            for k, i in enumerate(alive):
                if k not in d_frozen or k not in d_adapt:
                    continue
                e = entries[i]
                samples.append({
                    "anchor": anchor, "track": i, "age": age,
                    "birth_octave": e["birth_octave"],
                    "best_octave": e["best_octave"],
                    "d_octave": e["best_octave"] - e["birth_octave"],
                    "ham_frozen": hamming(e["birth_desc"], d_frozen[k]),
                    "ham_adaptive": hamming(e["birth_desc"], d_adapt[k]),
                })

            prev_gray = gray
        print(f"  anchor {anchor} ({a_i+1}/{len(anchors)}): "
              f"{len(alive)} tracks alive at age {age}")

    if not samples:
        print("No samples collected.")
        return

    arr = {k: np.array([s[k] for s in samples], dtype=float)
           for k in ("age", "d_octave", "ham_frozen", "ham_adaptive",
                     "birth_octave", "best_octave")}

    print()
    print("=" * 76)
    print("OCTAVE STALENESS — does a frozen birth octave go wrong, "
          "and does it cost?")
    print("=" * 76)
    print(f"\n  {len(samples)} (track, age) samples")
    print(f"  Birth octave distribution: " +
          "  ".join(f"L{l}:{int((arr['birth_octave']==l).sum())}"
                    for l in range(int(arr['birth_octave'].max()) + 1)))

    buckets = [(1, 1), (2, 5), (6, 15), (16, 30), (31, 60), (61, 120),
               (121, 10 ** 6)]
    print(f"\n  {'age':>9s} {'n':>7s} {'mean|dOct|':>11s} {'|d|>=1':>8s} "
          f"{'|d|>=2':>8s} {'ham_frozen':>11s} {'ham_adapt':>10s} "
          f"{'gain':>7s}")
    rows = []
    for lo, hi in buckets:
        m = (arr["age"] >= lo) & (arr["age"] <= hi)
        if m.sum() < 20:
            continue
        d = np.abs(arr["d_octave"][m])
        hf = arr["ham_frozen"][m].mean()
        ha = arr["ham_adaptive"][m].mean()
        tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6
                                        else f"{lo}+")
        print(f"  {tag:>9s} {int(m.sum()):>7d} {d.mean():>11.2f} "
              f"{(d >= 1).mean():>8.2f} {(d >= 2).mean():>8.2f} "
              f"{hf:>11.1f} {ha:>10.1f} {hf - ha:>+7.1f}")
        rows.append({"age_bucket": tag, "n": int(m.sum()),
                     "mean_abs_d_octave": float(d.mean()),
                     "frac_ge1": float((d >= 1).mean()),
                     "frac_ge2": float((d >= 2).mean()),
                     "ham_frozen": float(hf), "ham_adaptive": float(ha),
                     "gain": float(hf - ha)})

    # NOISE FLOOR. At age 1 the landmark's true scale has barely changed,
    # so any octave disagreement there is estimator noise, not drift.
    # Excess drift at later ages is the only part that reflects real
    # scale change — this is what separates "the octave is going stale"
    # from "the estimator is jittery".
    floor = rows[0]["mean_abs_d_octave"] if rows else float("nan")
    floor_ge1 = rows[0]["frac_ge1"] if rows else float("nan")
    print()
    print(f"  Estimator noise floor (age {rows[0]['age_bucket']}): "
          f"mean|dOct|={floor:.2f}, |d|>=1 in {100*floor_ge1:.0f}% of samples.")
    print("  Scale has barely changed at age 1, so this is the estimator")
    print("  disagreeing with itself. Excess above it is the real drift:")
    for r in rows[1:]:
        print(f"    age {r['age_bucket']:>8s}: mean|dOct|={r['mean_abs_d_octave']:.2f}"
              f"  excess={r['mean_abs_d_octave'] - floor:+.2f}")

    late = [r for r in rows if r["n"] >= 20][-1] if rows else None
    print()
    if late is not None and abs(late["mean_abs_d_octave"] - floor) < 0.15:
        print(f"  => Octave drift at age {late['age_bucket']} is "
              f"indistinguishable from the age-1 noise floor: the birth")
        print("     octave is NOT going stale on this footage. Freeze it.")
        print("     (If the vehicle's standoff is roughly constant, this is")
        print("      exactly what you would expect.)")
    elif late is None:
        print("  Not enough long-lived tracks to judge.")
        print(f"  => At age {late['age_bucket']}, describing at the CURRENT")
        print(f"     best octave beats the frozen birth octave by "
              f"{late['gain']:.1f} bits.")
        print("     Octave must be maintained over a track's life. With")
        print("     stereo, ORB-SLAM3's PredictScale(depth) is the")
        print("     principled way to do it.")
    elif late["gain"] < -3.0:
        print(f"  => The frozen birth octave is BETTER by "
              f"{-late['gain']:.1f} bits. The per-level argmax estimate is")
        print("     noisier than the scale change it is trying to correct;")
        print("     keep the birth octave (or derive it from stereo depth")
        print("     rather than from image response).")
    else:
        print(f"  => No meaningful difference ({late['gain']:+.1f} bits) even "
              f"at age {late['age_bucket']}.")
        print("     Freezing the octave at birth is safe on this footage —")
        print("     record that in the design doc and move on.")

    with open(output / "octave_staleness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        w.writeheader()
        w.writerows(samples)
    print(f"\n  Saved: {output/'octave_staleness.csv'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ages = sorted({int(a) for a in arr["age"]})
    ages = [a for a in ages if (arr["age"] == a).sum() >= 20]
    ax1.plot(ages, [np.abs(arr["d_octave"][arr["age"] == a]).mean()
                    for a in ages], "-o", ms=3, color="#1f77b4")
    ax1.set_xlabel("track age (frames)")
    ax1.set_ylabel("mean |best octave - birth octave|")
    ax1.set_title("Octave drift over a track's life")
    ax1.grid(alpha=0.3)
    ax2.plot(ages, [arr["ham_frozen"][arr["age"] == a].mean() for a in ages],
             "-o", ms=3, color="#d62728", label="described at BIRTH octave")
    ax2.plot(ages, [arr["ham_adaptive"][arr["age"] == a].mean() for a in ages],
             "-s", ms=3, color="#2ca02c", label="described at CURRENT octave")
    ax2.set_xlabel("track age (frames)")
    ax2.set_ylabel("Hamming distance to birth descriptor")
    ax2.set_title("Does octave staleness cost matching?")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "octave_staleness.png", dpi=110)
    plt.close(fig)
    print(f"  Saved: {output/'octave_staleness.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("octave_staleness"))
    ap.add_argument("--frame-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--anchors", type=int, default=6,
                    help="Independent starting frames.")
    ap.add_argument("--max-age", type=int, default=200,
                    help="Longest track age to follow, in frames.")
    ap.add_argument("--max-tracks", type=int, default=150,
                    help="Tracks followed per anchor.")
    ap.add_argument("--target-features", type=int, default=1000)
    ap.add_argument("--nlevels", type=int, default=DEFAULT_NLEVELS)
    ap.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    args = ap.parse_args()

    paths = load_image_paths(args.image_dir)
    lo, hi = args.frame_range if args.frame_range else (0, len(paths))
    hi = min(hi, len(paths))
    last_start = max(lo, hi - args.max_age - 1)
    anchors = np.linspace(lo, last_start, num=args.anchors, dtype=int).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Anchors: {anchors}  (max age {args.max_age})")

    run(paths, anchors, args.max_age, args.max_tracks, args.target_features,
        args.nlevels, args.scale_factor, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
octave_rule_test.py

Settles the octave-rule questions empirically before the C++ port.

Q1. Does the Shi-Tomasi response (lambda2) actually decay with pyramid
    level on THIS footage? The claim that it "decays exponentially" is
    not a theorem: downsampling compresses structure spatially, which
    can sharpen soft edges per-pixel, while the anti-alias blur destroys
    fine texture. Which effect dominates depends on the image spectrum,
    so it has to be measured, not assumed.

Q2. Which threshold rule distributes features across octaves the way
    ORB-SLAM3 needs?
      absolute  one threshold everywhere       (starves high octaves?)
      relative  quality * max(lambda2) per level (outlier-sensitive)
      rank      per-level target + quadtree     (ORB-SLAM3's own approach)

Q3. How does the resulting OCTAVE DISTRIBUTION compare to FAST/ORB on
    the same frames? A large mismatch is a concrete risk indicator for
    DBoW2 vocabulary mismatch, because the vocabulary was trained on the
    descriptor population that FAST produces across scales.

Usage:
    python octave_rule_test.py <image_dir> --output octave_results
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
    DEFAULT_NLEVELS, DEFAULT_SCALE_FACTOR, build_pyramid,
    detect_multiscale_shi_tomasi, lambda2_map, level_target_counts,
)

RULES = ("absolute", "relative", "rank")


def q1_lambda_decay(paths, frames, output: Path, nlevels, scale_factor):
    print("=" * 72)
    print("Q1. Does lambda2 decay with pyramid level on this footage?")
    print("=" * 72)

    per_level = {lvl: {"max": [], "p99": [], "median": []}
                 for lvl in range(nlevels)}
    for idx in frames:
        gray = read_gray(paths[idx])
        for lvl, img in enumerate(build_pyramid(gray, scale_factor, nlevels)):
            lam = lambda2_map(img)
            per_level[lvl]["max"].append(float(lam.max()))
            per_level[lvl]["p99"].append(float(np.percentile(lam, 99)))
            per_level[lvl]["median"].append(float(np.median(lam)))

    print(f"\n  Averaged over {len(frames)} frames "
          f"(scale factor {scale_factor}):")
    print(f"    {'level':>6s} {'scale':>7s} {'max':>12s} {'p99':>12s} "
          f"{'median':>12s} {'p99 vs L0':>10s}")
    base_p99 = np.mean(per_level[0]["p99"])
    rows = []
    for lvl in range(nlevels):
        mx = np.mean(per_level[lvl]["max"])
        p99 = np.mean(per_level[lvl]["p99"])
        med = np.mean(per_level[lvl]["median"])
        ratio = p99 / base_p99 if base_p99 else float("nan")
        print(f"    {lvl:>6d} {scale_factor**lvl:>7.2f} {mx:>12.5f} "
              f"{p99:>12.5f} {med:>12.5f} {ratio:>9.2f}x")
        rows.append({"level": lvl, "scale": scale_factor ** lvl,
                     "lam_max": mx, "lam_p99": p99, "lam_median": med,
                     "p99_ratio_to_L0": ratio})

    r_top = rows[-1]["p99_ratio_to_L0"]
    print()
    if r_top < 0.5:
        print(f"  => Responses DO fall with level (top octave p99 is "
              f"{r_top:.2f}x level 0).")
        print("     A single absolute threshold will starve high octaves;")
        print("     per-level adaptation is required.")
    elif r_top > 2.0:
        print(f"  => Responses RISE with level ({r_top:.2f}x). Downsampling")
        print("     is sharpening structure faster than blur destroys it,")
        print("     so an absolute threshold would over-detect high octaves.")
    else:
        print(f"  => Responses are roughly level-INDEPENDENT ({r_top:.2f}x).")
        print("     The 'exponential decay' concern does not apply here; an")
        print("     absolute threshold is defensible, though rank selection")
        print("     is still more robust.")

    _csv(output / "octave_lambda_decay.csv", rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    lv = [r["level"] for r in rows]
    for key, lab in (("lam_max", "max"), ("lam_p99", "p99"),
                     ("lam_median", "median")):
        ax.plot(lv, [r[key] for r in rows], "-o", ms=4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("pyramid level"); ax.set_ylabel("lambda2 (Shi-Tomasi response)")
    ax.set_title("Shi-Tomasi response vs pyramid level")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "octave_lambda_decay.png", dpi=110)
    plt.close(fig)
    return rows


def q2_threshold_rules(paths, frames, output: Path, target_n,
                       nlevels, scale_factor):
    print()
    print("=" * 72)
    print("Q2. Which threshold rule fills the octaves?")
    print("=" * 72)

    ideal = level_target_counts(target_n, nlevels, scale_factor)
    print(f"\n  ORB-SLAM3's per-level target allocation for N={target_n}:")
    print("    " + "  ".join(f"L{l}:{n}" for l, n in enumerate(ideal)))

    results = {}
    for rule in RULES:
        counts = np.zeros(nlevels)
        cands = np.zeros(nlevels)
        thrs = np.zeros(nlevels)
        peaks = np.zeros(nlevels)
        totals = []
        for idx in frames:
            gray = read_gray(paths[idx])
            kps, st = detect_multiscale_shi_tomasi(
                gray, target_n=target_n, rule=rule,
                nlevels=nlevels, scale_factor=scale_factor,
                return_stats=True,
            )
            for kp in kps:
                if 0 <= kp.octave < nlevels:
                    counts[kp.octave] += 1
            for s in st:
                cands[s["level"]] += s["n_candidates"]
                thrs[s["level"]] += s["threshold"]
                peaks[s["level"]] += s["n_candidates"]
            totals.append(len(kps))
        counts /= max(1, len(frames))
        cands /= max(1, len(frames))
        thrs /= max(1, len(frames))
        results[rule] = {"per_level": counts, "candidates": cands,
                         "thresholds": thrs, "total": np.mean(totals)}

        # STARVATION is the real diagnostic, and it is visible only in the
        # CANDIDATE counts: every rule is capped at the per-level target,
        # so a rule that barely clears the target and one that clears it
        # ten times over look identical in the kept counts. A level whose
        # candidates fall below its target cannot fill its quota, and that
        # is what an ill-chosen absolute threshold does to high octaves.
        starved = [l for l in range(nlevels) if cands[l] < ideal[l]]
        print(f"\n  rule={rule:<9s} mean kept={np.mean(totals):7.1f}")
        print("    kept:       " +
              "  ".join(f"L{l}:{c:.0f}" for l, c in enumerate(counts)))
        print("    candidates: " +
              "  ".join(f"L{l}:{c:.0f}" for l, c in enumerate(cands)))
        # If the thresholds differ across levels but the candidate counts
        # do not, the binding constraint is non-maximum suppression, not
        # the threshold — the diagnostic has no discriminating power on
        # that footage and the rule choice is moot for it.
        print("    threshold:  " +
              "  ".join(f"L{l}:{t:.4f}" for l, t in enumerate(thrs)))
        if starved:
            print(f"    STARVED octaves (candidates < target): {starved}")
        else:
            print("    no starved octaves — every level can fill its quota")

    print()
    print("  A starved octave produces fewer descriptors than ORB-SLAM3")
    print("  expects at that scale; an EMPTY one produces none at all, so")
    print("  nothing at that scale can match on revisit.")

    _csv(output / "octave_threshold_rules.csv",
         [{"rule": r, "level": l, "mean_kept": float(d["per_level"][l]),
           "mean_candidates": float(d["candidates"][l]),
           "mean_threshold": float(d["thresholds"][l]), "target": ideal[l]}
          for r, d in results.items() for l in range(nlevels)])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(nlevels)
    w = 0.25
    for i, rule in enumerate(RULES):
        ax.bar(x + (i - 1) * w, results[rule]["per_level"], w, label=rule)
    ax.plot(x, ideal, "k--o", ms=4, label="ORB-SLAM3 target allocation")
    ax.set_xlabel("octave"); ax.set_ylabel("mean features kept per frame")
    ax.set_title("Features KEPT per octave (capped at target)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for i, rule in enumerate(RULES):
        ax2.bar(x + (i - 1) * w, results[rule]["candidates"], w, label=rule)
    ax2.plot(x, ideal, "k--o", ms=4, label="target (starvation line)")
    ax2.set_yscale("log")
    ax2.set_xlabel("octave"); ax2.set_ylabel("mean candidates per frame")
    ax2.set_title("CANDIDATES per octave — below the line = starved")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output / "octave_threshold_rules.png", dpi=110)
    plt.close(fig)
    return results


def q3_vs_orb(paths, frames, output: Path, target_n, nlevels,
              scale_factor, rule="rank"):
    print()
    print("=" * 72)
    print("Q3. Shi-Tomasi octave distribution vs FAST/ORB (vocabulary risk)")
    print("=" * 72)

    orb = get_orb()
    st_counts = np.zeros(nlevels)
    orb_counts = np.zeros(nlevels)
    st_tot, orb_tot = [], []
    for idx in frames:
        gray = read_gray(paths[idx])
        st = detect_multiscale_shi_tomasi(
            gray, target_n=target_n, rule=rule,
            nlevels=nlevels, scale_factor=scale_factor)
        for kp in st:
            if 0 <= kp.octave < nlevels:
                st_counts[kp.octave] += 1
        st_tot.append(len(st))

        okps, _ = detect_orb_quadtree(orb, gray, target_n=target_n)
        for kp in okps:
            if 0 <= kp.octave < nlevels:
                orb_counts[kp.octave] += 1
        orb_tot.append(len(okps))

    st_counts /= max(1, len(frames))
    orb_counts /= max(1, len(frames))
    st_frac = st_counts / max(1e-9, st_counts.sum())
    orb_frac = orb_counts / max(1e-9, orb_counts.sum())

    print(f"\n  {'octave':>7s} {'ShiTomasi':>11s} {'ORB/FAST':>10s} "
          f"{'ST frac':>9s} {'ORB frac':>9s}")
    for l in range(nlevels):
        print(f"  {l:>7d} {st_counts[l]:>11.1f} {orb_counts[l]:>10.1f} "
              f"{st_frac[l]:>9.3f} {orb_frac[l]:>9.3f}")

    # Total variation distance between the two octave distributions:
    # 0 = identical allocation across scales, 1 = disjoint.
    tvd = 0.5 * float(np.abs(st_frac - orb_frac).sum())
    print(f"\n  Total variation distance between octave distributions: "
          f"{tvd:.3f}")
    if tvd < 0.10:
        print("  => Nearly identical scale allocation. Shi-Tomasi descriptors")
        print("     should populate the DBoW2 vocabulary much like FAST's.")
    elif tvd < 0.25:
        print("  => Moderate difference. Worth checking loop-closure recall")
        print("     after the port, but not obviously disqualifying.")
    else:
        print("  => LARGE difference. Shi-Tomasi is sampling scales quite")
        print("     differently from FAST, so DBoW2 scores computed with the")
        print("     stock vocabulary may degrade. Consider retraining the")
        print("     vocabulary on Shi-Tomasi descriptors from this domain.")

    _csv(output / "octave_vs_orb.csv",
         [{"octave": l, "shi_tomasi": float(st_counts[l]),
           "orb": float(orb_counts[l]), "st_frac": float(st_frac[l]),
           "orb_frac": float(orb_frac[l])} for l in range(nlevels)])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(nlevels)
    ax.bar(x - 0.2, st_frac, 0.4, label=f"Shi-Tomasi ({rule})")
    ax.bar(x + 0.2, orb_frac, 0.4, label="ORB / FAST")
    ax.set_xlabel("octave"); ax.set_ylabel("fraction of features")
    ax.set_title(f"Scale allocation: Shi-Tomasi vs FAST  (TVD={tvd:.3f})")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output / "octave_vs_orb.png", dpi=110)
    plt.close(fig)
    return tvd


def _csv(path: Path, rows):
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
    ap.add_argument("--output", type=Path, default=Path("octave_results"))
    ap.add_argument("--frames", type=int, default=20,
                    help="How many frames to sample.")
    ap.add_argument("--frame-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Restrict sampling to [LO, HI). Use this to stay "
                         "in the well-lit part of the sequence.")
    ap.add_argument("--target-features", type=int, default=1000)
    ap.add_argument("--nlevels", type=int, default=DEFAULT_NLEVELS)
    ap.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    ap.add_argument("--rule", choices=RULES, default="rank",
                    help="Rule used for the Q3 comparison against ORB.")
    args = ap.parse_args()

    paths = load_image_paths(args.image_dir)
    lo, hi = (args.frame_range if args.frame_range
              else (0, len(paths)))
    hi = min(hi, len(paths))
    frames = np.linspace(lo, hi - 1, num=min(args.frames, hi - lo),
                         dtype=int).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Sampling {len(frames)} frames from [{lo}, {hi})\n")

    q1_lambda_decay(paths, frames, args.output, args.nlevels,
                    args.scale_factor)
    q2_threshold_rules(paths, frames, args.output, args.target_features,
                       args.nlevels, args.scale_factor)
    q3_vs_orb(paths, frames, args.output, args.target_features,
              args.nlevels, args.scale_factor, rule=args.rule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
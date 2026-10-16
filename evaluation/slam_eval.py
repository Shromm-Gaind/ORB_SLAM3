#!/usr/bin/env python3
"""
ORB-SLAM3 vs ground-truth trajectory evaluation.

Extends the original rosbag odometry dump into a full pipeline:

    extract GT  ->  load SLAM (TUM)  ->  associate  ->  align  ->  ATE / RPE  ->  plots

Usage
-----
  # 1. dump ground truth from a bag (needs a sourced ROS 2 env)
  python slam_eval.py extract --bag /path/to/bag --topic /odometry/sim --out gt.csv

  # 2. dump the camera stamps you actually fed to ORB-SLAM3 (coverage denominator)
  python slam_eval.py frames --bag /path/to/bag --topic /camera/left/image_raw \
        --out frame_times.txt

  # 3. evaluate -- every figure as its own file, every table as its own CSV
  python slam_eval.py eval --gt gt.csv --slam CameraTrajectory.txt \
        --frames frame_times.txt --outdir results/klt --label klt \
        --append-to results/all_runs.csv

Notes
-----
* For STEREO / RGB-D ORB-SLAM3 use --align se3 (metric). Sim(3) alignment with a
  free scale hides real scale error; the fitted scale is always reported anyway.
* --align sim3 is the correct choice only for monocular.
* Tracking coverage needs CameraTrajectory.txt, not KeyFrameTrajectory.txt:
  ORB-SLAM3 omits lost frames when saving, so the holes in the per-frame file
  are the dropouts.  Keyframes are too sparse and irregular to tell a dropout
  from normal keyframe spacing.
* --state consumes the per-frame tracking-state log produced by the
  instrumentation patch (see instrument_tracking_state.md) and turns the
  coverage estimate into an exact measurement, including Atlas map IDs.
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
#  1. Ground truth extraction (your original script, with quaternions kept)
# --------------------------------------------------------------------------- #
def extract_gt(bag_path, topic, output_csv, msg_type_str="nav_msgs/msg/Odometry"):
    """Dump poses from a ROS 2 bag to CSV. Quaternion is kept alongside Euler:
    Euler angles are fine for plotting but useless for pose composition."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type = get_message(msg_type_str)

    n = 0
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["timestamp_sec", "x", "y", "z", "qx", "qy", "qz", "qw",
             "roll_deg", "pitch_deg", "yaw_deg"]
        )
        while reader.has_next():
            _topic, data, _t = reader.read_next()
            msg = deserialize_message(data, msg_type)

            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

            # header stamp, not the bag receive time -- the two can differ by
            # tens of ms and that is pure noise in your ATE.
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            w.writerow([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, roll, pitch, yaw])
            n += 1

    print(f"Extracted {n} poses -> {output_csv}")
    return output_csv


def extract_frame_times(bag_path, topic, output_txt,
                        msg_type_str="sensor_msgs/msg/Image"):
    """Dump the header stamps of every camera frame you fed to ORB-SLAM3.
    These are the denominator of the tracking-coverage metric."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type = get_message(msg_type_str)

    stamps = []
    while reader.has_next():
        _topic, data, _t = reader.read_next()
        msg = deserialize_message(data, msg_type)
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    np.savetxt(output_txt, np.sort(stamps), fmt="%.9f")
    print(f"Extracted {len(stamps)} frame timestamps -> {output_txt}")
    return output_txt


# --------------------------------------------------------------------------- #
#  2. Loading
# --------------------------------------------------------------------------- #
def _normalise_time(t):
    """ORB-SLAM3 echoes back whatever timestamp you fed it. Depending on the
    driver that is seconds, milliseconds, microseconds or nanoseconds.
    Guess the unit from the span+magnitude and return seconds."""
    span = float(t[-1] - t[0])
    if span > 1e15:
        return t * 1e-9, "ns"
    if span > 1e12:
        return t * 1e-6, "us"
    if span > 1e9:
        return t * 1e-9, "ns"      # e.g. 81_480_000_000 -> 81.48 s
    if span > 1e6:
        return t * 1e-3, "ms"
    return t, "s"


def load_tum(path):
    """TUM format: timestamp tx ty tz qx qy qz qw  (pose = T_world_body)."""
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 8:
        raise ValueError(f"{path}: expected 8 columns (TUM), got {d.shape[1]}")
    order = np.argsort(d[:, 0])          # KeyFrameTrajectory is not always sorted
    d = d[order]
    t, unit = _normalise_time(d[:, 0])
    print(f"  {path}: {len(d)} poses, timestamps interpreted as [{unit}], "
          f"t = {t[0]:.3f} .. {t[-1]:.3f} s")
    return t, d[:, 1:4], d[:, 4:8]


def load_gt_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")

    t = np.array([float(r["timestamp_sec"]) for r in rows])
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])

    if "qx" in rows[0]:
        quat = np.array(
            [[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
             for r in rows]
        )
    else:  # older CSVs that only stored Euler
        eul = np.array(
            [[float(r["roll_deg"]), float(r["pitch_deg"]), float(r["yaw_deg"])]
             for r in rows]
        )
        quat = R.from_euler("xyz", eul, degrees=True).as_quat()

    order = np.argsort(t)
    t, xyz, quat = t[order], xyz[order], quat[order]
    t, unit = _normalise_time(t)
    print(f"  {path}: {len(t)} poses, timestamps [{unit}], "
          f"t = {t[0]:.3f} .. {t[-1]:.3f} s")
    return t, xyz, quat


# --------------------------------------------------------------------------- #
#  3. Association
# --------------------------------------------------------------------------- #
def associate(t_slam, t_gt, max_dt=0.02, offset=0.0):
    """Nearest-neighbour match of each SLAM stamp to a GT stamp."""
    ts = t_slam + offset
    idx = np.searchsorted(t_gt, ts)
    idx = np.clip(idx, 1, len(t_gt) - 1)
    left, right = t_gt[idx - 1], t_gt[idx]
    pick_left = (ts - left) < (right - ts)
    j = np.where(pick_left, idx - 1, idx)
    dt = np.abs(t_gt[j] - ts)
    keep = dt <= max_dt
    return np.nonzero(keep)[0], j[keep], dt[keep]


def estimate_time_offset(t_slam, p_slam, t_gt, p_gt, search=1.0, step=0.005):
    """Brute-force the constant clock offset that best lines the two speed
    profiles up. Cheap insurance against a systematically shifted stamp."""
    def speed(t, p):
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.diff(t)
        ok = dt > 1e-9
        return t[1:][ok], d[ok] / dt[ok]

    ts, vs = speed(t_slam, p_slam)
    tg, vg = speed(t_gt, p_gt)
    if len(ts) < 5 or len(tg) < 5:
        return 0.0
    vs = (vs - vs.mean()) / (vs.std() + 1e-12)
    grid = np.arange(ts[0], ts[-1], 0.05)
    vs_i = np.interp(grid, ts, vs)

    best, best_score = 0.0, -np.inf
    for off in np.arange(-search, search + step, step):
        vg_i = np.interp(grid + off, tg, vg)
        vg_i = (vg_i - vg_i.mean()) / (vg_i.std() + 1e-12)
        score = float(np.dot(vs_i, vg_i))
        if score > best_score:
            best, best_score = float(off), score
    return best


# --------------------------------------------------------------------------- #
#  4. Alignment
# --------------------------------------------------------------------------- #
def umeyama(src, dst, with_scale=True):
    """Least-squares Sim(3)/SE(3) fit mapping src -> dst.
    Returns (R, t, s) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    C = Xd.T @ Xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * Rm @ mu_s
    return Rm, t, s


def average_rotation(mats):
    """Chordal L2 mean of a stack of rotation matrices (projection of the sum)."""
    U, _, Vt = np.linalg.svd(mats.sum(axis=0))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def estimate_body_extrinsic(R_align, q_slam, q_gt):
    """SLAM reports the *camera* frame, GT reports base_link. They differ by a
    constant right-hand rotation R_bc that Umeyama (positions only) cannot see.
    R_gt_i ~= R_align @ R_slam_i @ R_cb  ->  R_cb = mean(R_slam_i^T R_align^T R_gt_i)"""
    Rs = R.from_quat(q_slam).as_matrix()
    Rg = R.from_quat(q_gt).as_matrix()
    return average_rotation(np.einsum("nji,jk,nkl->nil", Rs, R_align.T, Rg))


# --------------------------------------------------------------------------- #
#  5. Metrics
# --------------------------------------------------------------------------- #
def ate(p_est, p_gt):
    e = np.linalg.norm(p_est - p_gt, axis=1)
    return {
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "std": float(e.std()),
        "min": float(e.min()),
        "max": float(e.max()),
        "errors": e,
    }


def _se3(Rm, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T


def rpe(t, p_est, q_est, p_gt, q_gt, delta=1.0, per_second=True):
    """Relative pose error over a fixed time delta.

    Using *consecutive keyframes* instead of a fixed delta is the classic way to
    get a meaningless rotation RPE: ORB-SLAM3 keyframe spacing swings between
    ~0.1 s and several seconds, so you end up averaging errors accumulated over
    wildly different intervals.
    """
    Te = [_se3(m, p) for m, p in zip(R.from_quat(q_est).as_matrix(), p_est)]
    Tg = [_se3(m, p) for m, p in zip(R.from_quat(q_gt).as_matrix(), p_gt)]

    trans, rot, times = [], [], []
    j_start = 0
    for i in range(len(t)):
        j = max(j_start, i + 1)
        while j < len(t) and t[j] - t[i] < delta:
            j += 1
        if j >= len(t):
            break
        dt = t[j] - t[i]
        if dt > 2.0 * delta:      # gap in the data, skip rather than fabricate
            continue
        E = np.linalg.inv(np.linalg.inv(Tg[i]) @ Tg[j]) @ (
            np.linalg.inv(Te[i]) @ Te[j]
        )
        dtrans = np.linalg.norm(E[:3, 3])
        drot = np.degrees(np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec()))
        if per_second:
            dtrans, drot = dtrans / dt, drot / dt
        trans.append(dtrans)
        rot.append(drot)
        times.append(t[i])

    trans, rot, times = map(np.asarray, (trans, rot, times))
    if len(trans) == 0:
        return None
    return {
        "trans_rmse": float(np.sqrt((trans ** 2).mean())),
        "rot_rmse": float(np.sqrt((rot ** 2).mean())),
        "trans_median": float(np.median(trans)),
        "rot_median": float(np.median(rot)),
        "trans": trans,
        "rot": rot,
        "t": times,
        "n": len(trans),
    }


# --------------------------------------------------------------------------- #
#  5b. Tracking coverage / Atlas continuity
# --------------------------------------------------------------------------- #
#  ORB-SLAM3 drops a frame from CameraTrajectory.txt when mlbLost[i] is true,
#  and drops frames whose reference keyframe is not in the surviving map.  So
#  a hole in the saved trajectory IS a tracking dropout -- the file does not
#  tell you about it, but the missing timestamps do.  After an Atlas merge the
#  two sides of the hole are welded into one frame, which is why the plot looks
#  continuous.  Everything below measures the holes and the quality of the weld.
# --------------------------------------------------------------------------- #
def load_frame_times(path):
    """One timestamp per line (or first column of a CSV), seconds/ns/us/ms."""
    t = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None,
                   ndmin=2, usecols=0, skiprows=0)
    t = np.sort(np.asarray(t).ravel())
    t, unit = _normalise_time(t)
    print(f"  {path}: {len(t)} camera frames, timestamps [{unit}], "
          f"{1.0/np.median(np.diff(t)):.1f} Hz nominal")
    return t


STATE_NAMES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED",
               2: "OK", 3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}


def load_state_log(path):
    """CSV written by the instrumentation patch: timestamp,state[,map_id].
    state follows ORB_SLAM3::Tracking::eTrackingState (2 == OK)."""
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    t = np.asarray(rows["timestamp"], dtype=float)
    st = np.asarray(rows["state"], dtype=int)
    mid = (np.asarray(rows["map_id"], dtype=int)
           if "map_id" in (rows.dtype.names or ()) else None)
    order = np.argsort(t)
    t, unit = _normalise_time(t[order])
    print(f"  {path}: {len(t)} state samples, timestamps [{unit}]")
    return t, st[order], (mid[order] if mid is not None else None)


def summarise_states(t, st, mid):
    """Exact tracking statistics straight from the tracker."""
    out = {"counts": {}, "ok_frac": float(np.mean(np.isin(st, (2, 5)))),
           "lost_intervals": [], "maps": None, "map_switches": []}
    for s in np.unique(st):
        out["counts"][STATE_NAMES.get(int(s), str(s))] = int((st == s).sum())
    ok = np.isin(st, (2, 5))
    edges = np.diff(np.concatenate(([False], ~ok, [False])).astype(int))
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        out["lost_intervals"].append(
            (float(t[a]), float(t[min(b, len(t) - 1)]),
             STATE_NAMES.get(int(st[a]), str(st[a]))))
    if mid is not None:
        out["maps"] = sorted(set(int(m) for m in mid))
        ch = np.nonzero(np.diff(mid) != 0)[0]
        out["map_switches"] = [(float(t[k + 1]), int(mid[k]), int(mid[k + 1]))
                               for k in ch]
    return out


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def gt_distance_between(t_gt, p_gt, t0, t1):
    m = (t_gt >= t0) & (t_gt <= t1)
    return path_length(p_gt[m]) if m.sum() > 1 else 0.0


def find_segments(t, gap_thresh):
    """Split a timestamp list into continuously-tracked runs."""
    if len(t) == 0:
        return []
    breaks = np.nonzero(np.diff(t) > gap_thresh)[0]
    return [s for s in np.split(np.arange(len(t)), breaks + 1) if len(s) > 0]


def tracking_coverage(t_sl, t_gt, p_gt, t_frames=None, gap_thresh=None,
                      frame_tol=None):
    """Separate POSE RATE from TRACKING COVERAGE -- they are not the same thing.

    A trajectory file can be sparse for two completely different reasons:

      (a) the tracker lost the camera and ORB-SLAM3 skipped those frames on
          save (mlbLost), or the frames were never fed to the tracker at all;
      (b) the build simply does not emit one pose per camera frame.

    Only (a) is a tracking failure.  Treating every missing frame as a dropout
    turns a healthy 8 Hz output from a 30 Hz camera into "1193 dropouts, 4%
    tracked", which is nonsense.  So a dropout is a gap long enough to be a
    real loss of localisation -- by default max(0.5 s, 5x the median pose
    interval) -- and the per-frame density is reported separately as a rate.
    """
    dt = np.diff(t_sl)
    dt_med = float(np.median(dt)) if len(dt) else 0.0
    pose_hz = 1.0 / dt_med if dt_med > 0 else float("nan")

    frame_dt = frame_hz = None
    if t_frames is not None and len(t_frames) > 1:
        frame_dt = float(np.median(np.diff(t_frames)))
        frame_hz = 1.0 / frame_dt

    auto_gap = gap_thresh is None
    if auto_gap:
        gap_thresh = max(0.5, 5.0 * dt_med)

    if t_frames is not None:
        tol = frame_tol if frame_tol is not None else 0.5 * frame_dt
        idx = np.clip(np.searchsorted(t_sl, t_frames), 1, max(len(t_sl) - 1, 1))
        near = np.minimum(
            np.abs(t_frames - t_sl[np.clip(idx - 1, 0, len(t_sl) - 1)]),
            np.abs(t_frames - t_sl[np.clip(idx, 0, len(t_sl) - 1)]))
        tracked = near <= tol
        t0_all, t1_all = float(t_frames[0]), float(t_frames[-1])
    else:
        tracked = None
        t0_all = float(min(t_sl[0], t_gt[0]))
        t1_all = float(max(t_sl[-1], t_gt[-1]))

    segments = find_segments(t_sl, gap_thresh)

    drops = []
    if len(segments) > 1:
        for a, b in zip(segments[:-1], segments[1:]):
            drops.append((float(t_sl[a[-1]]), float(t_sl[b[0]])))
    if t_sl[0] - t0_all > gap_thresh:
        drops.insert(0, (t0_all, float(t_sl[0])))        # never initialised
    if t1_all - t_sl[-1] > gap_thresh:
        drops.append((float(t_sl[-1]), t1_all))           # never recovered

    total_T = t1_all - t0_all
    lost_T = sum(b - a for a, b in drops)
    gt_total_d = path_length(p_gt)
    dropouts = [{"t0": a, "t1": b, "dur": b - a,
                 "gt_dist": gt_distance_between(t_gt, p_gt, a, b)}
                for a, b in drops]
    lost_d = sum(d["gt_dist"] for d in dropouts)
    seg_T = [float(t_sl[s[-1]] - t_sl[s[0]]) for s in segments]

    # every gap, dropout or not -- lets you see the shape of the distribution
    edges = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
    hist = [int(((dt >= lo) & (dt < hi)).sum()) for lo, hi in zip(edges, edges[1:])]
    biggest = dropouts and max(dropouts, key=lambda d: d["dur"]) or None

    return {
        "segments": segments,
        "dropouts": dropouts,
        "gap_thresh": gap_thresh,
        "auto_gap": auto_gap,
        "pose_hz": pose_hz,
        "dt_med": dt_med,
        "frame_hz": frame_hz,
        "frames_per_pose": (dt_med / frame_dt) if frame_dt else None,
        "gap_hist": list(zip(edges[:-1], edges[1:], hist)),
        "gaps": dt,
        "biggest": biggest,
        "frac_time": 1.0 - lost_T / total_T if total_T > 0 else float("nan"),
        "frac_dist": 1.0 - lost_d / gt_total_d if gt_total_d > 0 else float("nan"),
        "frac_frames": float(tracked.mean()) if tracked is not None else None,
        "n_frames": None if tracked is None else int(len(tracked)),
        "n_tracked_frames": None if tracked is None else int(tracked.sum()),
        "longest_seg": max(seg_T) if seg_T else 0.0,
        "longest_frac": (max(seg_T) / total_T) if seg_T and total_T > 0 else 0.0,
        "total_time": total_T,
        "lost_time": lost_T,
        "lost_dist": lost_d,
        "gt_dist": gt_total_d,
        "n_poses": len(t_sl),
    }


def seam_errors(segments, t_rel, p_al, q_al, p_gt, q_gt):
    """How well did the Atlas merge weld the two sides of each dropout?

    One relative-pose error measured across the hole.  Small values mean the
    merge recovered the true motion; large values mean the stitched trajectory
    is globally wrong even though it looks continuous.
    """
    out = []
    Rg = R.from_quat(q_gt).as_matrix()
    Re = R.from_quat(q_al).as_matrix()
    for a, b in zip(segments[:-1], segments[1:]):
        i, j = a[-1], b[0]
        d_est, d_gt = p_al[j] - p_al[i], p_gt[j] - p_gt[i]
        dR = (Re[j] @ Re[i].T) @ (Rg[j] @ Rg[i].T).T
        out.append({
            "t0": float(t_rel[i]), "t1": float(t_rel[j]),
            "trans": float(np.linalg.norm(d_est - d_gt)),
            "rot": float(np.degrees(np.linalg.norm(R.from_matrix(dR).as_rotvec()))),
            "gt_move": float(np.linalg.norm(d_gt)),
        })
    return out


def segment_errors(segments, p_sl, p_gt, align="se3", min_poses=10):
    """ATE of each tracked segment aligned on its own.

    A single global Umeyama over a merged trajectory flatters a bad merge: the
    fit splits the difference between well-tracked sections.  Per-segment ATE
    tells you how good tracking was *while it was tracking*, independent of how
    the pieces were stitched.  Global >> per-segment == the merge is the problem.
    """
    res = []
    for k, s in enumerate(segments):
        if len(s) < min_poses:
            res.append({"id": k, "n": len(s), "rmse": None, "scale": None})
            continue
        Rm, tv, sc = umeyama(p_sl[s], p_gt[s], with_scale=(align == "sim3"))
        pa = (sc * (Rm @ p_sl[s].T)).T + tv
        _, _, sfree = umeyama(p_sl[s], p_gt[s], with_scale=True)
        e = np.linalg.norm(pa - p_gt[s], axis=1)
        res.append({"id": k, "n": len(s),
                    "rmse": float(np.sqrt((e ** 2).mean())), "scale": float(sfree)})
    return res


# --------------------------------------------------------------------------- #
#  6. Figures
# --------------------------------------------------------------------------- #
#  Each panel is a function that draws onto an axis it is given, so the same
#  code produces both the combined dashboard and one file per figure.
#  Individual figures carry no title by default (captions belong in the
#  document); pass --titles to add them back.
# --------------------------------------------------------------------------- #
GT_C, SL_C = "green", "blue"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    return plt


def _shade(ax, ctx):
    """Grey spans over genuine dropouts on any time-axis plot."""
    for d in ctx["cov"]["dropouts"]:
        ax.axvspan(d["t0"] - ctx["t_abs0"], d["t1"] - ctx["t_abs0"],
                   color="grey", alpha=.28, lw=0, zorder=0)


def _plot_broken(ax, ctx, cols, **kw):
    """SLAM track with real gaps, never a fake straight line across a dropout."""
    data = ctx["p_al"]
    for k, s in enumerate(ctx["segs"]):
        style = dict(kw, marker=("." if len(s) == 1 else None))
        if k:
            style["label"] = None
        ax.plot(*[data[s, c] for c in cols], **style)


def _relocalisation_marks(ax, ctx, cols):
    segs = ctx["segs"]
    if len(segs) < 2:
        return
    p = ctx["p_al"]
    for s in segs[1:]:
        ax.plot(*[p[s[0], c] for c in cols], "o", ms=7, mfc="none",
                mec="orange", mew=2, zorder=5)
    ax.plot([], [], "o", ms=7, mfc="none", mec="orange", mew=2,
            label="re-localised / merged")


def panel_xy(ax, ctx, title):
    ax.plot(ctx["p_gt"][:, 0], ctx["p_gt"][:, 1], GT_C, lw=2, label="Ground truth")
    _plot_broken(ax, ctx, (0, 1), color=SL_C, lw=1.2, label="ORB-SLAM3 (aligned)")
    _relocalisation_marks(ax, ctx, (0, 1))
    ax.set(xlabel="X (m)", ylabel="Y (m)")
    if title:
        ax.set_title("Top-down trajectory (XY)")
    ax.legend(); ax.grid(alpha=.3); ax.axis("equal")


def panel_xz(ax, ctx, title):
    ax.plot(ctx["p_gt"][:, 0], ctx["p_gt"][:, 2], GT_C, lw=2, label="Ground truth")
    _plot_broken(ax, ctx, (0, 2), color=SL_C, lw=1.2, label="ORB-SLAM3 (aligned)")
    ax.set(xlabel="X (m)", ylabel="Z (m)")
    if title:
        ax.set_title("Side view (XZ)")
    ax.legend(); ax.grid(alpha=.3)


def panel_3d(ax, ctx, title):
    g = ctx["p_gt"]
    ax.plot(g[:, 0], g[:, 1], g[:, 2], GT_C, lw=2, label="Ground truth")
    _plot_broken(ax, ctx, (0, 1, 2), color=SL_C, lw=1.2, label="ORB-SLAM3")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    if title:
        ax.set_title("3D trajectory")
    ax.legend()


def panel_ape(ax, ctx, title):
    t, e = ctx["t_rel"], ctx["ate"]["errors"]
    ax.fill_between(t, e, color="red", alpha=.35)
    ax.plot(t, e, "r", lw=1, label="APE")
    ax.axhline(ctx["ate"]["rmse"], color="k", ls="--", lw=.8,
               label=f"RMSE {ctx['ate']['rmse']:.3f} m")
    _shade(ax, ctx)
    ax.set(xlabel="Time (s)", ylabel="Absolute position error (m)")
    if title:
        ax.set_title("Absolute pose error")
    ax.legend(fontsize=8); ax.grid(alpha=.3)


def panel_components(ax, ctx, title):
    t, g, p = ctx["t_rel"], ctx["p_gt"], ctx["p_al"]
    for k, (lbl, ls) in enumerate(zip("XYZ", ["-", "--", ":"])):
        ax.plot(t, g[:, k], GT_C, ls=ls, lw=2, label=f"GT {lbl}")
        ax.plot(t, p[:, k], SL_C, ls=ls, lw=1, label=f"SLAM {lbl}")
    _shade(ax, ctx)
    ax.set(xlabel="Time (s)", ylabel="Position (m)")
    if title:
        ax.set_title("Position components")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)


def panel_rpe(ax, ctx, title):
    r, u = ctx["rpe"], "/s" if ctx["per_second"] else ""
    if r:
        ax.plot(r["t"], r["trans"], "r", lw=.9, label="Translation")
        ax.set_ylabel(f"Translational RPE (m{u})", color="r")
        ax2 = ax.twinx()
        ax2.plot(r["t"], r["rot"], "b", lw=.9, alpha=.7, label="Rotation")
        ax2.set_ylabel(f"Rotational RPE (deg{u})", color="b")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)
    _shade(ax, ctx)
    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(f"Relative pose error (Δ = {ctx['delta']:.1f} s)")
    ax.grid(alpha=.3)


def panel_gaps(ax, ctx, title):
    gaps = ctx["cov"]["gaps"]
    gaps = gaps[gaps > 0]
    thr = ctx["cov"]["gap_thresh"]
    if len(gaps):
        lo, hi = max(gaps.min(), 1e-3), max(gaps.max(), thr * 1.5)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        ax.hist(gaps[gaps <= thr], bins=bins, color="grey", label="normal spacing")
        ax.hist(gaps[gaps > thr], bins=bins, color="red", label="dropout")
        ax.axvline(thr, color="k", ls="--", lw=1,
                   label=f"dropout threshold {thr:.2f} s")
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel="Gap between consecutive poses (s)", ylabel="Count")
    if title:
        ax.set_title("Pose gap distribution")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")


def panel_segments(ax, ctx, title):
    rows = [r for r in ctx["seg_res"] if r["rmse"] is not None]
    ax.bar([str(r["id"]) for r in rows], [r["rmse"] for r in rows], color=SL_C,
           alpha=.7, label="segment ATE (own alignment)")
    ax.axhline(ctx["ate"]["rmse"], color="r", ls="--", lw=1,
               label=f"global ATE {ctx['ate']['rmse']:.3f} m")
    ax.set(xlabel="Tracked segment", ylabel="ATE RMSE (m)")
    if title:
        ax.set_title("Per-segment vs global ATE")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")


# name, function, 3d?, size (in), included in the dashboard?
PANELS = [
    ("trajectory_xy",        panel_xy,         False, (7, 6),   True),
    ("trajectory_xz",        panel_xz,         False, (7, 5),   True),
    ("trajectory_3d",        panel_3d,         True,  (7, 6),   True),
    ("ape_over_time",        panel_ape,        False, (8, 4.5), True),
    ("position_components",  panel_components, False, (8, 4.5), True),
    ("rpe_over_time",        panel_rpe,        False, (8, 4.5), True),
    ("pose_gap_histogram",   panel_gaps,       False, (7, 4.5), False),
    ("segment_ate",          panel_segments,   False, (7, 4.5), False),
]


def _panel_applies(name, ctx):
    if name == "segment_ate":
        return sum(r["rmse"] is not None for r in ctx["seg_res"]) >= 2
    if name == "rpe_over_time":
        return ctx["rpe"] is not None
    return True


def make_dashboard(ctx, out_png):
    plt = _mpl()
    fig = plt.figure(figsize=(18, 20))
    cov = ctx["cov"]
    head = (f"ORB-SLAM3 vs ground truth   |   covered {cov['frac_time']*100:.1f}% "
            f"of time, {cov['frac_dist']*100:.1f}% of distance   |   "
            f"{len(cov['dropouts'])} dropout(s)")
    if cov["frac_frames"] is not None:
        head += f"   |   {cov['frac_frames']*100:.1f}% of frames have a pose"
    fig.suptitle(head, fontsize=15, fontweight="bold")
    for k, (name, fn, is3d, _sz, dash) in enumerate(p for p in PANELS if p[4]):
        ax = fig.add_subplot(3, 2, k + 1, **({"projection": "3d"} if is3d else {}))
        fn(ax, ctx, True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"Dashboard -> {out_png}")


def export_figures(ctx, outdir, titles=False, fmt="png", dpi=200):
    plt = _mpl()
    written = []
    for name, fn, is3d, size, _dash in PANELS:
        if not _panel_applies(name, ctx):
            continue
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, **({"projection": "3d"} if is3d else {}))
        fn(ax, ctx, titles)
        fig.tight_layout()
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
#  7. Tables
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else float(f"{v:.6g}")
    return v


def write_table(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    return path


def append_results_row(path, row):
    """One row per run, columns unioned across runs -- for KLT vs stock etc."""
    existing, fields = [], []
    if os.path.exists(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            existing = list(rd)
            fields = list(rd.fieldnames or [])
    for k in row:
        if k not in fields:
            fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing:
            w.writerow(r)
        w.writerow({k: _fmt(v) for k, v in row.items()})
    return path


def export_tables(ctx, M, outdir):
    """Every table the run produces, one CSV each."""
    t0 = ctx["t_abs0"]
    cov, out = ctx["cov"], []

    out.append(write_table(os.path.join(outdir, "metrics_summary.csv"),
                           ["section", "metric", "value", "unit"], M))

    out.append(write_table(
        os.path.join(outdir, "per_pose_errors.csv"),
        ["t_s", "gt_x_m", "gt_y_m", "gt_z_m", "slam_x_m", "slam_y_m", "slam_z_m",
         "ape_trans_m", "ape_rot_deg"],
        [[ctx["t_rel"][k], *ctx["p_gt"][k], *ctx["p_al"][k],
          ctx["ate"]["errors"][k], ctx["ape_rot"][k]]
         for k in range(len(ctx["t_rel"]))]))

    if ctx["rpe"]:
        r = ctx["rpe"]
        u = "_per_s" if ctx["per_second"] else ""
        out.append(write_table(
            os.path.join(outdir, "rpe_series.csv"),
            ["t_s", f"rpe_trans_m{u}", f"rpe_rot_deg{u}"],
            zip(r["t"], r["trans"], r["rot"])))

    out.append(write_table(
        os.path.join(outdir, "dropouts.csv"),
        ["dropout_id", "t_start_s", "t_end_s", "duration_s", "gt_distance_m"],
        [[k, d["t0"] - t0, d["t1"] - t0, d["dur"], d["gt_dist"]]
         for k, d in enumerate(cov["dropouts"])]))

    gaps, thr = cov["gaps"], cov["gap_thresh"]
    out.append(write_table(
        os.path.join(outdir, "pose_gap_histogram.csv"),
        ["bin_lo_s", "bin_hi_s", "n_gaps", "n_dropouts"],
        [[lo, hi, n, int(((gaps >= lo) & (gaps < hi) & (gaps > thr)).sum())]
         for lo, hi, n in cov["gap_hist"]]))

    tr = ctx["t_rel"]
    out.append(write_table(
        os.path.join(outdir, "segments.csv"),
        ["segment_id", "n_poses", "t_start_s", "t_end_s", "duration_s",
         "ate_rmse_m", "best_fit_scale"],
        [[r["id"], r["n"], tr[s[0]], tr[s[-1]], tr[s[-1]] - tr[s[0]],
          r["rmse"] if r["rmse"] is not None else "",
          r["scale"] if r["scale"] is not None else ""]
         for r, s in zip(ctx["seg_res"], ctx["segs"])]))

    if ctx["seams"]:
        out.append(write_table(
            os.path.join(outdir, "merge_seams.csv"),
            ["seam_id", "t_start_s", "t_end_s", "trans_error_m", "rot_error_deg",
             "gt_displacement_m"],
            [[k, sm["t0"], sm["t1"], sm["trans"], sm["rot"], sm["gt_move"]]
             for k, sm in enumerate(ctx["seams"])]))

    ss = ctx.get("states")
    if ss:
        n = sum(ss["counts"].values())
        out.append(write_table(
            os.path.join(outdir, "tracker_state_counts.csv"),
            ["state", "frames", "percent"],
            [[k, v, 100.0 * v / n] for k, v in ss["counts"].items()]))
        out.append(write_table(
            os.path.join(outdir, "tracker_lost_intervals.csv"),
            ["interval_id", "t_start_s", "t_end_s", "duration_s", "entered_as"],
            [[k, a - t0, b - t0, b - a, nm]
             for k, (a, b, nm) in enumerate(ss["lost_intervals"])]))
        if ss["maps"] is not None:
            out.append(write_table(
                os.path.join(outdir, "atlas_map_switches.csv"),
                ["switch_id", "t_s", "from_map", "to_map"],
                [[k, tt - t0, m0, m1]
                 for k, (tt, m0, m1) in enumerate(ss["map_switches"])]))
    return out


# --------------------------------------------------------------------------- #
#  8. Driver
# --------------------------------------------------------------------------- #
def evaluate(gt_csv, slam_txt, align="se3", max_dt=0.02, rpe_delta=1.0,
             per_second=True, auto_offset=False, fix_extrinsic=True,
             out_png=None, out_csv=None, frames_file=None, gap_thresh=None,
             state_file=None, outdir=None, label=None, append_to=None,
             titles=False, fig_format="png", dpi=200):
    print("=== ORB-SLAM3 Trajectory Evaluation ===")
    t_gt, p_gt_all, q_gt_all = load_gt_csv(gt_csv)
    t_sl, p_sl_all, q_sl_all = load_tum(slam_txt)
    t_frames = load_frame_times(frames_file) if frames_file else None
    states_raw = load_state_log(state_file) if state_file else None

    M = []  # (section, metric, value, unit) -- becomes metrics_summary.csv

    def m(section, metric, value, unit=""):
        M.append((section, metric, value, unit))

    m("run", "label", label or os.path.splitext(os.path.basename(slam_txt))[0])
    m("run", "slam_file", os.path.abspath(slam_txt))
    m("run", "gt_file", os.path.abspath(gt_csv))

    offset = 0.0
    overlap = min(t_sl[-1], t_gt[-1]) - max(t_sl[0], t_gt[0])
    if overlap <= 0:
        coarse = t_gt[0] - t_sl[0]
        print(f"  ! time ranges do not overlap (SLAM is probably on a clock "
              f"reset to 0). Applying coarse offset {coarse:+.3f} s.")
        offset += coarse
    if auto_offset:
        fine = estimate_time_offset(t_sl + offset, p_sl_all, t_gt, p_gt_all)
        offset += fine
        print(f"  fine clock offset: {fine:+.3f} s  (total {offset:+.3f} s)")

    i, j, dt = associate(t_sl, t_gt, max_dt=max_dt, offset=offset)
    if len(i) < 10:
        raise SystemExit(
            f"Only {len(i)} associations with max_dt={max_dt}s. "
            "Check timestamp units/offset (try --auto-offset or a larger --max-dt)."
        )
    print(f"\nSLAM poses: {len(t_sl)}   GT poses: {len(t_gt)}   "
          f"Associated: {len(i)}  (median |dt| = {np.median(dt)*1000:.1f} ms)")
    m("association", "slam_poses", len(t_sl))
    m("association", "gt_poses", len(t_gt))
    m("association", "associated_pairs", len(i))
    m("association", "median_time_diff", np.median(dt) * 1000, "ms")
    m("association", "max_time_diff_allowed", max_dt * 1000, "ms")
    m("association", "clock_offset", offset, "s")

    p_sl, q_sl = p_sl_all[i], q_sl_all[i]
    p_gt, q_gt = p_gt_all[j], q_gt_all[j]
    t_abs0 = float(t_gt[j][0])
    t_rel = t_gt[j] - t_abs0

    Rm, tv, s = umeyama(p_sl, p_gt, with_scale=(align == "sim3"))
    p_al = (s * (Rm @ p_sl.T)).T + tv

    q_al = (R.from_matrix(Rm) * R.from_quat(q_sl)).as_quat()
    rpy = np.zeros(3)
    if fix_extrinsic:
        R_cb = estimate_body_extrinsic(Rm, q_sl, q_gt)
        q_al = (R.from_quat(q_al) * R.from_matrix(R_cb)).as_quat()
        rpy = R.from_matrix(R_cb).as_euler("xyz", degrees=True)
        print(f"Body extrinsic (camera->base, rpy deg): "
              f"[{rpy[0]:7.2f} {rpy[1]:7.2f} {rpy[2]:7.2f}]")

    _, _, s_free = umeyama(p_sl, p_gt, with_scale=True)
    print(f"Alignment: {align.upper()}   applied scale: {s:.4f}   "
          f"best-fit scale: {s_free:.4f}  ({(s_free-1)*100:+.2f} %)")
    m("alignment", "mode", align.upper())
    m("alignment", "applied_scale", s)
    m("alignment", "best_fit_scale", s_free)
    m("alignment", "scale_error", (s_free - 1) * 100, "%")
    for ax_name, v in zip(("roll", "pitch", "yaw"), rpy):
        m("alignment", f"extrinsic_{ax_name}", v, "deg")

    a = ate(p_al, p_gt)
    ape_rot = np.degrees(np.linalg.norm(
        (R.from_quat(q_gt).inv() * R.from_quat(q_al)).as_rotvec(), axis=1))
    print("\nAbsolute Trajectory Error (ATE)")
    for k in ("rmse", "mean", "median", "std", "min", "max"):
        print(f"  {k.capitalize():7s} {a[k]:.4f} m")
        m("ate", f"trans_{k}", a[k], "m")
    m("ate", "rot_rmse", np.sqrt((ape_rot ** 2).mean()), "deg")
    m("ate", "rot_median", np.median(ape_rot), "deg")

    r = rpe(t_rel, p_al, q_al, p_gt, q_gt, delta=rpe_delta, per_second=per_second)
    u = "/s" if per_second else ""
    m("rpe", "delta", rpe_delta, "s")
    if r:
        print(f"\nRelative Pose Error (Δ = {rpe_delta:.1f} s, n = {r['n']})")
        print(f"  Translation RMSE:   {r['trans_rmse']:.4f} m{u}"
              f"   (median {r['trans_median']:.4f})")
        print(f"  Rotation RMSE:      {r['rot_rmse']:.4f} deg{u}"
              f" (median {r['rot_median']:.4f})")
        m("rpe", "pairs", r["n"])
        m("rpe", "trans_rmse", r["trans_rmse"], f"m{u}")
        m("rpe", "trans_median", r["trans_median"], f"m{u}")
        m("rpe", "rot_rmse", r["rot_rmse"], f"deg{u}")
        m("rpe", "rot_median", r["rot_median"], f"deg{u}")

    # ---- output rate and tracking coverage ---------------------------------
    cov = tracking_coverage(t_sl + offset, t_gt, p_gt_all,
                            t_frames=t_frames, gap_thresh=gap_thresh)

    print("\nPose output rate")
    print(f"  {cov['n_poses']} poses over {cov['total_time']:.1f} s "
          f"= {cov['pose_hz']:.1f} Hz (median spacing {cov['dt_med']*1000:.0f} ms)")
    m("output", "pose_rate", cov["pose_hz"], "Hz")
    m("output", "median_pose_spacing", cov["dt_med"] * 1000, "ms")
    if cov["frame_hz"]:
        print(f"  camera {cov['frame_hz']:.1f} Hz -> "
              f"1 pose per {cov['frames_per_pose']:.1f} camera frames; "
              f"{cov['frac_frames']*100:.1f} % of frames have a pose "
              f"({cov['n_tracked_frames']}/{cov['n_frames']})")
        m("output", "camera_rate", cov["frame_hz"], "Hz")
        m("output", "camera_frames", cov["n_frames"])
        m("output", "frames_with_pose", cov["n_tracked_frames"])
        m("output", "frames_with_pose_pct", cov["frac_frames"] * 100, "%")
        m("output", "frames_per_pose", cov["frames_per_pose"])
        if cov["frames_per_pose"] > 1.5:
            print("  ! the file is NOT one pose per frame. That is an output-rate\n"
                  "    property, not a tracking failure -- do not report it as\n"
                  "    coverage. Either frames are being dropped before the tracker\n"
                  "    (ROS subscriber queue, node too slow for real time) or this\n"
                  "    build does not append a pose for every tracked frame.\n"
                  "    --state tells the two apart.")

    print(f"\nTracking coverage (dropout = gap > {cov['gap_thresh']:.2f} s"
          f"{', auto' if cov['auto_gap'] else ''})")
    print(f"  Time covered:        {cov['frac_time']*100:6.2f} %  "
          f"({cov['total_time']-cov['lost_time']:.1f} / {cov['total_time']:.1f} s)")
    print(f"  GT distance covered: {cov['frac_dist']*100:6.2f} %  "
          f"({cov['gt_dist']-cov['lost_dist']:.1f} / {cov['gt_dist']:.1f} m)")
    print(f"  Continuous segments: {len(cov['segments'])}   "
          f"longest = {cov['longest_seg']:.1f} s "
          f"({cov['longest_frac']*100:.1f} % of run)")
    print("  Gap distribution:    " + "  ".join(
        f"{lo:g}-{'inf' if hi == np.inf else f'{hi:g}'}s:{n}"
        for lo, hi, n in cov["gap_hist"] if n))
    m("coverage", "dropout_threshold", cov["gap_thresh"], "s")
    m("coverage", "time_covered_pct", cov["frac_time"] * 100, "%")
    m("coverage", "distance_covered_pct", cov["frac_dist"] * 100, "%")
    m("coverage", "time_covered", cov["total_time"] - cov["lost_time"], "s")
    m("coverage", "time_total", cov["total_time"], "s")
    m("coverage", "distance_covered", cov["gt_dist"] - cov["lost_dist"], "m")
    m("coverage", "distance_total", cov["gt_dist"], "m")
    m("coverage", "dropouts", len(cov["dropouts"]))
    m("coverage", "segments", len(cov["segments"]))
    m("coverage", "longest_segment", cov["longest_seg"], "s")
    m("coverage", "longest_segment_pct", cov["longest_frac"] * 100, "%")
    if cov["dropouts"]:
        b = cov["biggest"]
        print(f"  Largest dropout:     {b['dur']:.2f} s / {b['gt_dist']:.2f} m "
              f"at t = {b['t0']-t_abs0:.1f} s")
        print(f"  Dropouts ({len(cov['dropouts'])}):")
        for d in cov["dropouts"][:20]:
            print(f"    {d['t0']-t_abs0:7.2f} -> {d['t1']-t_abs0:7.2f} s "
                  f"({d['dur']:5.2f} s, {d['gt_dist']:5.2f} m of GT motion lost)")
        if len(cov["dropouts"]) > 20:
            print(f"    ... {len(cov['dropouts'])-20} more (see dropouts.csv)")
        m("coverage", "largest_dropout", b["dur"], "s")
        m("coverage", "largest_dropout_distance", b["gt_dist"], "m")
    else:
        print("  No dropouts: the trajectory spans the whole run without a "
              "gap longer than the threshold.")
        m("coverage", "largest_dropout", 0.0, "s")
        m("coverage", "largest_dropout_distance", 0.0, "m")

    ss = None
    if states_raw:
        st_t, st_s, st_m = states_raw
        ss = summarise_states(st_t + offset, st_s, st_m)
        print("\nTracker state log")
        print("  frames OK: "
              f"{ss['ok_frac']*100:6.2f} %   " +
              "  ".join(f"{k}={v}" for k, v in ss["counts"].items()))
        for a_, b_, nm in ss["lost_intervals"]:
            print(f"    lost {a_-t_abs0:7.2f} -> {b_-t_abs0:7.2f} s "
                  f"({b_-a_:5.2f} s, entered as {nm})")
        m("tracker", "frames_logged", len(st_s))
        m("tracker", "frames_ok_pct", ss["ok_frac"] * 100, "%")
        for k, v in ss["counts"].items():
            m("tracker", f"frames_{k}", v)
        m("tracker", "lost_intervals", len(ss["lost_intervals"]))
        if ss["maps"] is not None:
            print(f"  Atlas maps touched: {ss['maps']}  "
                  f"({len(ss['maps'])} map(s), {len(ss['map_switches'])} switch(es))")
            for tt, m0, m1 in ss["map_switches"]:
                print(f"    {tt-t_abs0:7.2f} s : map {m0} -> {m1}")
            m("tracker", "atlas_maps", len(ss["maps"]))
            m("tracker", "atlas_map_switches", len(ss["map_switches"]))

    # ---- segment / seam analysis ------------------------------------------
    seg_assoc = find_segments(t_gt[j], cov["gap_thresh"])
    seg_res = segment_errors(seg_assoc, p_sl, p_gt, align=align)
    seams = []
    if len(seg_res) > 1:
        print("\nPer-segment ATE (each segment aligned independently)")
        for r_ in seg_res:
            if r_["rmse"] is None:
                print(f"  seg {r_['id']}: {r_['n']:4d} poses  (too short)")
            else:
                print(f"  seg {r_['id']}: {r_['n']:4d} poses  "
                      f"RMSE {r_['rmse']:.4f} m   best-fit scale {r_['scale']:.4f}")
        good = [r_["rmse"] for r_ in seg_res if r_["rmse"] is not None]
        if good:
            print(f"  worst segment: {max(good):.4f} m   "
                  f"global ATE / worst segment = {a['rmse']/max(good):.2f}x")
            m("segments", "worst_segment_ate_rmse", max(good), "m")
            m("segments", "global_over_worst_segment", a["rmse"] / max(good))

        seams = seam_errors(seg_assoc, t_rel, p_al, q_al, p_gt, q_gt)
        print("\nMerge / re-localisation seams (RPE across each gap)")
        for sm in seams:
            print(f"  {sm['t0']:7.2f} -> {sm['t1']:7.2f} s : "
                  f"translation {sm['trans']:.4f} m, rotation {sm['rot']:.3f} deg "
                  f"(GT moved {sm['gt_move']:.2f} m)")
        if seams:
            m("seams", "count", len(seams))
            m("seams", "max_trans_error", max(x["trans"] for x in seams), "m")
            m("seams", "max_rot_error", max(x["rot"] for x in seams), "deg")
            m("seams", "mean_trans_error",
              float(np.mean([x["trans"] for x in seams])), "m")

    # ---- outputs ------------------------------------------------------------
    ctx = dict(t_rel=t_rel, p_al=p_al, p_gt=p_gt, ate=a, ape_rot=ape_rot, rpe=r,
               delta=rpe_delta, per_second=per_second, cov=cov, segs=seg_assoc,
               seg_res=seg_res, seams=seams, t_abs0=t_abs0, states=ss)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        figs = export_figures(ctx, outdir, titles=titles, fmt=fig_format, dpi=dpi)
        tabs = export_tables(ctx, M, outdir)
        print(f"\n{len(figs)} figures and {len(tabs)} tables -> {outdir}/")
        for p in figs + tabs:
            print(f"  {os.path.basename(p)}")

    if out_csv:  # legacy --dump
        write_table(out_csv,
                    ["t_s", "gt_x_m", "gt_y_m", "gt_z_m",
                     "slam_x_m", "slam_y_m", "slam_z_m", "ape_trans_m"],
                    [[t_rel[k], *p_gt[k], *p_al[k], a["errors"][k]]
                     for k in range(len(t_rel))])
        print(f"Per-pose errors -> {out_csv}")

    if out_png:
        make_dashboard(ctx, out_png)

    if append_to:
        row = {f"{sec}_{met}" if sec != "run" else met: val
               for sec, met, val, _u in M}
        append_results_row(append_to, row)
        print(f"Run appended -> {append_to}")

    return M, ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="dump ground truth from a ROS 2 bag")
    e.add_argument("--bag", required=True)
    e.add_argument("--topic", default="/odometry/sim")
    e.add_argument("--out", default="ground_truth_pose.csv")
    e.add_argument("--msg-type", default="nav_msgs/msg/Odometry")

    f = sub.add_parser("frames", help="dump camera frame timestamps from a bag")
    f.add_argument("--bag", required=True)
    f.add_argument("--topic", required=True)
    f.add_argument("--out", default="frame_times.txt")
    f.add_argument("--msg-type", default="sensor_msgs/msg/Image")

    v = sub.add_parser("eval", help="compare SLAM output against ground truth")
    v.add_argument("--gt", required=True)
    v.add_argument("--slam", required=True)
    v.add_argument("--align", choices=["se3", "sim3"], default="se3",
                   help="se3 for stereo/RGB-D (metric), sim3 for monocular")
    v.add_argument("--max-dt", type=float, default=0.02)
    v.add_argument("--rpe-delta", type=float, default=1.0)
    v.add_argument("--rpe-raw", action="store_true",
                   help="do not normalise RPE per second")
    v.add_argument("--auto-offset", action="store_true")
    v.add_argument("--no-extrinsic", action="store_true",
                   help="skip constant camera->body rotation estimation")
    v.add_argument("--frames", default=None,
                   help="camera timestamps fed to ORB-SLAM3 (one per line)")
    v.add_argument("--state", default=None,
                   help="per-frame tracking state CSV from the instrumentation "
                        "patch (timestamp,state[,map_id])")
    v.add_argument("--gap", type=float, default=None,
                   help="seconds without a pose that counts as a dropout "
                        "(default max(0.5 s, 5x median pose spacing))")
    v.add_argument("--outdir", default=None,
                   help="write every figure and table as separate files here")
    v.add_argument("--label", default=None,
                   help="run name used in the summary tables (default: slam file name)")
    v.add_argument("--append-to", default=None,
                   help="append this run as one row to a results CSV shared "
                        "across runs")
    v.add_argument("--titles", action="store_true",
                   help="put titles on the individual figures (off by default)")
    v.add_argument("--fig-format", default="png", choices=["png", "pdf", "svg"])
    v.add_argument("--dpi", type=int, default=200)
    v.add_argument("--plot", default=None, help="combined dashboard image")
    v.add_argument("--dump", default=None, help="per-pose error CSV (legacy)")

    a = ap.parse_args()
    if a.cmd == "extract":
        extract_gt(a.bag, a.topic, a.out, a.msg_type)
    elif a.cmd == "frames":
        extract_frame_times(a.bag, a.topic, a.out, a.msg_type)
    else:
        evaluate(a.gt, a.slam, align=a.align, max_dt=a.max_dt,
                 rpe_delta=a.rpe_delta, per_second=not a.rpe_raw,
                 auto_offset=a.auto_offset, fix_extrinsic=not a.no_extrinsic,
                 out_png=a.plot, out_csv=a.dump, frames_file=a.frames,
                 gap_thresh=a.gap, state_file=a.state, outdir=a.outdir,
                 label=a.label, append_to=a.append_to, titles=a.titles,
                 fig_format=a.fig_format, dpi=a.dpi)


if __name__ == "__main__":
    sys.exit(main())

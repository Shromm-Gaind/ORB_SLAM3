"""Python side of the DormantTrackBuffer parity harness. Runs the same
operation script as tests/parity_dump.cc and prints the same CSV.

Locate dormant_buffer.py by any of, in order:
  1. --module-dir DIR
  2. $HYBRID_PYTHON_DIR
  3. ../python, .., or the current directory
"""
import argparse
import os
import sys

import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--module-dir", default=None,
                 help="directory containing dormant_buffer.py")
_args = _ap.parse_args()

_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _args.module_dir,
    os.environ.get("HYBRID_PYTHON_DIR"),
    os.path.join(_here, "..", "python"),
    os.path.join(_here, ".."),
    os.getcwd(),
]
for _c in _candidates:
    if _c and os.path.isfile(os.path.join(_c, "dormant_buffer.py")):
        sys.path.insert(0, _c)
        break
else:
    sys.exit("could not find dormant_buffer.py; pass --module-dir DIR")

from dormant_buffer import DormantTrack, DormantTrackBuffer  # noqa: E402


def mk(i, x, y, died, octave, age):
    return DormantTrack(id=i, last_x=x, last_y=y,
                        descriptor=np.full(32, i & 0xFF, dtype=np.uint8),
                        frame_died=died, octave=octave, age_at_death=age)


def dump(tag, b):
    parts = [f"{tag}|size={len(b)}"]
    for e in b.all_entries():
        parts.append(f"{e.id}:{e.last_x:.2f},{e.last_y:.2f}:"
                     f"d{e.frame_died}:o{e.octave}:a{e.age_at_death}")
    print("|".join(parts))


def dump_query(tag, b, x, y, r):
    hits = b.query_within(x, y, r)
    parts = [f"{tag}|n={len(hits)}"] + [str(h.id) for h in hits]
    print("|".join(parts))


def gap_frames(e, current):
    return current - e.frame_died if current > e.frame_died else 0


buf = DormantTrackBuffer(30)
buf.add(mk(1, 100.0, 100.0, 100, 0, 5))
buf.add(mk(2, 110.0, 100.0, 100, 2, 40))
buf.add(mk(3, 110.0, 110.0, 105, 5, 300))
buf.add(mk(4, 500.0, 500.0, 120, 7, 1))
dump("after_adds", buf)

dump_query("q_r10", buf, 100.0, 100.0, 10.0)
dump_query("q_r0", buf, 100.0, 100.0, 0.0)
dump_query("q_rneg", buf, 100.0, 100.0, -5.0)
dump_query("q_r9_99", buf, 100.0, 100.0, 9.99)

buf.translate_all(5.0, -2.0)
dump("after_translate", buf)
dump_query("q_after_tr", buf, 105.0, 98.0, 10.0)

buf.translate_all(0.0, 0.0)
buf.translate_all(-5.0, 2.0)
dump("after_translate_undo", buf)

buf.purge_older_than(5);   dump("purge_underflow", buf)
buf.purge_older_than(130); dump("purge_at_horizon", buf)
buf.purge_older_than(131); dump("purge_beyond", buf)
buf.purge_older_than(136); dump("purge_more", buf)

print(f"remove_hit={int(buf.remove(4))}")
print(f"remove_miss={int(buf.remove(999))}")
dump("after_remove", buf)

buf.add(mk(9, 10.0, 10.0, 140, 1, 7))
buf.clear()
dump("after_clear", buf)

t = mk(1, 0.0, 0.0, 100, 0, 0)
print(f"gap|{gap_frames(t,100)}|{gap_frames(t,130)}|{gap_frames(t,50)}")

# ---- SpatialDescriptorMatcher parity (same scenario as parity_dump.cc) ----
from spatial_descriptor_matcher import (  # noqa: E402
    MatchOptions, PixelCandidate, PixelQuery, spatial_descriptor_match,
)


def bits(n):
    d = np.zeros(32, dtype=np.uint8)
    for i in range(n):
        d[i // 8] |= np.uint8(1 << (i % 8))
    return d


qs = [
    PixelQuery(100.0, 100.0, bits(0), -1.0),
    PixelQuery(200.0, 200.0, bits(0), 40.0),
    PixelQuery(300.0, 300.0, bits(0), -1.0),
    PixelQuery(400.0, 400.0, bits(0), -1.0),
    PixelQuery(500.0, 500.0, bits(0), -1.0),
    PixelQuery(600.0, 600.0, bits(30), -1.0),
    PixelQuery(601.0, 600.0, bits(10), -1.0),
]
cs = [
    PixelCandidate(101.0, 100.0, bits(80), -1),
    PixelCandidate(102.0, 100.0, bits(20), -1),
    PixelCandidate(103.0, 100.0, bits(60), -1),
    PixelCandidate(230.0, 200.0, bits(40), 90),
    PixelCandidate(301.0, 300.0, bits(20), -1),
    PixelCandidate(302.0, 300.0, bits(22), -1),
    PixelCandidate(401.0, 400.0, bits(45), -1),
    PixelCandidate(501.0, 500.0, bits(25), 10),
    PixelCandidate(502.0, 500.0, bits(20), -1),
    PixelCandidate(602.0, 600.0, bits(0), -1),
    PixelCandidate(650.0, 600.0, bits(200), -1),
]
o = MatchOptions(default_radius=20.0, hamming_threshold=100,
                 unique_candidates=True, second_best_margin=15)
for i, m in enumerate(spatial_descriptor_match(qs, cs, o)):
    if m is None:
        print(f"match{i}|none")
    else:
        print(f"match{i}|{m.candidate_index}|{m.hamming_distance}")
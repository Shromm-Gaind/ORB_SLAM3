"""
test_python_modules.py

Quick parity tests for the Python ports of DormantTrackBuffer and
SpatialDescriptorMatcher. These mirror the C++ unit tests in spirit
(not 1:1) to catch any drift between the two implementations before
we use them in the proof-of-concept pipeline.

Run with: python -m pytest test_python_modules.py -v
Or just:  python test_python_modules.py
"""

import numpy as np
import pytest

from dormant_buffer import DormantTrack, DormantTrackBuffer
from spatial_descriptor_matcher import (
    Match,
    MatchOptions,
    PixelCandidate,
    PixelQuery,
    hamming_distance,
    hamming_distance_batch,
    spatial_descriptor_match,
)


def desc(fill: int = 0) -> np.ndarray:
    return np.full((32,), fill, dtype=np.uint8)


def desc_with_bits(n: int) -> np.ndarray:
    d = np.zeros(32, dtype=np.uint8)
    for i in range(n):
        d[i // 8] |= 1 << (i % 8)
    return d


# ---------------- DormantTrackBuffer ----------------

class TestDormantBuffer:
    def make(self, tid, x, y, frame_died=0):
        return DormantTrack(id=tid, last_x=x, last_y=y,
                            descriptor=desc(0), frame_died=frame_died)

    def test_empty(self):
        buf = DormantTrackBuffer(30)
        assert len(buf) == 0
        assert buf.empty()
        assert buf.query_within(0, 0, 5) == []

    def test_linf_radius(self):
        # Mirrors test_DormantTrackBuffer.cc QueryLInfinityShape
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 110, 100))
        buf.add(self.make(2, 100, 110))
        buf.add(self.make(3, 110, 110))     # corner — L-inf included, L2 not
        buf.add(self.make(4, 111, 100))     # just outside
        ids = sorted(e.id for e in buf.query_within(100, 100, 10))
        assert ids == [1, 2, 3]

    def test_purge_boundary_inclusive(self):
        # An entry whose frame_died == current_frame - horizon is STILL FRESH.
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 0, 0, frame_died=100))
        buf.purge_older_than(130)   # cutoff = 100; not strictly less than 100
        assert len(buf) == 1
        buf.purge_older_than(131)
        assert len(buf) == 0

    def test_underflow_safe(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 0, 0, frame_died=2))
        buf.purge_older_than(5)   # would underflow if computed naively
        assert len(buf) == 1

    def test_remove(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 0, 0))
        buf.add(self.make(2, 0, 0))
        assert buf.remove(1) is True
        assert buf.remove(999) is False
        assert len(buf) == 1

    def test_add_map_point_raises(self):
        buf = DormantTrackBuffer(30)
        t = self.make(1, 0, 0)
        t.map_point = object()
        with pytest.raises(AssertionError):
            buf.add(t)

    def test_add_duplicate_id_raises(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 0, 0))
        with pytest.raises(AssertionError):
            buf.add(self.make(1, 10, 10))

    def test_clear(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 0, 0))
        buf.add(self.make(2, 0, 0))
        buf.clear()
        assert len(buf) == 0


# ---------------- Hamming ----------------

class TestHamming:
    def test_zero(self):
        assert hamming_distance(desc(0), desc(0)) == 0

    def test_all_ones_vs_zero(self):
        assert hamming_distance(desc(0xFF), desc(0)) == 256

    def test_known_count(self):
        assert hamming_distance(desc_with_bits(50), desc(0)) == 50

    def test_symmetry(self):
        a = desc_with_bits(37)
        b = desc_with_bits(91)
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_batch_matches_scalar(self):
        # Batch must agree with scalar implementation.
        rng = np.random.default_rng(0)
        cands = rng.integers(0, 256, size=(20, 32), dtype=np.uint8)
        q = rng.integers(0, 256, size=(32,), dtype=np.uint8)
        scalar = np.array([hamming_distance(q, c) for c in cands])
        batch = hamming_distance_batch(q, cands)
        assert np.array_equal(scalar, batch)


# ---------------- SpatialDescriptorMatch ----------------

class TestMatcher:
    def test_empty_inputs(self):
        opts = MatchOptions()
        assert spatial_descriptor_match([], [PixelCandidate(0, 0, desc(0))], opts) == []
        out = spatial_descriptor_match([PixelQuery(0, 0, desc(0))], [], opts)
        assert out == [None]

    def test_exact_match(self):
        opts = MatchOptions(default_radius=20, hamming_threshold=50)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(105, 100, desc(0))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].candidate_index == 0
        assert out[0].hamming_distance == 0

    def test_outside_radius_misses(self):
        opts = MatchOptions(default_radius=20)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(200, 200, desc(0))]
        assert spatial_descriptor_match(q, c, opts)[0] is None

    def test_hamming_at_threshold_accepted(self):
        opts = MatchOptions(default_radius=20, hamming_threshold=50)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(100, 100, desc_with_bits(50))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].hamming_distance == 50

    def test_picks_lowest_hamming(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(80)),
            PixelCandidate(102, 100, desc_with_bits(20)),  # best
            PixelCandidate(103, 100, desc_with_bits(60)),
        ]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0].candidate_index == 1
        assert out[0].hamming_distance == 20

    def test_per_query_radius(self):
        opts = MatchOptions(default_radius=5, hamming_threshold=256)
        q = [
            PixelQuery(100, 100, desc(0), radius=-1.0),  # default 5
            PixelQuery(100, 100, desc(0), radius=20.0),
        ]
        c = [PixelCandidate(115, 100, desc(0))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is None
        assert out[1] is not None

    def test_unique_lower_hamming_wins(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            unique_candidates=True)
        q = [
            PixelQuery(100, 100, desc_with_bits(40)),
            PixelQuery(101, 100, desc_with_bits(10)),
        ]
        c = [PixelCandidate(102, 100, desc(0))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is None
        assert out[1] is not None
        assert out[1].hamming_distance == 10

    def test_unique_tie_lower_index_wins(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            unique_candidates=True)
        pattern = desc_with_bits(20)
        q = [
            PixelQuery(100, 100, pattern),
            PixelQuery(101, 100, pattern),
        ]
        c = [PixelCandidate(102, 100, desc(0))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[1] is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

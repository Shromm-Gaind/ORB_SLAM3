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


# ---------------- second_best_margin (distinctiveness gate) ----------------

class TestMarginGate:
    def test_ambiguous_rejected(self):
        # Two candidates 5 bits apart < margin 10 -> refuse to match.
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            second_best_margin=10)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20)),
            PixelCandidate(102, 100, desc_with_bits(25)),
        ]
        assert spatial_descriptor_match(q, c, opts)[0] is None

    def test_distinct_accepted(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            second_best_margin=10)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20)),
            PixelCandidate(102, 100, desc_with_bits(60)),  # gap 40 >= 10
        ]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].candidate_index == 0
        assert out[0].hamming_distance == 20

    def test_single_candidate_passes_trivially(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            second_best_margin=100)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(101, 100, desc_with_bits(20))]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None

    def test_competitor_above_threshold_still_blocks(self):
        # Second-best is over the Hamming threshold, but its existence
        # is still evidence of ambiguity — the margin applies to ALL
        # spatially-gated candidates.
        opts = MatchOptions(default_radius=50, hamming_threshold=22,
                            second_best_margin=10)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20)),  # passes threshold
            PixelCandidate(102, 100, desc_with_bits(25)),  # over threshold
        ]
        assert spatial_descriptor_match(q, c, opts)[0] is None

    def test_out_of_radius_competitor_ignored(self):
        # A near-identical competitor OUTSIDE the spatial gate must not
        # trigger the ambiguity rejection.
        opts = MatchOptions(default_radius=5, hamming_threshold=256,
                            second_best_margin=10)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20)),
            PixelCandidate(500, 500, desc_with_bits(21)),  # far away
        ]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].candidate_index == 0

    def test_margin_zero_is_legacy_behaviour(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            second_best_margin=0)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20)),
            PixelCandidate(102, 100, desc_with_bits(21)),
        ]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].candidate_index == 0


# ---------------- DormantTrackBuffer extras ----------------

class TestDormantBufferExtras:
    def make(self, tid, x, y, frame_died=0):
        return DormantTrack(id=tid, last_x=x, last_y=y,
                            descriptor=np.zeros(32, dtype=np.uint8),
                            frame_died=frame_died)

    def test_translate_all(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 100, 100))
        buf.add(self.make(2, 200, 50))
        buf.translate_all(3.0, -2.0)
        hits = buf.query_within(103, 98, 0.5)
        assert [e.id for e in hits] == [1]
        hits = buf.query_within(203, 48, 0.5)
        assert [e.id for e in hits] == [2]

    def test_translate_all_zero_noop(self):
        buf = DormantTrackBuffer(30)
        buf.add(self.make(1, 100, 100))
        buf.translate_all(0.0, 0.0)
        assert [e.id for e in buf.query_within(100, 100, 0.5)] == [1]

    def test_age_at_death_default_and_roundtrip(self):
        t = self.make(1, 0, 0)
        assert t.age_at_death == 0
        t2 = DormantTrack(id=2, last_x=0, last_y=0,
                          descriptor=np.zeros(32, dtype=np.uint8),
                          frame_died=5, age_at_death=42)
        buf = DormantTrackBuffer(30)
        buf.add(t2)
        assert buf.query_within(0, 0, 1)[0].age_at_death == 42


# ---------------- per-candidate Hamming threshold ----------------

class TestPerCandidateThreshold:
    def test_override_loosens_for_one_candidate(self):
        # Default threshold 25 would reject a distance-30 candidate;
        # its per-candidate override of 40 accepts it.
        opts = MatchOptions(default_radius=50, hamming_threshold=25)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(101, 100, desc_with_bits(30),
                            hamming_threshold=40)]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].hamming_distance == 30

    def test_default_applies_when_not_overridden(self):
        opts = MatchOptions(default_radius=50, hamming_threshold=25)
        q = [PixelQuery(100, 100, desc(0))]
        c = [PixelCandidate(101, 100, desc_with_bits(30))]  # thr -1 -> 25
        assert spatial_descriptor_match(q, c, opts)[0] is None

    def test_best_is_lowest_among_passing(self):
        # Global-min candidate fails its own strict threshold; the next
        # one passes its looser threshold and wins.
        opts = MatchOptions(default_radius=50, hamming_threshold=256)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20),
                           hamming_threshold=10),   # dist 20 > own 10
            PixelCandidate(102, 100, desc_with_bits(35),
                           hamming_threshold=40),   # dist 35 <= own 40
        ]
        out = spatial_descriptor_match(q, c, opts)
        assert out[0] is not None
        assert out[0].candidate_index == 1
        assert out[0].hamming_distance == 35

    def test_rejected_better_candidate_still_triggers_margin(self):
        # Same as above but with a margin: the rejected candidate has a
        # LOWER distance than the accepted one, which is ambiguity
        # evidence — the margin gate must refuse the match.
        opts = MatchOptions(default_radius=50, hamming_threshold=256,
                            second_best_margin=5)
        q = [PixelQuery(100, 100, desc(0))]
        c = [
            PixelCandidate(101, 100, desc_with_bits(20),
                           hamming_threshold=10),
            PixelCandidate(102, 100, desc_with_bits(35),
                           hamming_threshold=40),
        ]
        assert spatial_descriptor_match(q, c, opts)[0] is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------- representative descriptor (medoid) ----------------

from hybrid_frontend import _representative_descriptor  # noqa: E402


class TestRepresentativeDescriptor:
    def test_single_observation(self):
        d = desc_with_bits(10)
        assert np.array_equal(_representative_descriptor([d]), d)

    def test_picks_central_not_outlier(self):
        # Three near-identical observations plus one wildly corrupted
        # one. The medoid must be one of the cluster, never the outlier.
        cluster = [desc_with_bits(20), desc_with_bits(21), desc_with_bits(22)]
        outlier = desc(0xFF)
        rep = _representative_descriptor(cluster + [outlier])
        assert not np.array_equal(rep, outlier)
        assert any(np.array_equal(rep, c) for c in cluster)

    def test_robust_to_two_outliers_via_median(self):
        # The MEDIAN (not mean) criterion is what makes this robust:
        # with 5 clustered and 2 corrupted observations, the medoid stays
        # in the cluster even though the outliers inflate mean distance.
        cluster = [desc_with_bits(30 + i) for i in range(5)]
        outliers = [desc(0xFF), desc(0xF0)]
        rep = _representative_descriptor(cluster + outliers)
        assert any(np.array_equal(rep, c) for c in cluster)

    def test_returns_an_actual_observation(self):
        obs = [desc_with_bits(5), desc_with_bits(50), desc_with_bits(95)]
        rep = _representative_descriptor(obs)
        assert any(np.array_equal(rep, o) for o in obs), (
            "medoid must be one of the inputs, not a synthesized descriptor"
        )
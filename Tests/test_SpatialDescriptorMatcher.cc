// test_SpatialDescriptorMatcher.cc
//
// Unit tests for SpatialDescriptorMatcher and HammingDistance.

#include "SpatialDescriptorMatcher.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

using hybrid_frontend::Descriptor256;
using hybrid_frontend::HammingDistance;
using hybrid_frontend::Match;
using hybrid_frontend::MatchOptions;
using hybrid_frontend::PixelCandidate;
using hybrid_frontend::PixelQuery;
using hybrid_frontend::SpatialDescriptorMatch;

namespace {

Descriptor256 desc_zeros() {
    Descriptor256 d{};
    d.fill(0);
    return d;
}
Descriptor256 desc_ones() {
    Descriptor256 d{};
    d.fill(0xFF);
    return d;
}
Descriptor256 desc_with_bits(int n_ones) {
    // Set the first n_ones bits to 1.
    Descriptor256 d{};
    d.fill(0);
    for (int i = 0; i < n_ones; ++i) {
        d[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
    }
    return d;
}

PixelQuery query_at(float x, float y, const Descriptor256& d, float r = -1.0f) {
    PixelQuery q;
    q.x = x;
    q.y = y;
    q.descriptor = d;
    q.radius = r;
    return q;
}

PixelCandidate cand_at(float x, float y, const Descriptor256& d) {
    PixelCandidate c;
    c.x = x;
    c.y = y;
    c.descriptor = d;
    return c;
}

}  // namespace

// ---- HammingDistance ------------------------------------------------

TEST(HammingDistance, ZeroVsZero) {
    EXPECT_EQ(HammingDistance(desc_zeros(), desc_zeros()), 0);
}

TEST(HammingDistance, OnesVsOnes) {
    EXPECT_EQ(HammingDistance(desc_ones(), desc_ones()), 0);
}

TEST(HammingDistance, ZeroVsOnes) {
    // Maximally different: every one of 256 bits is set in one and not
    // the other.
    EXPECT_EQ(HammingDistance(desc_zeros(), desc_ones()), 256);
}

TEST(HammingDistance, KnownBitCount) {
    // 50 set bits in the first descriptor, 0 in the second → Hamming = 50.
    EXPECT_EQ(HammingDistance(desc_with_bits(50), desc_zeros()), 50);
}

TEST(HammingDistance, Symmetry) {
    const auto a = desc_with_bits(37);
    const auto b = desc_with_bits(91);
    EXPECT_EQ(HammingDistance(a, b), HammingDistance(b, a));
}

TEST(HammingDistance, EachByteIndependent) {
    // Sanity-check byte indexing in desc_with_bits and HammingDistance:
    // 8 bits per byte, set bit 0 of byte 5 (bit index 40 overall) and
    // verify Hamming against all-zero == 1.
    Descriptor256 d{};
    d.fill(0);
    d[5] = 0x01;
    EXPECT_EQ(HammingDistance(d, desc_zeros()), 1);
    d[5] = 0xFF;
    EXPECT_EQ(HammingDistance(d, desc_zeros()), 8);
}

// ---- Matcher: degenerate inputs -------------------------------------

TEST(SpatialDescriptorMatcher, EmptyQueries) {
    std::vector<PixelQuery> q;
    std::vector<PixelCandidate> c{cand_at(0.0f, 0.0f, desc_zeros())};
    auto out = SpatialDescriptorMatch(q, c, {});
    EXPECT_TRUE(out.empty());
}

TEST(SpatialDescriptorMatcher, EmptyCandidates) {
    std::vector<PixelQuery> q{query_at(0.0f, 0.0f, desc_zeros())};
    std::vector<PixelCandidate> c;
    auto out = SpatialDescriptorMatch(q, c, {});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FALSE(out[0].has_value());
}

// ---- Matcher: basic hits and misses ---------------------------------

TEST(SpatialDescriptorMatcher, ExactMatchWithinRadius) {
    MatchOptions opts;
    opts.default_radius = 20.0f;
    opts.hamming_threshold = 50;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{cand_at(105.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->candidate_index, 0u);
    EXPECT_EQ(out[0]->hamming_distance, 0);
}

TEST(SpatialDescriptorMatcher, OutsideRadiusReturnsNullopt) {
    MatchOptions opts;
    opts.default_radius = 20.0f;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{cand_at(200.0f, 200.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    EXPECT_FALSE(out[0].has_value());
}

TEST(SpatialDescriptorMatcher, HammingAboveThresholdReturnsNullopt) {
    MatchOptions opts;
    opts.default_radius = 20.0f;
    opts.hamming_threshold = 50;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    // Same position, but Hamming distance is 100 -> over threshold
    std::vector<PixelCandidate> c{cand_at(100.0f, 100.0f, desc_with_bits(100))};

    auto out = SpatialDescriptorMatch(q, c, opts);
    EXPECT_FALSE(out[0].has_value());
}

TEST(SpatialDescriptorMatcher, HammingAtThresholdAccepted) {
    // The threshold is inclusive (see header doc).
    MatchOptions opts;
    opts.default_radius = 20.0f;
    opts.hamming_threshold = 50;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{cand_at(100.0f, 100.0f, desc_with_bits(50))};

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->hamming_distance, 50);
}

TEST(SpatialDescriptorMatcher, BoundaryRadiusIncluded) {
    MatchOptions opts;
    opts.default_radius = 10.0f;
    opts.hamming_threshold = 256;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{cand_at(110.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    EXPECT_TRUE(out[0].has_value());
}

// ---- Matcher: best-of-many ------------------------------------------

TEST(SpatialDescriptorMatcher, PicksLowestHammingAmongCandidates) {
    MatchOptions opts;
    opts.default_radius = 50.0f;
    opts.hamming_threshold = 256;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{
        cand_at(101.0f, 100.0f, desc_with_bits(80)),  // Hamming 80
        cand_at(102.0f, 100.0f, desc_with_bits(20)),  // Hamming 20  <- best
        cand_at(103.0f, 100.0f, desc_with_bits(60)),  // Hamming 60
    };

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->candidate_index, 1u);
    EXPECT_EQ(out[0]->hamming_distance, 20);
}

TEST(SpatialDescriptorMatcher, IgnoresOutOfRadiusEvenIfBetter) {
    // A globally-better candidate that lies outside the radius must be
    // ignored. This is the gate that protects Step 5b from pulling in
    // matches from across the image.
    MatchOptions opts;
    opts.default_radius = 5.0f;
    opts.hamming_threshold = 256;

    std::vector<PixelQuery> q{query_at(100.0f, 100.0f, desc_zeros())};
    std::vector<PixelCandidate> c{
        cand_at(103.0f, 100.0f, desc_with_bits(60)),   // in radius, Hamming 60
        cand_at(500.0f, 500.0f, desc_zeros()),         // out of radius, perfect descriptor
    };

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->candidate_index, 0u);
    EXPECT_EQ(out[0]->hamming_distance, 60);
}

// ---- Matcher: per-query radius override -----------------------------

TEST(SpatialDescriptorMatcher, PerQueryRadiusOverridesDefault) {
    // Two queries: one with default radius (small, miss), one with
    // larger per-query radius (hit). Models the §4.7 octave-scaled
    // r_TLM(ℓ) case.
    MatchOptions opts;
    opts.default_radius = 5.0f;
    opts.hamming_threshold = 256;

    std::vector<PixelQuery> q{
        query_at(100.0f, 100.0f, desc_zeros(), /*radius=*/-1.0f),  // default 5
        query_at(100.0f, 100.0f, desc_zeros(), /*radius=*/20.0f),  // override 20
    };
    std::vector<PixelCandidate> c{cand_at(115.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    EXPECT_FALSE(out[0].has_value());  // 15 px > 5 px
    EXPECT_TRUE(out[1].has_value());   // 15 px <= 20 px
}

// ---- Matcher: unique_candidates -------------------------------------

TEST(SpatialDescriptorMatcher, NonUniqueAllowsCollision) {
    // Default mode: two queries can both claim the same candidate. The
    // caller resolves conflicts if it cares.
    MatchOptions opts;
    opts.default_radius = 50.0f;
    opts.hamming_threshold = 256;
    opts.unique_candidates = false;

    std::vector<PixelQuery> q{
        query_at(100.0f, 100.0f, desc_with_bits(10)),
        query_at(101.0f, 100.0f, desc_with_bits(30)),
    };
    std::vector<PixelCandidate> c{cand_at(102.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    ASSERT_TRUE(out[1].has_value());
    EXPECT_EQ(out[0]->candidate_index, 0u);
    EXPECT_EQ(out[1]->candidate_index, 0u);  // collision allowed
}

TEST(SpatialDescriptorMatcher, UniqueModeLowerHammingWins) {
    MatchOptions opts;
    opts.default_radius = 50.0f;
    opts.hamming_threshold = 256;
    opts.unique_candidates = true;

    std::vector<PixelQuery> q{
        query_at(100.0f, 100.0f, desc_with_bits(40)),  // worse (40)
        query_at(101.0f, 100.0f, desc_with_bits(10)),  // better (10)
    };
    std::vector<PixelCandidate> c{cand_at(102.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    EXPECT_FALSE(out[0].has_value());  // loses
    ASSERT_TRUE(out[1].has_value());
    EXPECT_EQ(out[1]->candidate_index, 0u);
    EXPECT_EQ(out[1]->hamming_distance, 10);
}

TEST(SpatialDescriptorMatcher, UniqueModeTieGoesToLowerIndex) {
    MatchOptions opts;
    opts.default_radius = 50.0f;
    opts.hamming_threshold = 256;
    opts.unique_candidates = true;

    const auto pattern = desc_with_bits(20);
    std::vector<PixelQuery> q{
        query_at(100.0f, 100.0f, pattern),
        query_at(101.0f, 100.0f, pattern),
    };
    std::vector<PixelCandidate> c{cand_at(102.0f, 100.0f, desc_zeros())};

    auto out = SpatialDescriptorMatch(q, c, opts);
    // Hamming distances are equal (20). Documented tie-break: lower query
    // index keeps the candidate.
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->candidate_index, 0u);
    EXPECT_FALSE(out[1].has_value());
}

TEST(SpatialDescriptorMatcher, UniqueModeMultipleCandidatesIndependent) {
    // Two queries, two candidates. Each query picks its best; uniqueness
    // is not triggered because the picks are different.
    MatchOptions opts;
    opts.default_radius = 50.0f;
    opts.hamming_threshold = 256;
    opts.unique_candidates = true;

    std::vector<PixelQuery> q{
        query_at(100.0f, 100.0f, desc_zeros()),
        query_at(200.0f, 200.0f, desc_with_bits(10)),
    };
    std::vector<PixelCandidate> c{
        cand_at(100.0f, 100.0f, desc_zeros()),        // best for q0
        cand_at(200.0f, 200.0f, desc_with_bits(10)),  // best for q1
    };

    auto out = SpatialDescriptorMatch(q, c, opts);
    ASSERT_TRUE(out[0].has_value());
    EXPECT_EQ(out[0]->candidate_index, 0u);
    ASSERT_TRUE(out[1].has_value());
    EXPECT_EQ(out[1]->candidate_index, 1u);
}

// Shared primitive for Step 5 (dormant track re-ID) and Step 5b
// (TrackLocalMap). Both steps reduce to: given a list of predicted
// query locations + descriptors, and a list of candidate locations +
// descriptors, return the best candidate per query under (a) a spatial
// L-infinity radius gate and (b) a Hamming-distance gate.
//
// No state. No ORB-SLAM3 dependency. Unit-testable on synthetic inputs.
//
// Design references: §4.6 (Step 5), §4.7 (Step 5b), §4.8 (shared primitive).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace hybrid_frontend {

using Descriptor256 = std::array<uint8_t, 32>;

struct PixelQuery {
    float x = 0.0f;
    float y = 0.0f;
    Descriptor256 descriptor{};
    // Optional per-query spatial radius. If <= 0, the matcher uses the
    // default radius passed in MatchOptions. This supports §4.7's
    // octave-scaled r_TLM(ℓ) without forcing every caller to grow it.
    float radius = -1.0f;
};

struct PixelCandidate {
    float x = 0.0f;
    float y = 0.0f;
    Descriptor256 descriptor{};
    // Optional per-candidate Hamming ceiling (inclusive). If <= 0, the
    // matcher uses opts.hamming_threshold. Mirrors PixelQuery::radius.
    //
    // This supports gap-scaled re-ID acceptance (§4.6): a dormant
    // candidate that died g frames ago has accumulated ~g frames of
    // viewpoint-driven descriptor drift, so its acceptance ceiling is
    //     theta_eff(g) = min(theta_cap, theta_base + slope * g)
    // computed by the caller (DormantTrackBuffer::gap_frames supplies
    // g), while a 1-frame-gap candidate keeps the strict base
    // threshold.
    int hamming_threshold = -1;
};

struct MatchOptions {
    // Default L-infinity (square window) spatial radius in pixels. Used
    // when a query's own `radius` is <= 0. Corresponds to r_reid (§4.6)
    // or r_TLM (§4.7) depending on caller.
    float default_radius = 20.0f;

    // Hamming distance ceiling (inclusive on equality). 50/256 is the
    // ORB-SLAM3 default for both θ_reid and θ_TLM (§8). A query whose
    // best candidate has Hamming > this value returns nullopt.
    int hamming_threshold = 50;

    // If true, each candidate is awarded to at most one query (the one
    // with the smaller Hamming distance; ties broken by lower query
    // index). When false, two queries may both claim the same candidate;
    // it is then up to the caller to resolve. ORB-SLAM3's TrackLocalMap
    // (§4.7) wants unique candidates; raw Step 5 re-ID (§4.6) operates
    // on tiny candidate sets where conflicts are rare and the caller is
    // happy to handle them. Default false to keep the primitive minimal.
    bool unique_candidates = false;

    // Ambiguity (distinctiveness) gate, ORB-SLAM3-style. If > 0, a
    // query is only matched when its best passing candidate beats the
    // second-best SPATIALLY-GATED candidate by at least this many
    // Hamming bits:
    //     second_best - best >= second_best_margin
    // The second-best is taken over ALL spatial candidates, not only
    // threshold-passing ones: a competitor just above its threshold is
    // still evidence of ambiguity. A query with a single spatial
    // candidate passes trivially (nothing to be confused with). 0
    // disables the gate (legacy behaviour).
    //
    // This is the standard defence against descriptor aliasing on
    // self-similar texture (§4.6): when two nearby candidates look
    // equally good, refusing to match — a missed re-ID is recoverable,
    // a wrong one poisons a landmark — beats guessing.
    int second_best_margin = 0;
};

struct Match {
    std::size_t candidate_index = 0;
    int hamming_distance = 0;
};

// Compute Hamming distance between two 256-bit BRIEF descriptors.
// Uses popcount on 64-bit words. Exposed for tests and benchmarking.
int HammingDistance(const Descriptor256& a, const Descriptor256& b) noexcept;

// Match each query to its best candidate, subject to the gates in
// `opts`. Returns a vector of length `queries.size()`. Entry i is the
// match found for queries[i], or nullopt if nothing passed the gates.
//
// Complexity: O(Q * C) where Q = queries.size(), C = candidates.size().
// The expected use case has Q, C ≤ a few hundred per frame, so this is
// fine. A spatial index can be added later if profiling demands it.
std::vector<std::optional<Match>> SpatialDescriptorMatch(
    const std::vector<PixelQuery>& queries,
    const std::vector<PixelCandidate>& candidates,
    const MatchOptions& opts);

}  // namespace hybrid_frontend
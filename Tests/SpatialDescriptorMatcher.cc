// SpatialDescriptorMatcher.cc

#include "SpatialDescriptorMatcher.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace hybrid_frontend {

int HammingDistance(const Descriptor256& a, const Descriptor256& b) noexcept {
    // Treat the 32-byte descriptor as four 64-bit words. memcpy is the
    // standards-compliant way to do this — it lets the compiler emit an
    // aligned load on platforms that allow it and a safe unaligned one
    // elsewhere, with no UB. On x86-64 with -O2 this folds to four
    // movs + four popcnts.
    static_assert(sizeof(Descriptor256) == 32, "BRIEF descriptor must be 32 bytes");
    std::uint64_t w_a[4];
    std::uint64_t w_b[4];
    std::memcpy(w_a, a.data(), 32);
    std::memcpy(w_b, b.data(), 32);
    int dist = 0;
    for (int i = 0; i < 4; ++i) {
        dist += __builtin_popcountll(w_a[i] ^ w_b[i]);
    }
    return dist;
}

namespace {

// L-infinity (chebyshev) gate test. Inlined here so the compiler can
// hoist the radius compare ahead of the more expensive Hamming compute.
inline bool within_radius(float qx, float qy, float cx, float cy,
                          float radius) noexcept {
    return std::fabs(qx - cx) <= radius && std::fabs(qy - cy) <= radius;
}

}  // namespace

std::vector<std::optional<Match>> SpatialDescriptorMatch(
    const std::vector<PixelQuery>& queries,
    const std::vector<PixelCandidate>& candidates,
    const MatchOptions& opts) {

    const std::size_t Q = queries.size();
    const std::size_t C = candidates.size();
    std::vector<std::optional<Match>> out(Q);

    if (Q == 0 || C == 0) {
        return out;
    }

    // First pass: per-query best candidate (or none).
    // We loop queries on the outside so each query's radius is set once
    // and we early-exit on the cheap spatial test before computing
    // Hamming distance.
    for (std::size_t i = 0; i < Q; ++i) {
        const PixelQuery& q = queries[i];
        const float radius = (q.radius > 0.0f) ? q.radius : opts.default_radius;

        int best_dist = std::numeric_limits<int>::max();
        std::size_t best_idx = 0;
        bool found = false;

        for (std::size_t j = 0; j < C; ++j) {
            const PixelCandidate& c = candidates[j];
            if (!within_radius(q.x, q.y, c.x, c.y, radius)) {
                continue;
            }
            const int d = HammingDistance(q.descriptor, c.descriptor);
            if (d <= opts.hamming_threshold && d < best_dist) {
                best_dist = d;
                best_idx = j;
                found = true;
            }
        }
        if (found) {
            out[i] = Match{best_idx, best_dist};
        }
    }

    // Optional second pass: enforce unique candidates. For each
    // candidate claimed by more than one query, keep only the smallest
    // Hamming distance (ties broken by lower query index, which is the
    // natural fall-out of the iteration order below). The losers get
    // nullopt — they do NOT fall back to a second-choice candidate, to
    // keep the semantics simple and matchable in tests. If a caller
    // needs cascading fallback they can run multiple passes themselves.
    if (opts.unique_candidates) {
        // Build candidate_idx -> winning query_idx map. We iterate
        // queries in order so lower indices win ties as documented.
        std::vector<std::optional<std::size_t>> winner_of(C);
        std::vector<int>                        winner_dist(C, std::numeric_limits<int>::max());

        for (std::size_t i = 0; i < Q; ++i) {
            if (!out[i].has_value()) continue;
            const std::size_t j = out[i]->candidate_index;
            const int d = out[i]->hamming_distance;
            if (!winner_of[j].has_value() || d < winner_dist[j]) {
                // If a prior query was holding j, evict it.
                if (winner_of[j].has_value()) {
                    out[*winner_of[j]].reset();
                }
                winner_of[j] = i;
                winner_dist[j] = d;
            } else {
                // i loses to the previous holder of j.
                out[i].reset();
            }
        }
    }

    return out;
}

}  // namespace hybrid_frontend

// Short-term buffer of recently-died "infant" tracks (tracks that died
// before being triangulated to a map point), keyed for spatial query.
// Used by Step 5 of the hybrid frontend to re-identify a freshly spawned
// Shi-Tomasi corner as the resurrection of a track whose KLT update
// failed in a recent frame.
//
// Lifetime invariant (Design §4.1): the buffer stores ONLY infant tracks.
// A track in the buffer never holds a MapPoint reference. The MapPoint
// slot is retained in the API for forward compatibility with §7.4.2 and
// is asserted null on add().
//
// This module has no dependency on ORB-SLAM3 and is unit-testable in
// isolation. It does not depend on OpenCV either — pixel positions are
// passed as plain floats — so it compiles standalone.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace hybrid_frontend {

// 256-bit BRIEF descriptor, bit-compatible with ORB-SLAM3's cv::Mat
// row-of-uint8 layout (32 bytes per descriptor).
using Descriptor256 = std::array<uint8_t, 32>;

// Opaque map point pointer slot. The buffer never dereferences this; it
// is here only so the surrounding integration code can shuttle a pointer
// through re-ID without losing track identity. Per §4.1, the value
// stored in the buffer must always be null (asserted on add()).
using MapPointHandle = void*;

struct DormantTrack {
    std::uint64_t id = 0;        // globally unique track id (§4.1)
    float last_x = 0.0f;         // last observed pixel x in the parent frame
    float last_y = 0.0f;         // last observed pixel y
    Descriptor256 descriptor{};  // birth descriptor (single shot, §4.1)
    std::uint64_t frame_died = 0;  // frame index at which KLT failed
    MapPointHandle map_point = nullptr;  // always nullptr for infants (§4.1)

    // Pyramid level the feature was detected at. Carried through re-ID so
    // the resurrected track keeps its characteristic scale: BRIEF is
    // computed at kp.octave, TrackLocalMap's search radius is
    // r_TLM(l) = 2 * 1.2^l, and bundle adjustment weights the observation
    // by invLevelSigma2[octave]. A resurrected track that lost its octave
    // would be described and searched at the wrong scale.
    int octave = 0;

    // Track age (frames survived) at the moment of death, carried back
    // onto the resurrected track so an established landmark stays
    // "established" for age-gated logic downstream (§4.1). Without this,
    // resurrection silently demotes a long-lived track to a newborn and
    // any age threshold has to re-earn its confidence from zero.
    std::uint32_t age_at_death = 0;
};

class DormantTrackBuffer {
public:
    // dormant_horizon_frames corresponds to Δ_dormant in the design doc
    // (§4.1, §8): tracks older than `current_frame - dormant_horizon_frames`
    // are permanently dropped by purge_older_than().
    explicit DormantTrackBuffer(std::uint64_t dormant_horizon_frames);

    // Add a freshly-died infant track to the buffer.
    // PRECONDITION (asserted): track.map_point == nullptr (§4.1 invariant).
    // PRECONDITION (asserted): no entry with the same id is currently
    // present (§7.4.2: "The dormant buffer never contains a track whose
    // ID is also in the active set" — and a corollary, no duplicates in
    // the dormant set itself).
    void add(const DormantTrack& track);

    // Drop entries with frame_died + horizon < current_frame.
    // Idempotent. Safe to call every frame.
    void purge_older_than(std::uint64_t current_frame);

    // Return all entries within an L-infinity (chebyshev) ball of
    // `radius` pixels around (x, y). Entries are returned by value (copy)
    // so callers cannot accidentally invalidate them. The radius
    // corresponds to r_reid in the design doc (§4.6).
    //
    // L-infinity (square window) is used rather than L2 because it
    // matches the prediction-window convention elsewhere in ORB-SLAM3
    // (TrackLocalMap's W(u, r) is also square — see §4.7).
    std::vector<DormantTrack> query_within(float x, float y,
                                           float radius) const;

    // Remove the entry with the given id. No-op if not present.
    // Returns true iff an entry was removed.
    bool remove(std::uint64_t id);

    // Shift every entry's predicted position by (dx, dy).
    //
    // Motion compensation for §4.6: the re-ID spatial window is
    // "optionally propagated by the current motion model from the
    // dormant track's last position". Callers apply the dominant image
    // motion of the current frame (e.g. the median KLT flow of the
    // surviving inlier tracks) ONCE PER FRAME, after purge and before
    // any query_within, so that (last_x, last_y) becomes the *predicted*
    // pixel location of the dormant landmark in the current frame rather
    // than its stale location at death.
    //
    // Calling this more than once per frame double-counts the motion;
    // calling it never means r_reid must absorb the full accumulated
    // drift over the dormant horizon, which forces a radius wide enough
    // to invite descriptor aliasing. It is a translation only — no
    // rotation or scale — which is exactly right for the small
    // inter-frame motions the dormant horizon spans.
    //
    // (dx, dy) == (0, 0) is a no-op. Positions are NOT clamped to the
    // image: an entry pushed off-frame simply stops matching any query,
    // which is the correct outcome, and it will expire on schedule.
    void translate_all(float dx, float dy);

    // Frames elapsed since the given entry died — the gap g used by the
    // gap-scaled re-ID acceptance threshold in §4.6:
    //     theta_eff(g) = min(theta_cap, theta_base + slope * g)
    // A track that died g frames ago has accumulated ~g frames of
    // viewpoint-driven descriptor drift, so its acceptance ceiling is
    // relaxed in proportion. Returns 0 if current_frame precedes
    // frame_died (clock went backwards; treat as "just died").
    static std::uint64_t gap_frames(const DormantTrack& e,
                                    std::uint64_t current_frame) noexcept {
        return (current_frame > e.frame_died) ? (current_frame - e.frame_died)
                                              : 0u;
    }

    // Convenience for callers / tests.
    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    std::uint64_t horizon() const noexcept { return horizon_; }

    // Read-only view of every entry, for inspection and testing. Mirrors
    // the Python all_entries(). Returned by const reference — do not
    // hold it across a mutating call.
    const std::deque<DormantTrack>& all_entries() const noexcept {
        return entries_;
    }

    // Drop everything (used on relocalization — §6.4: "purge the dormant
    // buffer on relocalization").
    void clear() noexcept { entries_.clear(); }

private:
    std::uint64_t horizon_;
    // deque rather than vector: purge_older_than() pops from the front,
    // add() pushes to the back. With monotonic frame numbers the buffer
    // is naturally sorted by frame_died and purge is O(k) where k is the
    // number of expiring entries. Average steady-state O(1) amortized.
    //
    // Note: remove() by id is O(n), but n is bounded by Δ_dormant times
    // average new-track-spawn rate, which is small (typically <= a few
    // hundred). For larger n a secondary id→iterator hash index could be
    // bolted on; the spec calls for an O(n) buffer and that is what this is.
    std::deque<DormantTrack> entries_;
};

}  // namespace hybrid_frontend
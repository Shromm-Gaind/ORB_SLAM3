// DormantTrackBuffer.h
//
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

    // Convenience for callers / tests.
    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    std::uint64_t horizon() const noexcept { return horizon_; }

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

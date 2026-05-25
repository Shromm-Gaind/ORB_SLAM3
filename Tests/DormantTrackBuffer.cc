// DormantTrackBuffer.cc

#include "DormantTrackBuffer.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace hybrid_frontend {

DormantTrackBuffer::DormantTrackBuffer(std::uint64_t dormant_horizon_frames)
    : horizon_(dormant_horizon_frames) {}

void DormantTrackBuffer::add(const DormantTrack& track) {
    // §4.1 invariant: dormant tracks are always infants (no map point).
    assert(track.map_point == nullptr &&
           "DormantTrackBuffer holds infant tracks only; map_point must be null");
    // §7.4.2 invariant: no duplicate ids in the buffer.
    assert(std::none_of(entries_.begin(), entries_.end(),
                        [&](const DormantTrack& e) { return e.id == track.id; }) &&
           "Duplicate track id in dormant buffer");
    entries_.push_back(track);
}

void DormantTrackBuffer::purge_older_than(std::uint64_t current_frame) {
    // An entry is expired when `current_frame - frame_died > horizon_`.
    // Frame numbers are monotonic, so frame_died is non-decreasing within
    // the deque if entries are added in death order. We do NOT rely on
    // that strict ordering, however — callers may not perfectly serialize
    // adds, and a frame's KLT failures can be processed in any internal
    // order. We therefore scan the whole deque.
    //
    // Underflow guard: if current_frame < horizon_ (early frames), nothing
    // can be expired yet.
    if (current_frame <= horizon_) {
        return;
    }
    const std::uint64_t cutoff = current_frame - horizon_;
    // erase-remove idiom on a deque: ok because deque supports random
    // iterators and erase() in the middle is allowed (O(n) but acceptable).
    entries_.erase(
        std::remove_if(entries_.begin(), entries_.end(),
                       [cutoff](const DormantTrack& e) {
                           return e.frame_died < cutoff;
                       }),
        entries_.end());
}

std::vector<DormantTrack> DormantTrackBuffer::query_within(float x, float y,
                                                           float radius) const {
    std::vector<DormantTrack> hits;
    // Linear scan. n is bounded by horizon * spawn_rate (~tens to low
    // hundreds in practice). A square (L-infinity) window matches
    // TrackLocalMap's convention (§4.7).
    //
    // Negative radius is treated as zero — be tolerant of caller error
    // rather than crash. A zero radius matches only entries at exactly
    // (x, y), which is fine for testing degenerate cases.
    const float r = radius < 0.0f ? 0.0f : radius;
    for (const auto& e : entries_) {
        if (std::fabs(e.last_x - x) <= r && std::fabs(e.last_y - y) <= r) {
            hits.push_back(e);
        }
    }
    return hits;
}

bool DormantTrackBuffer::remove(std::uint64_t id) {
    // Linear scan + erase. Acceptable for small n.
    auto it = std::find_if(entries_.begin(), entries_.end(),
                           [id](const DormantTrack& e) { return e.id == id; });
    if (it == entries_.end()) {
        return false;
    }
    entries_.erase(it);
    return true;
}

}  // namespace hybrid_frontend

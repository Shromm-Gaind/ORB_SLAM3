/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of
 * Zaragoza.
 *
 * Modifications Copyright (C) 2026 Shromm Gaind. Work done as a honours student at Griffith University
 * Modified 2026: hybrid KLT/Shi-Tomasi front end backend is left untouched.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */


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

void DormantTrackBuffer::translate_all(float dx, float dy) {
    // Exact no-op guard, matching the Python reference. Worth keeping
    // even though the loop below would be a no-op anyway: a stationary
    // frame is the common case in a hovering AUV and this skips the
    // whole pass.
    if (dx == 0.0f && dy == 0.0f) {
        return;
    }
    for (auto& e : entries_) {
        e.last_x += dx;
        e.last_y += dy;
    }
}

}  // namespace hybrid_frontend
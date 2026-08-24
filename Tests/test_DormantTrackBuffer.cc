// test_DormantTrackBuffer.cc
//
// Unit tests for DormantTrackBuffer. Covers §4.1 / §7.4.2 invariants
// and edge cases (empty, boundary radii, frame underflow, etc.).

#include "DormantTrackBuffer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

using hybrid_frontend::Descriptor256;
using hybrid_frontend::DormantTrack;
using hybrid_frontend::DormantTrackBuffer;

namespace {

DormantTrack make_track(std::uint64_t id, float x, float y,
                        std::uint64_t frame_died, std::uint8_t fill = 0) {
    DormantTrack t;
    t.id = id;
    t.last_x = x;
    t.last_y = y;
    t.frame_died = frame_died;
    t.descriptor.fill(fill);
    t.map_point = nullptr;
    return t;
}

}  // namespace

// ---- basic state -----------------------------------------------------

TEST(DormantTrackBuffer, EmptyBufferState) {
    DormantTrackBuffer buf(30);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.horizon(), 30u);

    auto hits = buf.query_within(100.0f, 100.0f, 50.0f);
    EXPECT_TRUE(hits.empty());
}

TEST(DormantTrackBuffer, AddIncrementsSize) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 50.0f, 50.0f, 0));
    EXPECT_EQ(buf.size(), 1u);
    EXPECT_FALSE(buf.empty());

    buf.add(make_track(2, 80.0f, 80.0f, 0));
    EXPECT_EQ(buf.size(), 2u);
}

// ---- spatial query --------------------------------------------------

TEST(DormantTrackBuffer, QueryWithinRadiusHits) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));

    auto hits = buf.query_within(100.0f, 100.0f, 5.0f);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 1u);
}

TEST(DormantTrackBuffer, QueryOutsideRadiusMisses) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));

    auto hits = buf.query_within(200.0f, 200.0f, 5.0f);
    EXPECT_TRUE(hits.empty());
}

TEST(DormantTrackBuffer, QueryLInfinityShape) {
    // L-infinity (square window) means a point at (100+r, 100) is
    // included; a point at (100+r+epsilon, 100) is not. A diagonal
    // point at (100+r, 100+r) IS included (L-inf accepts it; L2 would
    // reject because the L2 distance is r*sqrt(2)). This is the
    // documented behaviour.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 110.0f, 100.0f, 0));  // exactly at right edge
    buf.add(make_track(2, 100.0f, 110.0f, 0));  // exactly at bottom edge
    buf.add(make_track(3, 110.0f, 110.0f, 0));  // corner — L2 ~ 14.14, L-inf 10
    buf.add(make_track(4, 111.0f, 100.0f, 0));  // just outside
    buf.add(make_track(5, 100.0f, 111.0f, 0));  // just outside

    auto hits = buf.query_within(100.0f, 100.0f, 10.0f);

    std::vector<std::uint64_t> ids;
    for (const auto& h : hits) ids.push_back(h.id);
    std::sort(ids.begin(), ids.end());

    EXPECT_EQ(ids, (std::vector<std::uint64_t>{1, 2, 3}));
}

TEST(DormantTrackBuffer, QueryNegativeRadiusTreatedAsZero) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    // Negative radius → treated as 0 → only exact-match locations hit
    auto hits = buf.query_within(100.0f, 100.0f, -5.0f);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 1u);

    auto miss = buf.query_within(101.0f, 100.0f, -5.0f);
    EXPECT_TRUE(miss.empty());
}

TEST(DormantTrackBuffer, QueryReturnsMultipleHits) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    buf.add(make_track(2, 105.0f, 105.0f, 0));
    buf.add(make_track(3, 95.0f, 95.0f, 0));
    buf.add(make_track(4, 500.0f, 500.0f, 0));  // far away

    auto hits = buf.query_within(100.0f, 100.0f, 10.0f);
    EXPECT_EQ(hits.size(), 3u);  // not the far one
}

// ---- purge / horizon ------------------------------------------------

TEST(DormantTrackBuffer, PurgeKeepsFreshEntries) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, /*frame_died=*/100));

    // current frame = 110; 110 - 100 = 10 <= 30, still fresh
    buf.purge_older_than(110);
    EXPECT_EQ(buf.size(), 1u);
}

TEST(DormantTrackBuffer, PurgeBoundaryStillKeepsAtHorizonExactly) {
    // The contract: an entry expires when frame_died < current_frame - horizon.
    // So if frame_died == current_frame - horizon, the entry is still fresh.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, /*frame_died=*/100));

    buf.purge_older_than(130);  // 130 - 30 = 100 == frame_died → not <  → kept
    EXPECT_EQ(buf.size(), 1u);
}

TEST(DormantTrackBuffer, PurgeOneBeyondHorizonDrops) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, /*frame_died=*/100));

    buf.purge_older_than(131);  // 131 - 30 = 101 > 100 → drop
    EXPECT_EQ(buf.size(), 0u);
}

TEST(DormantTrackBuffer, PurgePartial) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, /*frame_died=*/100));
    buf.add(make_track(2, 0.0f, 0.0f, /*frame_died=*/120));
    buf.add(make_track(3, 0.0f, 0.0f, /*frame_died=*/140));

    // cutoff = 145 - 30 = 115. Entry 1 (died at 100) drops. 2 and 3 stay.
    buf.purge_older_than(145);
    EXPECT_EQ(buf.size(), 2u);

    auto hits = buf.query_within(0.0f, 0.0f, 1.0f);
    std::vector<std::uint64_t> ids;
    for (const auto& h : hits) ids.push_back(h.id);
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<std::uint64_t>{2, 3}));
}

TEST(DormantTrackBuffer, PurgeUnderflowSafe) {
    // current_frame=5, horizon=30 → would underflow. Must be safe.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, /*frame_died=*/2));
    buf.purge_older_than(5);  // 5 - 30 would underflow if done with subtraction
    EXPECT_EQ(buf.size(), 1u);  // nothing should have been dropped
}

TEST(DormantTrackBuffer, PurgeIsIdempotent) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, /*frame_died=*/0));
    buf.add(make_track(2, 0.0f, 0.0f, /*frame_died=*/20));

    // Advance to frame 51: cutoff = 51-30 = 21, so both (died at 0 and
    // 20) satisfy frame_died < cutoff and are dropped.
    buf.purge_older_than(51);
    EXPECT_EQ(buf.size(), 0u);
    buf.purge_older_than(51);  // still 0 — repeated call should be a no-op
    EXPECT_EQ(buf.size(), 0u);
    buf.purge_older_than(100);
    EXPECT_EQ(buf.size(), 0u);
}

// ---- remove ---------------------------------------------------------

TEST(DormantTrackBuffer, RemoveExistingReturnsTrue) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, 0));
    buf.add(make_track(2, 0.0f, 0.0f, 0));

    EXPECT_TRUE(buf.remove(1));
    EXPECT_EQ(buf.size(), 1u);

    auto hits = buf.query_within(0.0f, 0.0f, 1.0f);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 2u);
}

TEST(DormantTrackBuffer, RemoveNonexistentReturnsFalse) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, 0));

    EXPECT_FALSE(buf.remove(999));
    EXPECT_EQ(buf.size(), 1u);
}

// ---- clear ----------------------------------------------------------

TEST(DormantTrackBuffer, ClearEmpties) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, 0));
    buf.add(make_track(2, 0.0f, 0.0f, 0));
    buf.clear();
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
}

// ---- translate_all: motion compensation (§4.6) -----------------------

TEST(DormantTrackBuffer, TranslateAllShiftsEveryEntry) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    buf.add(make_track(2, 200.0f, 50.0f, 0));

    buf.translate_all(5.0f, -3.0f);

    // Query at the ORIGINAL location now misses with a tight radius...
    auto stale = buf.query_within(100.0f, 100.0f, 1.0f);
    EXPECT_TRUE(stale.empty());
    // ...and the PREDICTED location hits.
    auto moved = buf.query_within(105.0f, 97.0f, 1.0f);
    ASSERT_EQ(moved.size(), 1u);
    EXPECT_EQ(moved[0].id, 1u);

    auto moved2 = buf.query_within(205.0f, 47.0f, 1.0f);
    ASSERT_EQ(moved2.size(), 1u);
    EXPECT_EQ(moved2[0].id, 2u);
}

TEST(DormantTrackBuffer, TranslateAllZeroIsNoOp) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    buf.translate_all(0.0f, 0.0f);
    auto hits = buf.query_within(100.0f, 100.0f, 0.0f);
    ASSERT_EQ(hits.size(), 1u);
}

TEST(DormantTrackBuffer, TranslateAllAccumulates) {
    // Two frames of motion compensation compose additively. This is the
    // contract that makes "call it exactly once per frame" correct — and
    // the reason calling it twice in one frame double-counts.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    buf.translate_all(4.0f, 0.0f);
    buf.translate_all(3.0f, 0.0f);
    auto hits = buf.query_within(107.0f, 100.0f, 0.0f);
    EXPECT_EQ(hits.size(), 1u);
}

TEST(DormantTrackBuffer, TranslateAllOnEmptyBufferIsSafe) {
    DormantTrackBuffer buf(30);
    buf.translate_all(10.0f, 10.0f);
    EXPECT_TRUE(buf.empty());
}

TEST(DormantTrackBuffer, TranslateAllAppliesToLaterAddsOnlyOnce) {
    // An entry added AFTER this frame's translate_all must not receive
    // that frame's shift — it was born at its true current position.
    // This pins the intended per-frame ordering: purge -> translate ->
    // query/re-ID -> add this frame's newly-dead tracks.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 100.0f, 100.0f, 0));
    buf.translate_all(10.0f, 0.0f);      // frame N motion
    buf.add(make_track(2, 100.0f, 100.0f, 1));  // died at frame N, true position

    auto old_entry = buf.query_within(110.0f, 100.0f, 0.0f);
    ASSERT_EQ(old_entry.size(), 1u);
    EXPECT_EQ(old_entry[0].id, 1u);

    auto new_entry = buf.query_within(100.0f, 100.0f, 0.0f);
    ASSERT_EQ(new_entry.size(), 1u);
    EXPECT_EQ(new_entry[0].id, 2u);
}

TEST(DormantTrackBuffer, TranslateAllMayPushEntryOffFrame) {
    // Documented behaviour: positions are not clamped. An entry driven
    // off-image simply stops matching; it still expires on schedule.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 10.0f, 10.0f, 0));
    buf.translate_all(-500.0f, 0.0f);
    EXPECT_EQ(buf.size(), 1u);
    auto hits = buf.query_within(10.0f, 10.0f, 20.0f);
    EXPECT_TRUE(hits.empty());
}

// ---- gap_frames: input to the gap-scaled threshold (§4.6) ------------

TEST(DormantTrackBuffer, GapFramesBasic) {
    auto t = make_track(1, 0.0f, 0.0f, /*frame_died=*/100);
    EXPECT_EQ(DormantTrackBuffer::gap_frames(t, 100), 0u);
    EXPECT_EQ(DormantTrackBuffer::gap_frames(t, 101), 1u);
    EXPECT_EQ(DormantTrackBuffer::gap_frames(t, 130), 30u);
}

TEST(DormantTrackBuffer, GapFramesUnderflowSafe) {
    // current_frame before frame_died must not wrap around on unsigned
    // arithmetic — a 1.8e19 gap would blow past any theta_cap.
    auto t = make_track(1, 0.0f, 0.0f, /*frame_died=*/100);
    EXPECT_EQ(DormantTrackBuffer::gap_frames(t, 50), 0u);
}

// ---- octave and age_at_death carry-through ---------------------------

TEST(DormantTrackBuffer, OctaveAndAgeSurviveRoundTrip) {
    DormantTrackBuffer buf(30);
    DormantTrack t = make_track(7, 100.0f, 100.0f, 42);
    t.octave = 3;
    t.age_at_death = 250;
    buf.add(t);

    auto hits = buf.query_within(100.0f, 100.0f, 1.0f);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].octave, 3);
    EXPECT_EQ(hits[0].age_at_death, 250u);
    EXPECT_EQ(hits[0].frame_died, 42u);
}

TEST(DormantTrackBuffer, DefaultsAreInfantLike) {
    // A track built without setting the new fields must look like a
    // level-0 newborn, so existing call sites keep their old meaning.
    DormantTrack t;
    EXPECT_EQ(t.octave, 0);
    EXPECT_EQ(t.age_at_death, 0u);
    EXPECT_EQ(t.map_point, nullptr);
}

// ---- all_entries -----------------------------------------------------

TEST(DormantTrackBuffer, AllEntriesReflectsContents) {
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, 0));
    buf.add(make_track(2, 0.0f, 0.0f, 0));

    const auto& all = buf.all_entries();
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(all[0].id, 1u);
    EXPECT_EQ(all[1].id, 2u);

    buf.remove(1);
    EXPECT_EQ(buf.all_entries().size(), 1u);
    EXPECT_EQ(buf.all_entries()[0].id, 2u);
}

// ---- §4.1 / §7.4.2 invariants (debug-only asserts) ------------------

#ifndef NDEBUG
TEST(DormantTrackBufferDeathTest, AddingWithMapPointAborts) {
    // §4.1: "dormant track buffer holds infant tracks only"; map_point
    // must be null.
    DormantTrackBuffer buf(30);
    DormantTrack t = make_track(1, 0.0f, 0.0f, 0);
    int dummy = 0;
    t.map_point = static_cast<void*>(&dummy);
    // GTest death-tests run in a forked subprocess so the assert() abort
    // doesn't kill the test binary.
    EXPECT_DEATH({ buf.add(t); }, "map_point must be null");
}

TEST(DormantTrackBufferDeathTest, AddingDuplicateIdAborts) {
    // §7.4.2 invariant: no duplicate ids in the buffer.
    DormantTrackBuffer buf(30);
    buf.add(make_track(1, 0.0f, 0.0f, 0));
    EXPECT_DEATH({ buf.add(make_track(1, 10.0f, 10.0f, 5)); },
                 "Duplicate track id");
}
#endif  // !NDEBUG
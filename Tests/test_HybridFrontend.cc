#include "HybridFrontend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include <opencv2/imgproc.hpp>

using namespace hybrid_frontend;

namespace {

// Textured noise: uniform random uint8, lightly blurred so KLT windows
// have gradients everywhere (mirrors make_textured in the Python).
cv::Mat make_textured(int h = 480, int w = 640, uint64 seed = 7) {
    cv::Mat img(h, w, CV_8U);
    cv::RNG rng(seed);
    rng.fill(img, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

// Integer shift with zero padding (content moves by (+dx, +dy)).
cv::Mat shift_image(const cv::Mat& src, int dx, int dy) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    const int w = src.cols, h = src.rows;
    const int sx0 = std::max(0, -dx), sy0 = std::max(0, -dy);
    const int dx0 = std::max(0, dx), dy0 = std::max(0, dy);
    const int cw = w - std::abs(dx), ch = h - std::abs(dy);
    if (cw <= 0 || ch <= 0) return dst;
    src(cv::Rect(sx0, sy0, cw, ch)).copyTo(dst(cv::Rect(dx0, dy0, cw, ch)));
    return dst;
}

HybridConfig test_config() {
    HybridConfig c;
    c.target_active_tracks = 100;
    c.shi_tomasi_quality = 0.01;
    c.shi_tomasi_min_distance = 10;
    c.dormant_horizon_frames = 30;
    c.reid_radius_px = 15.0f;
    c.reid_hamming_threshold = 80;
    return c;
}

std::vector<std::uint64_t> active_ids(const HybridFrontend& hf) {
    std::vector<std::uint64_t> ids;
    for (const auto& kv : hf.active_tracks()) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TEST(HybridFrontend, InitializePopulatesTracks) {
    HybridFrontend hf(test_config());
    hf.initialize(make_textured());
    const auto& tracks = hf.active_tracks();
    EXPECT_GT(tracks.size(), 0u);
    EXPECT_LE(static_cast<int>(tracks.size()),
              hf.config().target_active_tracks);
    for (const auto& kv : tracks) {
        EXPECT_EQ(kv.second.age, 0u);
        EXPECT_EQ(kv.second.map_point, nullptr);
    }
    EXPECT_EQ(hf.frame_index(), 0u);
    EXPECT_TRUE(hf.dormant_buffer().empty());
}

TEST(HybridFrontend, InitializedTracksAreSpreadOut) {
    // Quadtree sanity: on uniformly textured input the selected corners
    // must span most of the image, not clump in one quadrant.
    HybridFrontend hf(test_config());
    hf.initialize(make_textured());
    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
    for (const auto& kv : hf.active_tracks()) {
        minx = std::min(minx, kv.second.x);
        maxx = std::max(maxx, kv.second.x);
        miny = std::min(miny, kv.second.y);
        maxy = std::max(maxy, kv.second.y);
    }
    EXPECT_GT(maxx - minx, 320.0f);   // > half the width
    EXPECT_GT(maxy - miny, 240.0f);   // > half the height
}

TEST(HybridFrontend, TrackIdsPersistAcrossEasyFrames) {
    HybridFrontend hf(test_config());
    cv::Mat base = make_textured();
    hf.initialize(base);
    const auto ids0 = active_ids(hf);
    ASSERT_GT(ids0.size(), 10u);

    FrameResult last{};
    for (int k = 1; k <= 3; ++k)
        last = hf.process_frame(shift_image(base, k, 0));

    // > 50% of the original ids must still be active (a 1 px/frame
    // shift on rich texture is as easy as KLT gets).
    const auto ids3 = active_ids(hf);
    std::size_t survived = 0;
    for (std::uint64_t id : ids0)
        if (std::binary_search(ids3.begin(), ids3.end(), id)) ++survived;
    EXPECT_GT(survived, ids0.size() / 2);
    EXPECT_LE(last.tracks_out, hf.config().target_active_tracks);
    // Frame counters are coherent.
    EXPECT_GE(last.tracks_in, last.tracks_after_klt);
    EXPECT_GE(last.tracks_after_klt, last.tracks_after_fb);
    EXPECT_GE(last.tracks_after_fb, last.tracks_after_ransac);
}

TEST(HybridFrontend, MedianFlowMatchesKnownShift) {
    HybridFrontend hf(test_config());
    cv::Mat base = make_textured();
    hf.initialize(base);
    const FrameResult r = hf.process_frame(shift_image(base, 1, 0));
    EXPECT_NEAR(r.median_flow_dx, 1.0f, 0.3f);
    EXPECT_NEAR(r.median_flow_dy, 0.0f, 0.3f);
}

TEST(HybridFrontend, ForceKillMovesTracksToDormant) {
    HybridFrontend hf(test_config());
    cv::Mat base = make_textured();
    hf.initialize(base);
    hf.process_frame(shift_image(base, 1, 0));

    auto ids = active_ids(hf);
    ASSERT_GE(ids.size(), 10u);
    std::vector<std::uint64_t> victims(ids.begin(), ids.begin() + 10);
    const std::size_t active_before = hf.active_tracks().size();
    const std::size_t dormant_before = hf.dormant_buffer().size();

    auto killed = hf.force_kill(victims);
    EXPECT_EQ(killed.size(), 10u);
    EXPECT_EQ(hf.active_tracks().size(), active_before - 10);
    EXPECT_EQ(hf.dormant_buffer().size(), dormant_before + 10);
    for (const auto& k : killed)
        EXPECT_EQ(hf.active_tracks().count(k.id), 0u);
}

TEST(HybridFrontend, ReidResurrectsAfterForceKill) {
    HybridFrontend hf(test_config());
    cv::Mat base = make_textured();
    hf.initialize(base);
    hf.process_frame(shift_image(base, 1, 0));

    // Kill up to 20 established tracks, then advance one frame: their
    // corners are still present (shifted by 1 px), so Step 4/5 should
    // re-detect and resurrect a nonzero number of them.
    auto ids = active_ids(hf);
    std::vector<std::uint64_t> victims(
        ids.begin(), ids.begin() + std::min<std::size_t>(20, ids.size()));
    hf.force_kill(victims);

    const FrameResult r = hf.process_frame(shift_image(base, 2, 0));
    std::set<std::uint64_t> vset(victims.begin(), victims.end());
    std::size_t overlap = 0;
    for (std::uint64_t id : r.resurrected_ids)
        if (vset.count(id)) ++overlap;
    EXPECT_GT(overlap, 0u);
    EXPECT_EQ(r.resurrected_ids.size(),
              static_cast<std::size_t>(r.reids_succeeded));
    // A resurrected track carries its pre-death age (nonzero: victims
    // survived one processed frame before the kill).
    if (!r.resurrected_ids.empty()) {
        const auto& t = hf.active_tracks().at(r.resurrected_ids.front());
        EXPECT_GT(t.age, 0u);
    }
}

TEST(HybridFrontend, DormantBufferStaysBounded) {
    HybridFrontend hf(test_config());
    cv::Mat base = make_textured();
    hf.initialize(base);
    for (int k = 1; k <= 9; ++k) {
        const FrameResult r = hf.process_frame(shift_image(base, k, 0));
        EXPECT_LT(r.dormant_buffer_size,
                  static_cast<std::size_t>(
                      hf.config().target_active_tracks) / 2);
        EXPECT_LE(r.tracks_out, hf.config().target_active_tracks);
    }
}

TEST(HybridFrontend, RepresentativeDescriptorAccumulates) {
    HybridConfig c = test_config();
    c.representative_sample_stride = 1;   // sample every frame
    c.representative_max_observations = 4;
    HybridFrontend hf(c);
    cv::Mat base = make_textured();
    hf.initialize(base);
    for (int k = 1; k <= 6; ++k)
        hf.process_frame(shift_image(base, k, 0));

    bool any_multi = false;
    for (const auto& kv : hf.active_tracks()) {
        const ActiveTrack& t = kv.second;
        ASSERT_TRUE(t.has_representative);
        ASSERT_GE(t.descriptor_history.size(), 1u);
        EXPECT_LE(static_cast<int>(t.descriptor_history.size()),
                  c.representative_max_observations);
        if (t.descriptor_history.size() > 1) any_multi = true;
        // The representative is one of the observations (a medoid, not
        // an average).
        bool found = false;
        for (const auto& d : t.descriptor_history)
            if (d == t.representative_descriptor) { found = true; break; }
        EXPECT_TRUE(found);
    }
    EXPECT_TRUE(any_multi);
}

TEST(HybridFrontend, RepresentativeDisabledLeavesHistoryEmpty) {
    HybridConfig c = test_config();
    c.use_representative_descriptor = false;
    HybridFrontend hf(c);
    cv::Mat base = make_textured();
    hf.initialize(base);
    hf.process_frame(shift_image(base, 1, 0));
    for (const auto& kv : hf.active_tracks()) {
        EXPECT_TRUE(kv.second.descriptor_history.empty());
        EXPECT_FALSE(kv.second.has_representative);
    }
}

TEST(HybridFrontend, LocalDetectNeverInflatesActiveSet) {
    HybridConfig c = test_config();
    c.local_detect_in_dormant_windows = true;
    HybridFrontend hf(c);
    cv::Mat base = make_textured();
    hf.initialize(base);
    hf.process_frame(shift_image(base, 1, 0));
    // Populate the dormant buffer heavily, then step several frames:
    // unmatched local-only corners must be discarded, so the active set
    // can never exceed N_target.
    auto ids = active_ids(hf);
    std::vector<std::uint64_t> victims(
        ids.begin(), ids.begin() + std::min<std::size_t>(40, ids.size()));
    hf.force_kill(victims);
    for (int k = 2; k <= 5; ++k) {
        const FrameResult r = hf.process_frame(shift_image(base, k, 0));
        EXPECT_LE(r.tracks_out, c.target_active_tracks);
    }
}

TEST(HybridFrontend, ReidDisabledStillMaintainsBuffer) {
    HybridConfig c = test_config();
    c.enable_reid = false;
    HybridFrontend hf(c);
    cv::Mat base = make_textured();
    hf.initialize(base);
    hf.process_frame(shift_image(base, 1, 0));
    auto ids = active_ids(hf);
    std::vector<std::uint64_t> victims(
        ids.begin(), ids.begin() + std::min<std::size_t>(10, ids.size()));
    hf.force_kill(victims);
    const std::size_t dormant_after_kill = hf.dormant_buffer().size();
    EXPECT_GE(dormant_after_kill, victims.size());

    const FrameResult r = hf.process_frame(shift_image(base, 2, 0));
    // Ablation semantics: buffer maintained, never queried.
    EXPECT_EQ(r.reids_attempted, 0);
    EXPECT_EQ(r.reids_succeeded, 0);
    EXPECT_TRUE(r.resurrected_ids.empty());
}

TEST(HybridFrontend, SetMapPointAttachesHandle) {
    HybridFrontend hf(test_config());
    hf.initialize(make_textured());
    auto ids = active_ids(hf);
    ASSERT_FALSE(ids.empty());
    int dummy = 42;
    EXPECT_TRUE(hf.set_map_point(ids.front(), &dummy));
    EXPECT_EQ(hf.active_tracks().at(ids.front()).map_point, &dummy);
    EXPECT_FALSE(hf.set_map_point(999999u, &dummy));
}
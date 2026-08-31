// C++ port of hybrid_frontend.py: the stateful per-frame pipeline
// implementing Steps 1-5 of the hybrid frontend (design doc §4).
//
//   Step 1  KLT forward tracking of active features
//   Step 2  forward-backward consistency filter
//   Step 3  RANSAC fundamental-matrix outlier rejection
//   Step 4  Shi-Tomasi top-up to N_target (occupancy mask + quadtree,
//           with dormant seeding), steered BRIEF at the new corners
//   Step 5  re-ID of new corners against the DormantTrackBuffer via
//           SpatialDescriptorMatch (gap-scaled θ_reid, distinctiveness
//           margin, unique candidates)
//
// Dependencies: OpenCV (core, imgproc, video, features2d, calib3d) and
// the two sibling modules. NO ORB-SLAM3 dependency — this class is the
// engine; Tracking.cc integration is a thin caller (see
// TRACKING_INTEGRATION.md).
//
// Faithful-port notes: semantics follow hybrid_frontend.py exactly,
// including step ordering (purge → KLT/FB/RANSAC → motion-compensate
// dormant → deaths buffered flow-adjusted → top-up → re-ID under a
// deficit budget with resurrections first). Python's diagnostics-only
// instrumentation (per-query candidate counts, drift histograms) is not
// ported; per-frame counts and events are.

#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DormantTrackBuffer.h"
#include "SpatialDescriptorMatcher.h"

namespace hybrid_frontend {

struct HybridConfig {
    // Step 1 — KLT
    cv::Size klt_window{21, 21};
    int klt_pyramid_levels = 3;
    int klt_max_iter = 30;
    double klt_eps = 0.01;

    // Step 2 — forward-backward consistency
    float fb_threshold_px = 1.0f;

    // Step 3 — RANSAC fundamental matrix
    double ransac_reproj_px = 3.0;
    int min_matches_for_F = 8;

    // Step 4 — Shi-Tomasi top-up
    int target_active_tracks = 1000;     // N_target
    double shi_tomasi_quality = 0.01;    // θ_ST proxy (OpenCV scale)
    int shi_tomasi_min_distance = 7;
    int shi_tomasi_block_size = 7;
    int occupancy_mask_radius = 10;      // r_mask
    int descriptor_patch_size = 31;      // steered BRIEF patch
    // Descriptor pyramid (§8 s_p, n_lev). MUST match the ORBextractor's
    // scaleFactor/nLevels: once Stage B lands, dormant descriptors are
    // compared against extractor descriptors, and distances between
    // differently-built pyramids are meaningless.
    float descriptor_scale_factor = 1.2f;   // s_p
    int descriptor_levels = 8;              // n_lev
    // Multi-octave detection (multiscale_shitomasi.py, §4.5.2): detect
    // per pyramid level with the RANK rule (per-level geometric target
    // + quadtree, low noise floor only), stamp kp.octave, describe at
    // the detected scale. false = PoC-faithful level-0-only detection
    // (the single-scale ablation). Detection reuses
    // descriptor_scale_factor / descriptor_levels so the detection
    // pyramid and cv::ORB's description pyramid can never disagree.
    bool multiscale_detection = true;
    // Amortisation. MINIMUM DEFICIT, as a fraction of N_target, that
    // must accumulate before Step 4/5 runs at all. Detection is a fixed
    // ~3x single-scale pyramid cost regardless of how few corners are
    // wanted, so topping up 130 corners costs the same as topping up
    // 900; letting a deficit accumulate amortises that.
    //   0.00 -> detect whenever there is ANY deficit (default; this is
    //           the measured, PoC-faithful behaviour)
    //   0.10 -> detect once active has sagged to 90% of N_target,
    //           i.e. roughly every 2-3 frames in steady state
    // The value is the DEFICIT fraction, not the retained fraction:
    // 0.90 would mean "wait until 90% of tracks are gone", which is
    // almost certainly not what you want.
    // NOTE: skipped frames also skip re-ID. With dormant_horizon_frames
    // = 10 an entry still gets several chances, but set this to 0 for
    // any run whose re-ID recall you intend to report.
    double min_deficit_fraction = 0.0;

    // Step 5 — re-ID
    // §8 REVISED values. θ_eff(g) = min(θ_cap, θ_base + β·g); note that
    // with the horizon at 10 the cap is unreachable (saturation needs
    // g≈19), so the effective ceiling tops out at 75 bits and θ_cap is
    // no longer a live tuning knob — β is.
    int dormant_horizon_frames = 10;     // Δ_dormant (was 30, §12.1)
    float reid_radius_px = 10.0f;        // r_reid    (was 20, §4.6)
    int reid_hamming_threshold = 63;     // θ_base    (was 32, §12.2 p95)
    double reid_hamming_slope_per_frame = 1.2;  // β  (was 1.0)
    int reid_hamming_cap = 86;           // θ_cap     (was 55)
    int reid_second_best_margin = 0;     // δ, 0=off  (was 10, §4.6.1)
    bool motion_compensate_dormant = true;
    int dormant_min_track_age = 3;       // natural deaths younger than
                                         // this are retired, not buffered
    bool seed_corners_near_dormant = true;
    bool enable_reid = true;

    // Representative descriptor (ORB-SLAM3 ComputeDistinctiveDescriptors)
    bool use_representative_descriptor = true;
    int representative_max_observations = 8;
    int representative_sample_stride = 3;

    // Targeted local detection inside dormant windows
    bool local_detect_in_dormant_windows = true;
    double local_detect_quality_scale = 0.3;
    int local_detect_max_windows = 400;

    // Tracking-lost threshold (§8 N_min). Carried for the caller;
    // process_frame itself does not act on it (matches the Python).
    int min_active_tracks = 50;
};

struct ActiveTrack {
    std::uint64_t id = 0;
    float x = 0.0f;
    float y = 0.0f;
    Descriptor256 birth_descriptor{};
    int octave = 0;
    std::uint32_t age = 0;
    MapPointHandle map_point = nullptr;   // owned by the caller (Tracking)
    // Sampled observations over the track's life (index 0 = birth), and
    // the medoid computed from them.
    std::vector<Descriptor256> descriptor_history;
    bool has_representative = false;
    Descriptor256 representative_descriptor{};
};

struct DeathRecord {
    std::uint64_t id;
    float last_x;
    float last_y;
};

struct FrameResult {
    std::uint64_t frame_index = 0;
    int tracks_in = 0;
    int tracks_after_klt = 0;
    int tracks_after_fb = 0;
    int tracks_after_ransac = 0;
    int new_corners_detected = 0;
    int reids_attempted = 0;
    int reids_succeeded = 0;
    int tracks_out = 0;
    std::size_t dormant_buffer_size = 0;
    std::vector<DeathRecord> died_this_frame;
    std::vector<std::uint64_t> resurrected_ids;
    // Median (dx, dy) of RANSAC-inlier KLT flow; (0,0) if none.
    float median_flow_dx = 0.0f;
    float median_flow_dy = 0.0f;
    // Per-stage wall time, milliseconds. ms_total covers process_frame.
    double ms_track = 0.0;    // Steps 1-3 + survivor/death bookkeeping
    double ms_detect = 0.0;   // Step 4 detection + BRIEF
    double ms_reid = 0.0;     // Step 5 queries, matching, spawning
    double ms_total = 0.0;
    // Active tracks per octave at end of frame (size = descriptor_levels).
    // Direct evidence the octave rule is live, and the §12.6 standoff
    // distribution: all-zero here means single-scale detection.
    std::vector<int> octave_histogram;
};

// ORB-SLAM3's geometric per-level feature allocation (lower levels get
// proportionally more; counts sum to target_n). Exposed for tests.
std::vector<int> level_target_counts(int target_n, int nlevels,
                                     double scale_factor);

class HybridFrontend {
public:
    explicit HybridFrontend(const HybridConfig& cfg = HybridConfig());

    // Bootstrap on the first frame: detect corners, compute
    // descriptors, populate active tracks. No KLT yet.
    void initialize(const cv::Mat& first_gray);

    // Run Steps 1-5 for a new incoming frame.
    FrameResult process_frame(const cv::Mat& curr_gray);

    // Force-kill tracks into the dormant buffer as if KLT had just
    // failed on them (unconditional: no min-age gate). Call AFTER
    // process_frame so victims carry their just-tracked positions.
    // Returns (id, last_x, last_y) per killed track.
    std::vector<DeathRecord> force_kill(const std::vector<std::uint64_t>& ids);

    // ---- state access (Tracking.cc reads these to build its Frame) --
    const std::unordered_map<std::uint64_t, ActiveTrack>& active_tracks() const
        { return active_tracks_; }
    const DormantTrackBuffer& dormant_buffer() const { return dormant_buffer_; }
    DormantTrackBuffer& dormant_buffer() { return dormant_buffer_; }
    std::uint64_t frame_index() const { return frame_index_; }
    const HybridConfig& config() const { return cfg_; }

    // Stage B: fresh steered-BRIEF descriptors for every active track at
    // its CURRENT position in the last processed frame, in ascending-id
    // order, with kp.angle/octave/size set. This is what the Frame
    // consumes. Tracks whose descriptor patch falls outside the image
    // are omitted (they stay active in the frontend). Call AFTER
    // initialize()/process_frame() for the frame in question.
    void describe_current(std::vector<std::uint64_t>& ids,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors);

    // Attach/detach a map-point handle to a live track (Tracking.cc's
    // hook once triangulation exists; §4.1's infant→established edge).
    // Returns false if the id is not active.
    bool set_map_point(std::uint64_t id, MapPointHandle mp);

private:
    std::uint64_t spawn_track(float x, float y, const Descriptor256& desc,
                              int octave, std::uint64_t forced_id,
                              bool has_forced_id, std::uint32_t age,
                              const Descriptor256* seed_history);

    HybridConfig cfg_;
    std::unordered_map<std::uint64_t, ActiveTrack> active_tracks_;
    // Insertion-ordered view of active ids: the Python dict preserves
    // insertion order and every per-frame loop runs in that order, so
    // determinism requires carrying it explicitly here.
    std::vector<std::uint64_t> active_order_;
    DormantTrackBuffer dormant_buffer_;
    std::uint64_t frame_index_ = 0;
    bool initialized_ = false;
    std::uint64_t next_id_ = 1;
    cv::Mat prev_gray_;
    // Unblurred pyramids for IC-angle computation; pyr_prev_ always
    // corresponds to prev_gray_, pyr_curr_ to the frame being processed.
    std::vector<cv::Mat> pyr_prev_, pyr_curr_;
    std::vector<int> umax_;
    cv::Ptr<cv::ORB> orb_;
};

}  // namespace hybrid_frontend
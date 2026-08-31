#include "HybridFrontend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <chrono>
#include <list>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace hybrid_frontend {

namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- small numeric helpers -----------------------------------------

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    std::sort(v.begin(), v.end());
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

Descriptor256 to_desc(const cv::Mat& row) {
    Descriptor256 d{};
    assert(row.cols == 32 && row.type() == CV_8U);
    std::memcpy(d.data(), row.ptr<std::uint8_t>(0), 32);
    return d;
}

// Medoid: the observation with minimum MEDIAN Hamming distance to all
// others (self-distance 0 included in each row, as in the Python).
// ORB-SLAM3's ComputeDistinctiveDescriptors.
Descriptor256 representative_descriptor(const std::vector<Descriptor256>& obs) {
    const std::size_t n = obs.size();
    assert(n >= 1);
    if (n == 1) return obs[0];
    std::vector<double> medians(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> row(n);
        for (std::size_t j = 0; j < n; ++j)
            row[j] = static_cast<double>(HammingDistance(obs[i], obs[j]));
        medians[i] = median_of(std::move(row));
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (medians[i] < medians[best]) best = i;   // ties -> lowest index
    return obs[best];
}

// ---- occupancy mask (§4.5.1 M_k) -----------------------------------

cv::Mat build_occupancy_mask(
        const cv::Size& sz,
        const std::unordered_map<std::uint64_t, ActiveTrack>& tracks,
        const std::vector<std::uint64_t>& order, int radius) {
    cv::Mat mask(sz, CV_8U, cv::Scalar(255));
    for (std::uint64_t id : order) {
        const auto it = tracks.find(id);
        if (it == tracks.end()) continue;
        const ActiveTrack& t = it->second;
        const int cx = static_cast<int>(std::lround(t.x));
        const int cy = static_cast<int>(std::lround(t.y));
        const int x0 = std::max(0, cx - radius);
        const int x1 = std::min(sz.width, cx + radius + 1);
        const int y0 = std::max(0, cy - radius);
        const int y1 = std::min(sz.height, cy + radius + 1);
        if (x0 < x1 && y0 < y1)
            mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)).setTo(0);
    }
    return mask;
}

// Nonzero inside an L-infinity square of `radius` around each center.
cv::Mat stamp_squares(const cv::Size& sz,
                      const std::vector<cv::Point2f>& centers, float radius) {
    cv::Mat hit(sz, CV_8U, cv::Scalar(0));
    const int r = static_cast<int>(std::lround(radius));
    for (const auto& c : centers) {
        const int cx = static_cast<int>(std::lround(c.x));
        const int cy = static_cast<int>(std::lround(c.y));
        const int x0 = std::max(0, cx - r);
        const int x1 = std::min(sz.width, cx + r + 1);
        const int y0 = std::max(0, cy - r);
        const int y1 = std::min(sz.height, cy + r + 1);
        if (x0 < x1 && y0 < y1)
            hit(cv::Rect(x0, y0, x1 - x0, y1 - y0)).setTo(1);
    }
    return hit;
}

// ---- ORB-SLAM3 quadtree distribution -------------------------------
// Self-contained port of ORBextractor::DistributeOctTree (the same
// algorithm the Python side ports in feature_comparison): iterative
// node splitting until the node count reaches N, then the max-response
// keypoint per surviving node.

struct QNode {
    std::vector<cv::KeyPoint> keys;
    cv::Point2i UL, UR, BL, BR;
    bool no_more = false;

    void divide(QNode& n1, QNode& n2, QNode& n3, QNode& n4) const {
        const int halfX = static_cast<int>(
            std::ceil(static_cast<float>(UR.x - UL.x) / 2.0f));
        const int halfY = static_cast<int>(
            std::ceil(static_cast<float>(BR.y - UL.y) / 2.0f));
        n1.UL = UL;
        n1.UR = {UL.x + halfX, UL.y};
        n1.BL = {UL.x, UL.y + halfY};
        n1.BR = {UL.x + halfX, UL.y + halfY};
        n2.UL = n1.UR; n2.UR = UR; n2.BL = n1.BR; n2.BR = {UR.x, UL.y + halfY};
        n3.UL = n1.BL; n3.UR = n1.BR; n3.BL = BL; n3.BR = {n1.BR.x, BL.y};
        n4.UL = n3.UR; n4.UR = n2.BR; n4.BL = n3.BR; n4.BR = BR;
        for (const auto& kp : keys) {
            QNode& tgt = (kp.pt.x < static_cast<float>(n1.UR.x))
                ? ((kp.pt.y < static_cast<float>(n1.BR.y)) ? n1 : n3)
                : ((kp.pt.y < static_cast<float>(n1.BR.y)) ? n2 : n4);
            tgt.keys.push_back(kp);
        }
        for (QNode* c : {&n1, &n2, &n3, &n4})
            if (c->keys.size() == 1) c->no_more = true;
    }
};

std::vector<cv::KeyPoint> distribute_quadtree(
        const std::vector<cv::KeyPoint>& in, int width, int height, int N) {
    if (static_cast<int>(in.size()) <= N || in.empty() || N <= 0)
        return in;

    const int nIni = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(width) / std::max(1, height))));
    const float hX = static_cast<float>(width) / static_cast<float>(nIni);

    std::list<QNode> nodes;
    std::vector<QNode*> ini(static_cast<std::size_t>(nIni));
    for (int i = 0; i < nIni; ++i) {
        QNode n;
        n.UL = {static_cast<int>(hX * static_cast<float>(i)), 0};
        n.UR = {static_cast<int>(hX * static_cast<float>(i + 1)), 0};
        n.BL = {n.UL.x, height};
        n.BR = {n.UR.x, height};
        nodes.push_back(n);
        ini[static_cast<std::size_t>(i)] = &nodes.back();
    }
    for (const auto& kp : in) {
        int b = static_cast<int>(kp.pt.x / hX);
        b = std::min(std::max(b, 0), nIni - 1);
        ini[static_cast<std::size_t>(b)]->keys.push_back(kp);
    }
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (it->keys.size() == 1) { it->no_more = true; ++it; }
        else if (it->keys.empty()) it = nodes.erase(it);
        else ++it;
    }

    bool finish = false;
    std::vector<std::pair<int, QNode*>> to_expand;
    while (!finish) {
        const std::size_t prev = nodes.size();
        to_expand.clear();
        int n_expandable = 0;
        for (auto it = nodes.begin(); it != nodes.end();) {
            if (it->no_more) { ++it; continue; }
            QNode c1, c2, c3, c4;
            it->divide(c1, c2, c3, c4);
            for (QNode* c : {&c1, &c2, &c3, &c4}) {
                if (c->keys.empty()) continue;
                nodes.push_front(*c);
                if (c->keys.size() > 1) {
                    ++n_expandable;
                    to_expand.emplace_back(
                        static_cast<int>(nodes.front().keys.size()),
                        &nodes.front());
                }
            }
            it = nodes.erase(it);
        }
        if (static_cast<int>(nodes.size()) >= N || nodes.size() == prev) {
            finish = true;
        } else if (static_cast<int>(nodes.size()) + n_expandable * 3 > N) {
            // Final stage: expand the fullest nodes first so the count
            // lands just past N instead of overshooting.
            while (!finish) {
                const std::size_t prev2 = nodes.size();
                auto expanding = to_expand;
                to_expand.clear();
                std::sort(expanding.begin(), expanding.end(),
                          [](const std::pair<int, QNode*>& a,
                             const std::pair<int, QNode*>& b)
                          { return a.first < b.first; });
                for (auto it = expanding.rbegin();
                     it != expanding.rend(); ++it) {
                    QNode c1, c2, c3, c4;
                    it->second->divide(c1, c2, c3, c4);
                    for (QNode* c : {&c1, &c2, &c3, &c4}) {
                        if (c->keys.empty()) continue;
                        nodes.push_front(*c);
                        if (c->keys.size() > 1)
                            to_expand.emplace_back(
                                static_cast<int>(nodes.front().keys.size()),
                                &nodes.front());
                    }
                    for (auto nit = nodes.begin(); nit != nodes.end(); ++nit) {
                        if (&*nit == it->second) { nodes.erase(nit); break; }
                    }
                    if (static_cast<int>(nodes.size()) >= N) break;
                }
                if (static_cast<int>(nodes.size()) >= N ||
                    nodes.size() == prev2)
                    finish = true;
            }
        }
    }

    std::vector<cv::KeyPoint> out;
    out.reserve(nodes.size());
    for (const auto& n : nodes) {
        const cv::KeyPoint* best = &n.keys.front();
        for (const auto& kp : n.keys)
            if (kp.response > best->response) best = &kp;
        out.push_back(*best);
    }
    return out;
}

// ---- multi-octave detection (multiscale_shitomasi.py port) ---------

// ORB-SLAM3-style pyramid: successive resize by 1/scale_factor,
// INTER_LINEAR, no explicit pre-blur (cv::resize low-passes), floored
// at 2*edge+2 so a level always has an interior.
std::vector<cv::Mat> build_pyramid(const cv::Mat& gray, float scale_factor,
                                   int nlevels, int edge) {
    std::vector<cv::Mat> pyr{gray};
    for (int l = 1; l < nlevels; ++l) {
        const cv::Mat& prev = pyr.back();
        const int w = std::max(2 * edge + 2, static_cast<int>(std::lround(
            prev.cols / static_cast<double>(scale_factor))));
        const int h = std::max(2 * edge + 2, static_cast<int>(std::lround(
            prev.rows / static_cast<double>(scale_factor))));
        cv::Mat next;
        cv::resize(prev, next, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        pyr.push_back(next);
    }
    return pyr;
}

// Per-level Shi-Tomasi with the RANK rule settled by octave_rule_test:
// 7x7 NMS peaks above a low noise floor (0.01*quality*level_max), then
// the per-level geometric target enforced by the quadtree. Returns
// LEVEL-0 coordinates with octave, size = patch*sf^octave, response =
// lambda2. `level0_eig` is the caller's shared cornerMinEigenVal map.
// The occupancy mask is applied per level via nearest resize; the
// priority (dormant-seeding) split runs per level on lifted coords.
// NOTE (faithful quirk): the quadtree receives the ROI dimensions
// (w-2e, h-2e) but UNSHIFTED level coordinates, exactly as the Python
// reference does; the resulting cell skew is part of the validated
// selection behaviour and is preserved deliberately.
std::vector<cv::KeyPoint> detect_multiscale_shi_tomasi(
        const cv::Mat& gray, const std::vector<cv::Mat>& pyr,
        const cv::Mat& level0_eig,
        const cv::Mat& mask, int target_n, const HybridConfig& cfg,
        const cv::Mat* priority_map) {
    const float sf = cfg.descriptor_scale_factor;
    const int nlevels = std::max(1, cfg.descriptor_levels);
    const int edge = cfg.descriptor_patch_size / 2;
    const int nms_r = std::max(1, cfg.shi_tomasi_min_distance / 2);
    const int block = cfg.shi_tomasi_block_size;

    (void)gray;   // level 0 is pyr[0]
    const auto targets = level_target_counts(target_n, nlevels,
                                             static_cast<double>(sf));
    std::vector<cv::KeyPoint> all;
    const cv::Mat nms_kernel = cv::Mat::ones(2 * nms_r + 1,
                                             2 * nms_r + 1, CV_8U);
    for (int lvl = 0; lvl < nlevels; ++lvl) {
        const cv::Mat& img = pyr[static_cast<std::size_t>(lvl)];
        const int h = img.rows, w = img.cols;
        if (w <= 2 * edge || h <= 2 * edge) continue;

        cv::Mat lam;
        if (lvl == 0 && block == cfg.shi_tomasi_block_size &&
            !level0_eig.empty())
            lam = level0_eig;
        else
            cv::cornerMinEigenVal(img, lam, block);

        cv::Mat lmask;
        if (lvl == 0) lmask = mask;
        else cv::resize(mask, lmask, img.size(), 0, 0, cv::INTER_NEAREST);

        double lam_max = 0.0;
        cv::minMaxLoc(lam(cv::Rect(edge, edge, w - 2 * edge,
                                   h - 2 * edge)),
                      nullptr, &lam_max);
        const float thr = static_cast<float>(
            0.01 * cfg.shi_tomasi_quality * lam_max);

        cv::Mat dil;
        cv::dilate(lam, dil, nms_kernel);

        const float s = std::pow(sf, static_cast<float>(lvl));
        std::vector<cv::KeyPoint> cands;
        std::vector<cv::KeyPoint> near_kps, far_kps;
        // Vectorised candidate selection: peak AND above-floor AND
        // unmasked. The scalar equivalent walked ~3M pyramid pixels per
        // frame and dominated the detect budget.
        {
            cv::Mat peak, above, sel;
            cv::compare(lam, dil, peak, cv::CMP_GE);
            cv::compare(lam, thr, above, cv::CMP_GT);
            cv::bitwise_and(peak, above, sel);
            cv::bitwise_and(sel, lmask, sel);
            const cv::Rect roi(edge, edge, w - 2 * edge, h - 2 * edge);
            cv::Mat sel_roi = sel(roi);
            if (cv::countNonZero(sel_roi) > 0) {
                std::vector<cv::Point> pts;
                cv::findNonZero(sel_roi, pts);
                cands.reserve(pts.size());
                for (const auto& pt : pts) {
                    const int x = pt.x + edge, y = pt.y + edge;
                    cv::KeyPoint kp(static_cast<float>(x),
                                    static_cast<float>(y), 7.0f, -1.0f,
                                    lam.at<float>(y, x));
                    kp.octave = lvl;
                    cands.push_back(kp);
                }
            }
        }
        const int t_l = targets[static_cast<std::size_t>(lvl)];
        std::vector<cv::KeyPoint> keep;
        if (priority_map != nullptr &&
            static_cast<int>(cands.size()) > t_l) {
            const int ph = priority_map->rows, pw = priority_map->cols;
            for (const auto& kp : cands) {
                const int x0 = std::min(std::max(static_cast<int>(
                    std::lround(kp.pt.x * s)), 0), pw - 1);
                const int y0 = std::min(std::max(static_cast<int>(
                    std::lround(kp.pt.y * s)), 0), ph - 1);
                (priority_map->at<std::uint8_t>(y0, x0) ? near_kps
                                                        : far_kps)
                    .push_back(kp);
            }
            std::stable_sort(near_kps.begin(), near_kps.end(),
                             [](const cv::KeyPoint& a,
                                const cv::KeyPoint& b)
                             { return a.response > b.response; });
            if (static_cast<int>(near_kps.size()) > t_l)
                near_kps.resize(static_cast<std::size_t>(t_l));
            keep = near_kps;
            const int rem = t_l - static_cast<int>(keep.size());
            if (rem > 0) {
                if (static_cast<int>(far_kps.size()) > rem) {
                    auto extra = distribute_quadtree(
                        far_kps, w - 2 * edge, h - 2 * edge, rem);
                    keep.insert(keep.end(), extra.begin(), extra.end());
                } else {
                    keep.insert(keep.end(), far_kps.begin(),
                                far_kps.end());
                }
            }
        } else if (static_cast<int>(cands.size()) > t_l) {
            keep = distribute_quadtree(cands, w - 2 * edge,
                                       h - 2 * edge, t_l);
        } else {
            keep = cands;
        }
        for (auto& kp : keep) {
            kp.pt.x *= s;
            kp.pt.y *= s;
            kp.octave = lvl;
            kp.size = static_cast<float>(cfg.descriptor_patch_size) * s;
        }
        all.insert(all.end(), keep.begin(), keep.end());
    }
    return all;
}

// ---- Step 4 detection ----------------------------------------------

std::vector<cv::KeyPoint> detect_shi_tomasi_with_quadtree(
        const cv::Mat& gray, const cv::Mat& eig, const cv::Mat& mask,
        int target_n, double quality, int min_distance, int block_size,
        const cv::Mat* priority_map, int pool_n) {
    // `eig` is the caller's shared cornerMinEigenVal map for `gray`.
    // Recomputing it here was a full-frame pass per call, and the cost
    // is per-PIXEL, not per-feature, so it dominated at 1280x720.
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray, pts, pool_n, quality,
                            static_cast<double>(min_distance), mask,
                            block_size);
    if (pts.empty()) return {};
    const int h = gray.rows, w = gray.cols;
    std::vector<cv::KeyPoint> kps;
    kps.reserve(pts.size());
    for (const auto& p : pts) {
        const int xi = std::min(std::max(
            static_cast<int>(std::lround(p.x)), 0), w - 1);
        const int yi = std::min(std::max(
            static_cast<int>(std::lround(p.y)), 0), h - 1);
        kps.emplace_back(p.x, p.y, 7.0f, -1.0f, eig.at<float>(yi, xi));
    }
    if (priority_map != nullptr && static_cast<int>(kps.size()) > target_n) {
        std::vector<cv::KeyPoint> near_kps, far_kps;
        for (const auto& kp : kps) {
            const int xi = std::min(std::max(
                static_cast<int>(std::lround(kp.pt.x)), 0), w - 1);
            const int yi = std::min(std::max(
                static_cast<int>(std::lround(kp.pt.y)), 0), h - 1);
            (priority_map->at<std::uint8_t>(yi, xi) ? near_kps : far_kps)
                .push_back(kp);
        }
        std::stable_sort(near_kps.begin(), near_kps.end(),
                         [](const cv::KeyPoint& a, const cv::KeyPoint& b)
                         { return a.response > b.response; });
        if (static_cast<int>(near_kps.size()) > target_n)
            near_kps.resize(static_cast<std::size_t>(target_n));
        std::vector<cv::KeyPoint> keep = near_kps;
        const int remaining = target_n - static_cast<int>(keep.size());
        if (remaining > 0) {
            if (static_cast<int>(far_kps.size()) > remaining) {
                auto extra = distribute_quadtree(far_kps, w, h, remaining);
                keep.insert(keep.end(), extra.begin(), extra.end());
            } else {
                keep.insert(keep.end(), far_kps.begin(), far_kps.end());
            }
        }
        return keep;
    }
    if (static_cast<int>(kps.size()) > target_n)
        kps = distribute_quadtree(kps, w, h, target_n);
    return kps;
}

// ---- steered BRIEF at caller-provided positions --------------------

// ---- intensity-centroid orientation (ORB-SLAM3 IC_Angle) ------------
// cv::ORB::compute with USER-PROVIDED keypoints does NOT compute
// orientation: it uses whatever kp.angle already holds. With the default
// -1 every descriptor this frontend produced was effectively unsteered.
// Harmless while frontend descriptors were only compared with each other
// (re-ID), fatal once they meet ORBextractor's steered descriptors in
// ComputeStereoMatches. This is the same IC angle ORBextractor computes,
// on the same unblurred pyramid level, so the two are now comparable.

constexpr int kHalfPatch = 15;

std::vector<int> make_umax() {
    std::vector<int> umax(kHalfPatch + 1);
    const int vmax = static_cast<int>(std::floor(
        kHalfPatch * std::sqrt(2.0) / 2 + 1));
    const int vmin = static_cast<int>(std::ceil(
        kHalfPatch * std::sqrt(2.0) / 2));
    const double hp2 = static_cast<double>(kHalfPatch) * kHalfPatch;
    for (int v = 0; v <= vmax; ++v)
        umax[static_cast<std::size_t>(v)] = static_cast<int>(
            std::lround(std::sqrt(hp2 - static_cast<double>(v) * v)));
    for (int v = kHalfPatch, v0 = 0; v >= vmin; --v) {
        while (umax[static_cast<std::size_t>(v0)] ==
               umax[static_cast<std::size_t>(v0 + 1)])
            ++v0;
        umax[static_cast<std::size_t>(v)] = v0;
        ++v0;
    }
    return umax;
}

// Angle in degrees at an integer pixel of a level image; caller
// guarantees the (2*kHalfPatch+1)^2 patch is inside the image.
float ic_angle(const cv::Mat& img, int cx, int cy,
               const std::vector<int>& umax) {
    int m_01 = 0, m_10 = 0;
    const std::uint8_t* center = img.ptr<std::uint8_t>(cy) + cx;
    for (int u = -kHalfPatch; u <= kHalfPatch; ++u)
        m_10 += u * center[u];
    const int step = static_cast<int>(img.step1());
    for (int v = 1; v <= kHalfPatch; ++v) {
        int v_sum = 0;
        const int d = umax[static_cast<std::size_t>(v)];
        for (int u = -d; u <= d; ++u) {
            const int val_plus = center[u + v * step];
            const int val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    return cv::fastAtan2(static_cast<float>(m_01),
                         static_cast<float>(m_10));
}

// Stamp kp.angle for every keypoint from the pyramid level named by its
// octave. Keypoints whose patch leaves the level image keep angle=-1,
// and are then dropped by ORB.compute's own border filter anyway.
void assign_ic_angles(std::vector<cv::KeyPoint>& kps,
                      const std::vector<cv::Mat>& pyr, float scale_factor,
                      const std::vector<int>& umax) {
    for (auto& kp : kps) {
        const int lvl = kp.octave;
        if (lvl < 0 || lvl >= static_cast<int>(pyr.size())) continue;
        const cv::Mat& img = pyr[static_cast<std::size_t>(lvl)];
        const float inv = 1.0f / std::pow(scale_factor,
                                          static_cast<float>(lvl));
        const int cx = static_cast<int>(std::lround(kp.pt.x * inv));
        const int cy = static_cast<int>(std::lround(kp.pt.y * inv));
        if (cx - kHalfPatch < 0 || cy - kHalfPatch < 0 ||
            cx + kHalfPatch >= img.cols || cy + kHalfPatch >= img.rows)
            continue;
        kp.angle = ic_angle(img, cx, cy, umax);
    }
}

void compute_steered_brief(const cv::Mat& gray,
                           std::vector<cv::KeyPoint>& kps,
                           cv::Ptr<cv::ORB>& orb, int patch_size,
                           float scale_factor,
                           const std::vector<cv::Mat>& pyr,
                           const std::vector<int>& umax,
                           std::vector<cv::KeyPoint>& kept,
                           cv::Mat& desc) {
    kept.clear();
    desc.release();
    if (kps.empty()) return;
    // ORB.compute honours kp.octave (which of ITS pyramid levels the
    // descriptor is taken from), so the patch size is stamped at the
    // level's scale. Octave-0 keypoints degrade to the flat behaviour.
    for (auto& kp : kps)
        kp.size = static_cast<float>(patch_size) *
                  std::pow(scale_factor, static_cast<float>(kp.octave));
    assign_ic_angles(kps, pyr, scale_factor, umax);
    orb->compute(gray, kps, desc);
    kept = kps;   // OpenCV drops border keypoints from the vector itself
    if (desc.empty()) kept.clear();
}

// {input_index -> descriptor} for positions that survived the border
// drop, aligned via the class_id tag (ORB.compute preserves it).
std::unordered_map<int, Descriptor256> descriptors_at_positions(
        const cv::Mat& gray, const std::vector<cv::Point2f>& xy,
        const std::vector<int>& octaves,
        cv::Ptr<cv::ORB>& orb, int patch_size, float scale_factor,
        const std::vector<cv::Mat>& pyr, const std::vector<int>& umax) {
    std::unordered_map<int, Descriptor256> out;
    if (xy.empty()) return out;
    assert(octaves.size() == xy.size());
    std::vector<cv::KeyPoint> kps;
    kps.reserve(xy.size());
    for (std::size_t i = 0; i < xy.size(); ++i) {
        const float sc = std::pow(scale_factor,
                                  static_cast<float>(octaves[i]));
        cv::KeyPoint kp(xy[i].x, xy[i].y,
                        static_cast<float>(patch_size) * sc);
        kp.octave = octaves[i];
        kp.class_id = static_cast<int>(i);
        kps.push_back(kp);
    }
    assign_ic_angles(kps, pyr, scale_factor, umax);
    cv::Mat desc;
    orb->compute(gray, kps, desc);
    if (desc.empty()) return out;
    for (int r = 0; r < desc.rows; ++r) {
        const int src = kps[static_cast<std::size_t>(r)].class_id;
        if (src >= 0 && src < static_cast<int>(xy.size()))
            out[src] = to_desc(desc.row(r));
    }
    return out;
}

// ---- local corners inside dormant windows --------------------------

std::vector<cv::Point2f> local_corners_in_windows(
        const cv::Mat& gray, const cv::Mat& eig, const cv::Mat& mask,
        const std::vector<cv::Point2f>& centers, float radius,
        double quality, double quality_scale,
        int max_windows, float min_separation,
        const std::vector<cv::Point2f>& existing) {
    std::vector<cv::Point2f> chosen;
    if (centers.empty()) return chosen;
    const int h = gray.rows, w = gray.cols;
    double eig_max = 0.0;
    cv::minMaxLoc(eig, nullptr, &eig_max);
    const float thresh = static_cast<float>(eig_max * quality * quality_scale);
    const int r = std::max(1, static_cast<int>(std::lround(radius)));
    std::vector<cv::Point2f> taken = existing;
    const float sep2 = min_separation * min_separation;
    const int limit = std::min<int>(max_windows,
                                    static_cast<int>(centers.size()));
    for (int ci = 0; ci < limit; ++ci) {
        const auto& ctr = centers[static_cast<std::size_t>(ci)];
        const int cx = static_cast<int>(std::lround(ctr.x));
        const int cy = static_cast<int>(std::lround(ctr.y));
        const int x0 = std::max(0, cx - r), x1 = std::min(w, cx + r + 1);
        const int y0 = std::max(0, cy - r), y1 = std::min(h, cy + r + 1);
        if (x0 >= x1 || y0 >= y1) continue;
        float best = -1.0f;
        int bx = -1, by = -1;
        for (int y = y0; y < y1; ++y) {
            const float* e = eig.ptr<float>(y);
            const std::uint8_t* m = mask.ptr<std::uint8_t>(y);
            for (int x = x0; x < x1; ++x) {
                if (m[x] == 0) continue;
                if (e[x] > best) { best = e[x]; bx = x; by = y; }
            }
        }
        if (bx < 0 || best < thresh) continue;
        const cv::Point2f p(static_cast<float>(bx), static_cast<float>(by));
        bool clash = false;
        for (const auto& t : taken) {
            const float dx = p.x - t.x, dy = p.y - t.y;
            if (dx * dx + dy * dy < sep2) { clash = true; break; }
        }
        if (clash) continue;
        chosen.push_back(p);
        taken.push_back(p);
    }
    return chosen;
}

// ---- Step 3 RANSAC on F --------------------------------------------

std::vector<bool> ransac_inlier_mask(const std::vector<cv::Point2f>& prev,
                                     const std::vector<cv::Point2f>& curr,
                                     double ransac_px, int min_n) {
    const std::size_t n = prev.size();
    std::vector<bool> out(n, false);
    if (static_cast<int>(n) < min_n) return out;
    float minx1 = std::numeric_limits<float>::max(), maxx1 = -minx1;
    float miny1 = minx1, maxy1 = -minx1;
    float minx2 = minx1, maxx2 = -minx1, miny2 = minx1, maxy2 = -minx1;
    for (std::size_t i = 0; i < n; ++i) {
        minx1 = std::min(minx1, prev[i].x); maxx1 = std::max(maxx1, prev[i].x);
        miny1 = std::min(miny1, prev[i].y); maxy1 = std::max(maxy1, prev[i].y);
        minx2 = std::min(minx2, curr[i].x); maxx2 = std::max(maxx2, curr[i].x);
        miny2 = std::min(miny2, curr[i].y); maxy2 = std::max(maxy2, curr[i].y);
    }
    // Degeneracy guard, as in the Python: a near-collinear or
    // point-like cloud makes F meaningless — reject all survivors.
    if (std::min(maxx1 - minx1, maxy1 - miny1) < 5.0f ||
        std::min(maxx2 - minx2, maxy2 - miny2) < 5.0f)
        return out;
    cv::Mat inl, F;
    try {
        F = cv::findFundamentalMat(prev, curr, cv::FM_RANSAC,
                                   ransac_px, 0.99, inl);
    } catch (const cv::Exception&) {
        return out;
    }
    if (F.empty() || F.rows != 3 || F.cols != 3 || inl.empty())
        return out;
    for (std::size_t i = 0; i < n; ++i)
        out[i] = inl.at<std::uint8_t>(static_cast<int>(i)) != 0;
    return out;
}

}  // namespace

std::vector<int> level_target_counts(int target_n, int nlevels,
                                     double scale_factor) {
    // Faithful to multiscale_shitomasi.level_target_counts: geometric
    // series rounded per level, remainder (>=1) into the top level.
    std::vector<int> counts;
    if (nlevels <= 1) { counts.push_back(target_n); return counts; }
    const double factor = 1.0 / scale_factor;
    const double unscaled = target_n * (1.0 - factor) /
                            (1.0 - std::pow(factor, nlevels));
    int cum = 0;
    for (int lvl = 0; lvl < nlevels - 1; ++lvl) {
        const int n = static_cast<int>(std::lround(
            unscaled * std::pow(factor, lvl)));
        counts.push_back(n);
        cum += n;
    }
    counts.push_back(std::max(1, target_n - cum));
    return counts;
}

// ---- HybridFrontend ------------------------------------------------

HybridFrontend::HybridFrontend(const HybridConfig& cfg)
    : cfg_(cfg),
      dormant_buffer_(static_cast<std::uint64_t>(
          std::max(0, cfg.dormant_horizon_frames))) {
    orb_ = cv::ORB::create(4 * cfg_.target_active_tracks,
                           cfg_.descriptor_scale_factor,
                           cfg_.descriptor_levels,
                           /*edgeThreshold=*/cfg_.descriptor_patch_size / 2,
                           /*firstLevel=*/0, /*WTA_K=*/2,
                           cv::ORB::HARRIS_SCORE,
                           /*patchSize=*/cfg_.descriptor_patch_size);
    umax_ = make_umax();
}

void HybridFrontend::initialize(const cv::Mat& first_gray) {
    assert(first_gray.type() == CV_8U);
    frame_index_ = 0;
    initialized_ = true;
    active_tracks_.clear();
    active_order_.clear();
    dormant_buffer_.clear();
    cv::Mat full_mask(first_gray.size(), CV_8U, cv::Scalar(255));
    pyr_prev_ = build_pyramid(first_gray, cfg_.descriptor_scale_factor,
                              std::max(1, cfg_.descriptor_levels),
                              cfg_.descriptor_patch_size / 2);
    cv::Mat eig;
    cv::cornerMinEigenVal(first_gray, eig, cfg_.shi_tomasi_block_size);
    std::vector<cv::KeyPoint> kps;
    if (cfg_.multiscale_detection)
        kps = detect_multiscale_shi_tomasi(first_gray, pyr_prev_, eig,
                                           full_mask,
                                           cfg_.target_active_tracks,
                                           cfg_, nullptr);
    else
        kps = detect_shi_tomasi_with_quadtree(
            first_gray, eig, full_mask, cfg_.target_active_tracks,
            cfg_.shi_tomasi_quality, cfg_.shi_tomasi_min_distance,
            cfg_.shi_tomasi_block_size, nullptr,
            cfg_.target_active_tracks * 2);
    std::vector<cv::KeyPoint> kept;
    cv::Mat desc;
    compute_steered_brief(first_gray, kps, orb_,
                          cfg_.descriptor_patch_size,
                          cfg_.descriptor_scale_factor, pyr_prev_, umax_,
                          kept, desc);
    for (int i = 0; i < static_cast<int>(kept.size()); ++i)
        spawn_track(kept[static_cast<std::size_t>(i)].pt.x,
                    kept[static_cast<std::size_t>(i)].pt.y,
                    to_desc(desc.row(i)),
                    kept[static_cast<std::size_t>(i)].octave,
                    0, false, 0, nullptr);
    prev_gray_ = first_gray.clone();
}

FrameResult HybridFrontend::process_frame(const cv::Mat& curr_gray) {
    assert(initialized_ && "Call initialize() first");
    assert(curr_gray.type() == CV_8U);
    ++frame_index_;
    const HybridConfig& cfg = cfg_;
    const Clock::time_point t_start = Clock::now();
    Clock::time_point t_mark = t_start;

    // Purge every frame — purging only when Step 4 fires lets the
    // buffer grow unboundedly whenever the active set stays at target.
    dormant_buffer_.purge_older_than(frame_index_);

    // One pyramid per frame: orientation for every descriptor computed
    // on this frame, and Step 4 detection, all read from it.
    pyr_curr_ = build_pyramid(curr_gray, cfg.descriptor_scale_factor,
                              std::max(1, cfg.descriptor_levels),
                              cfg.descriptor_patch_size / 2);

    FrameResult res;
    res.frame_index = frame_index_;
    res.tracks_in = static_cast<int>(active_tracks_.size());
    float flow_dx = 0.0f, flow_dy = 0.0f;

    // ============ Steps 1-3: KLT, FB filter, RANSAC =================
    if (!active_tracks_.empty()) {
        const std::vector<std::uint64_t> ids = active_order_;
        std::vector<cv::Point2f> prev_pts;
        prev_pts.reserve(ids.size());
        for (std::uint64_t id : ids) {
            const ActiveTrack& t = active_tracks_.at(id);
            prev_pts.emplace_back(t.x, t.y);
        }
        std::vector<cv::Point2f> curr_pts, back_pts;
        std::vector<std::uint8_t> st_fwd, st_bwd;
        std::vector<float> err;
        const cv::TermCriteria crit(
            cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
            cfg.klt_max_iter, cfg.klt_eps);
        cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_pts, curr_pts,
                                 st_fwd, err, cfg.klt_window,
                                 cfg.klt_pyramid_levels, crit);
        cv::calcOpticalFlowPyrLK(curr_gray, prev_gray_, curr_pts, back_pts,
                                 st_bwd, err, cfg.klt_window,
                                 cfg.klt_pyramid_levels, crit);

        const std::size_t n = ids.size();
        std::vector<bool> surv(n, false);
        int n_klt = 0, n_fb = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const bool klt_ok = st_fwd[i] == 1 && st_bwd[i] == 1;
            if (klt_ok) ++n_klt;
            const float dx = prev_pts[i].x - back_pts[i].x;
            const float dy = prev_pts[i].y - back_pts[i].y;
            const bool fb_ok =
                std::sqrt(dx * dx + dy * dy) < cfg.fb_threshold_px;
            // KLT can report status=1 for a point that has drifted
            // just outside the image. Downstream consumers (Frame's
            // stereo matcher indexes rows by pt.y unchecked) cannot
            // tolerate that, and neither can any descriptor step, so
            // an off-image track is dead.
            const bool inside =
                curr_pts[i].x >= 0.0f && curr_pts[i].y >= 0.0f &&
                curr_pts[i].x < static_cast<float>(curr_gray.cols) &&
                curr_pts[i].y < static_cast<float>(curr_gray.rows);
            surv[i] = klt_ok && fb_ok && inside;
            if (surv[i]) ++n_fb;
        }
        res.tracks_after_klt = n_klt;
        res.tracks_after_fb = n_fb;

        std::vector<std::uint64_t> surv_ids;
        std::vector<cv::Point2f> surv_prev, surv_curr;
        for (std::size_t i = 0; i < n; ++i) {
            if (!surv[i]) continue;
            surv_ids.push_back(ids[i]);
            surv_prev.push_back(prev_pts[i]);
            surv_curr.push_back(curr_pts[i]);
        }

        std::vector<bool> inlier;
        if (static_cast<int>(surv_ids.size()) >= cfg.min_matches_for_F)
            inlier = ransac_inlier_mask(surv_prev, surv_curr,
                                        cfg.ransac_reproj_px,
                                        cfg.min_matches_for_F);
        else
            inlier.assign(surv_ids.size(), true);   // too few to test

        int n_inl = 0;
        std::vector<double> fx, fy;
        for (std::size_t i = 0; i < inlier.size(); ++i) {
            if (!inlier[i]) continue;
            ++n_inl;
            fx.push_back(static_cast<double>(surv_curr[i].x - surv_prev[i].x));
            fy.push_back(static_cast<double>(surv_curr[i].y - surv_prev[i].y));
        }
        res.tracks_after_ransac = n_inl;
        if (n_inl > 0) {
            flow_dx = static_cast<float>(median_of(fx));
            flow_dy = static_cast<float>(median_of(fy));
        }

        // Motion-compensate existing dormant predictions BEFORE this
        // frame's deaths are added (deaths are stored flow-adjusted).
        if (cfg.motion_compensate_dormant)
            dormant_buffer_.translate_all(flow_dx, flow_dy);

        // Survivors: update position and age.
        std::vector<std::uint64_t> kept_ids;
        for (std::size_t i = 0; i < surv_ids.size(); ++i) {
            if (!inlier[i]) continue;
            ActiveTrack& t = active_tracks_.at(surv_ids[i]);
            t.x = surv_curr[i].x;
            t.y = surv_curr[i].y;
            t.age += 1;
            kept_ids.push_back(surv_ids[i]);
        }

        // Representative-descriptor accumulation for survivors.
        if (cfg.use_representative_descriptor && !kept_ids.empty()) {
            // Apply the stride gate BEFORE describing: only tracks due
            // for a sample this frame need a descriptor computed.
            const std::uint32_t stride = static_cast<std::uint32_t>(
                std::max(1, cfg.representative_sample_stride));
            std::vector<std::uint64_t> sampled;
            std::vector<cv::Point2f> pos;
            std::vector<int> oct;
            for (std::uint64_t id : kept_ids) {
                const ActiveTrack& t = active_tracks_.at(id);
                if (t.age % stride != 0) continue;
                sampled.push_back(id);
                pos.emplace_back(t.x, t.y);
                oct.push_back(t.octave);
            }
            auto fresh = descriptors_at_positions(
                curr_gray, pos, oct, orb_, cfg.descriptor_patch_size,
                cfg.descriptor_scale_factor, pyr_curr_, umax_);
            for (const auto& kv : fresh) {
                ActiveTrack& t = active_tracks_.at(
                    sampled[static_cast<std::size_t>(kv.first)]);
                t.descriptor_history.push_back(kv.second);
                if (static_cast<int>(t.descriptor_history.size()) >
                    cfg.representative_max_observations)
                    // Drop the OLDEST non-birth observation; index 0 is
                    // the birth appearance, kept as an anchor.
                    t.descriptor_history.erase(
                        t.descriptor_history.begin() + 1);
                t.representative_descriptor =
                    representative_descriptor(t.descriptor_history);
                t.has_representative = true;
            }
        }

        // Deaths: everything not kept. Store the descriptor recomputed
        // at the LAST SUCCESSFUL OBSERVATION on the PREVIOUS frame (the
        // stale-birth-descriptor fix), preferring the medoid.
        std::vector<std::uint64_t> dying;
        {
            std::unordered_map<std::uint64_t, bool> kept_set;
            for (std::uint64_t id : kept_ids) kept_set[id] = true;
            for (std::uint64_t id : ids)
                if (!kept_set.count(id)) dying.push_back(id);
        }
        if (!dying.empty()) {
            // Only tracks that will actually be BUFFERED need a
            // death-time descriptor. Infants are retired unread by the
            // min-age gate below, and on real footage they dominate
            // deaths, so describing them first was the single largest
            // wasted cost per frame.
            const std::uint32_t min_age = static_cast<std::uint32_t>(
                std::max(0, cfg.dormant_min_track_age));
            std::vector<cv::Point2f> dpos;
            std::vector<int> doct;
            std::vector<std::size_t> dmap;   // dpos index -> dying index
            for (std::size_t k = 0; k < dying.size(); ++k) {
                const ActiveTrack& t = active_tracks_.at(dying[k]);
                if (t.age < min_age) continue;
                dmap.push_back(k);
                dpos.emplace_back(t.x, t.y);
                doct.push_back(t.octave);
            }
            auto desc_by_pos = descriptors_at_positions(
                prev_gray_, dpos, doct, orb_, cfg.descriptor_patch_size,
                cfg.descriptor_scale_factor, pyr_prev_, umax_);
            std::unordered_map<std::size_t, Descriptor256> death_desc;
            for (const auto& kv : desc_by_pos)
                death_desc[dmap[static_cast<std::size_t>(kv.first)]] = kv.second;
            for (std::size_t k = 0; k < dying.size(); ++k) {
                const std::uint64_t tid = dying[k];
                ActiveTrack t = active_tracks_.at(tid);
                active_tracks_.erase(tid);
                active_order_.erase(
                    std::find(active_order_.begin(), active_order_.end(),
                              tid));
                res.died_this_frame.push_back({tid, t.x, t.y});
                if (t.age < min_age)
                    continue;   // retire: mostly detector noise
                Descriptor256 stored = t.birth_descriptor;
                const auto it = death_desc.find(k);
                if (it != death_desc.end()) stored = it->second;
                if (cfg.use_representative_descriptor && t.has_representative)
                    stored = t.representative_descriptor;
                float px = t.x, py = t.y;
                if (cfg.motion_compensate_dormant) {
                    px += flow_dx;
                    py += flow_dy;
                }
                DormantTrack d;
                d.id = tid;
                d.last_x = px;
                d.last_y = py;
                d.descriptor = stored;
                d.frame_died = frame_index_;
                d.octave = t.octave;
                d.age_at_death = t.age;
                d.map_point = nullptr;
                dormant_buffer_.add(d);
            }
        }
    }

    res.ms_track = ms_since(t_mark);
    t_mark = Clock::now();

    // ============ Step 4: Shi-Tomasi top-up + BRIEF =================
    const int deficit =
        cfg.target_active_tracks - static_cast<int>(active_tracks_.size());
    // Amortise the fixed detection cost: a small deficit is not worth a
    // full pyramid pass. Purging already happened above, so the dormant
    // buffer stays correctly bounded on skipped frames.
    const int deficit_floor = static_cast<int>(std::lround(
        cfg.min_deficit_fraction * cfg.target_active_tracks));
    if (deficit > 0 && deficit >= deficit_floor) {
        // ONE cornerMinEigenVal pass per frame, shared by Step 4
        // response sampling and Step 5 local detection.
        cv::Mat eig;
        cv::cornerMinEigenVal(curr_gray, eig, cfg.shi_tomasi_block_size);
        cv::Mat mask = build_occupancy_mask(curr_gray.size(), active_tracks_,
                                            active_order_,
                                            cfg.occupancy_mask_radius);
        cv::Mat priority;
        const cv::Mat* priority_ptr = nullptr;
        int pool_n = deficit * 2;
        if (cfg.seed_corners_near_dormant && !dormant_buffer_.empty()) {
            std::vector<cv::Point2f> dorm_xy;
            for (const auto& e : dormant_buffer_.all_entries())
                dorm_xy.emplace_back(e.last_x, e.last_y);
            priority = stamp_squares(curr_gray.size(), dorm_xy,
                                     cfg.reid_radius_px);
            priority_ptr = &priority;
            pool_n = deficit * 2 +
                     std::min<int>(static_cast<int>(dormant_buffer_.size()),
                                   500);
        }
        std::vector<cv::KeyPoint> kps;
        if (cfg.multiscale_detection)
            kps = detect_multiscale_shi_tomasi(curr_gray, pyr_curr_, eig,
                                               mask, deficit, cfg,
                                               priority_ptr);
        else
            kps = detect_shi_tomasi_with_quadtree(
                curr_gray, eig, mask, deficit, cfg.shi_tomasi_quality,
                cfg.shi_tomasi_min_distance, cfg.shi_tomasi_block_size,
                priority_ptr, pool_n);
        std::vector<cv::KeyPoint> kept;
        cv::Mat desc;
        compute_steered_brief(curr_gray, kps, orb_,
                              cfg.descriptor_patch_size,
                              cfg.descriptor_scale_factor, pyr_curr_, umax_,
                              kept, desc);
        res.new_corners_detected = static_cast<int>(kept.size());

        res.ms_detect = ms_since(t_mark);
        t_mark = Clock::now();

        // ============ Step 5: re-ID against the dormant buffer ======
        if (cfg.enable_reid && !dormant_buffer_.empty()) {
            // Queries: new corners (global + local); candidates:
            // dormant entries. Local-only queries (index >= n_global)
            // are discarded when unmatched — they can never inflate the
            // active set past N_target.
            std::vector<cv::Point2f> q_xy;
            std::vector<Descriptor256> q_desc;
            std::vector<int> q_oct;
            for (int i = 0; i < static_cast<int>(kept.size()); ++i) {
                q_xy.push_back(kept[static_cast<std::size_t>(i)].pt);
                q_desc.push_back(to_desc(desc.row(i)));
                q_oct.push_back(kept[static_cast<std::size_t>(i)].octave);
            }
            const int n_global = static_cast<int>(q_xy.size());

            if (cfg.local_detect_in_dormant_windows) {
                std::vector<cv::Point2f> dorm_pos;
                for (const auto& e : dormant_buffer_.all_entries())
                    dorm_pos.emplace_back(e.last_x, e.last_y);
                // Same mask as Step 4: nothing has been spawned yet, so
                // rebuilding it would produce an identical result.
                auto local_xy = local_corners_in_windows(
                    curr_gray, eig, mask, dorm_pos, cfg.reid_radius_px,
                    cfg.shi_tomasi_quality,
                    cfg.local_detect_quality_scale,
                    cfg.local_detect_max_windows,
                    static_cast<float>(cfg.shi_tomasi_min_distance), q_xy);
                if (!local_xy.empty()) {
                    std::vector<cv::KeyPoint> lk;
                    for (const auto& p : local_xy)
                        lk.emplace_back(
                            p.x, p.y,
                            static_cast<float>(cfg.descriptor_patch_size));
                    std::vector<cv::KeyPoint> lkept;
                    cv::Mat ldesc;
                    compute_steered_brief(curr_gray, lk, orb_,
                                          cfg.descriptor_patch_size,
                                          cfg.descriptor_scale_factor,
                                          pyr_curr_, umax_,
                                          lkept, ldesc);
                    for (int j = 0; j < static_cast<int>(lkept.size()); ++j) {
                        q_xy.push_back(
                            lkept[static_cast<std::size_t>(j)].pt);
                        q_desc.push_back(to_desc(ldesc.row(j)));
                        q_oct.push_back(0);   // level-0 booster corners
                    }
                }
            }

            std::vector<PixelQuery> queries;
            queries.reserve(q_xy.size());
            for (std::size_t i = 0; i < q_xy.size(); ++i) {
                PixelQuery q;
                q.x = q_xy[i].x;
                q.y = q_xy[i].y;
                q.descriptor = q_desc[i];
                queries.push_back(q);
            }
            const auto& entries = dormant_buffer_.all_entries();
            std::vector<DormantTrack> all_dormant(entries.begin(),
                                                  entries.end());
            std::vector<PixelCandidate> candidates;
            candidates.reserve(all_dormant.size());
            for (const auto& e : all_dormant) {
                PixelCandidate c;
                c.x = e.last_x;
                c.y = e.last_y;
                c.descriptor = e.descriptor;
                // Gap-scaled acceptance (§4.6): the ceiling covers this
                // entry's OWN dormancy gap.
                const std::uint64_t g =
                    DormantTrackBuffer::gap_frames(e, frame_index_);
                c.hamming_threshold = std::min<int>(
                    cfg.reid_hamming_cap,
                    static_cast<int>(std::llround(
                        cfg.reid_hamming_threshold +
                        cfg.reid_hamming_slope_per_frame *
                            static_cast<double>(g))));
                candidates.push_back(c);
            }
            MatchOptions opts;
            opts.default_radius = cfg.reid_radius_px;
            opts.hamming_threshold = cfg.reid_hamming_threshold;
            opts.unique_candidates = true;   // one dormant id, one corner
            opts.second_best_margin = cfg.reid_second_best_margin;

            auto matches = SpatialDescriptorMatch(queries, candidates, opts);
            res.reids_attempted = static_cast<int>(queries.size());

            // Apply under a strict budget: resurrections + fresh spawns
            // <= deficit. Resurrections FIRST — identity is worth more
            // than an anonymous corner; a corner can be re-spawned next
            // frame, an expiring dormant identity cannot.
            int budget = deficit;
            for (std::size_t i = 0; i < matches.size(); ++i) {
                if (!matches[i].has_value() || budget <= 0) continue;
                const DormantTrack& dt =
                    all_dormant[matches[i]->candidate_index];
                // Resurrect at the track's ORIGINAL octave: identity
                // includes scale. (In the PoC every octave was 0, so
                // this is behaviourally identical in that regime.)
                spawn_track(q_xy[i].x, q_xy[i].y, q_desc[i], dt.octave,
                            dt.id, true, dt.age_at_death, &dt.descriptor);
                dormant_buffer_.remove(dt.id);
                res.resurrected_ids.push_back(dt.id);
                ++res.reids_succeeded;
                --budget;
            }
            for (std::size_t i = 0; i < matches.size(); ++i) {
                if (matches[i].has_value() ||
                    static_cast<int>(i) >= n_global || budget <= 0)
                    continue;
                spawn_track(q_xy[i].x, q_xy[i].y, q_desc[i], q_oct[i],
                            0, false, 0, nullptr);
                --budget;
            }
        } else {
            for (int i = 0; i < static_cast<int>(kept.size()); ++i)
                spawn_track(kept[static_cast<std::size_t>(i)].pt.x,
                            kept[static_cast<std::size_t>(i)].pt.y,
                            to_desc(desc.row(i)),
                            kept[static_cast<std::size_t>(i)].octave,
                            0, false, 0, nullptr);
        }
    }

    prev_gray_ = curr_gray.clone();
    pyr_prev_.swap(pyr_curr_);   // pyr_prev_ now matches prev_gray_
    res.tracks_out = static_cast<int>(active_tracks_.size());
    res.dormant_buffer_size = dormant_buffer_.size();
    res.median_flow_dx = flow_dx;
    res.median_flow_dy = flow_dy;
    res.octave_histogram.assign(
        static_cast<std::size_t>(std::max(1, cfg.descriptor_levels)), 0);
    for (const auto& kv : active_tracks_) {
        const int o = kv.second.octave;
        if (o >= 0 && o < static_cast<int>(res.octave_histogram.size()))
            ++res.octave_histogram[static_cast<std::size_t>(o)];
    }
    res.ms_reid = ms_since(t_mark);
    res.ms_total = ms_since(t_start);
    return res;
}

std::vector<DeathRecord> HybridFrontend::force_kill(
        const std::vector<std::uint64_t>& ids) {
    std::vector<std::uint64_t> victims;
    std::vector<cv::Point2f> pos;
    std::vector<int> oct;
    for (std::uint64_t id : ids) {
        auto it = active_tracks_.find(id);
        if (it == active_tracks_.end()) continue;
        victims.push_back(id);
        pos.emplace_back(it->second.x, it->second.y);
        oct.push_back(it->second.octave);
    }
    auto death_desc = descriptors_at_positions(
        prev_gray_, pos, oct, orb_, cfg_.descriptor_patch_size,
        cfg_.descriptor_scale_factor, pyr_prev_, umax_);
    std::vector<DeathRecord> killed;
    for (std::size_t k = 0; k < victims.size(); ++k) {
        const std::uint64_t tid = victims[k];
        ActiveTrack t = active_tracks_.at(tid);
        active_tracks_.erase(tid);
        active_order_.erase(
            std::find(active_order_.begin(), active_order_.end(), tid));
        killed.push_back({tid, t.x, t.y});
        Descriptor256 stored = t.birth_descriptor;
        const auto it = death_desc.find(static_cast<int>(k));
        if (it != death_desc.end()) stored = it->second;
        if (cfg_.use_representative_descriptor && t.has_representative)
            stored = t.representative_descriptor;
        DormantTrack d;
        d.id = tid;
        d.last_x = t.x;
        d.last_y = t.y;
        d.descriptor = stored;
        d.frame_died = frame_index_;
        d.octave = t.octave;
        d.age_at_death = t.age;
        d.map_point = nullptr;
        dormant_buffer_.add(d);
    }
    return killed;
}

void HybridFrontend::describe_current(std::vector<std::uint64_t>& ids,
                                      std::vector<cv::KeyPoint>& keypoints,
                                      cv::Mat& descriptors) {
    ids.clear();
    keypoints.clear();
    descriptors.release();
    if (!initialized_ || active_tracks_.empty()) return;

    // Deterministic order: ascending id.
    std::vector<std::uint64_t> order;
    order.reserve(active_tracks_.size());
    for (const auto& kv : active_tracks_) order.push_back(kv.first);
    std::sort(order.begin(), order.end());

    // Fresh descriptors at CURRENT positions in the LAST PROCESSED frame
    // (prev_gray_ / pyr_prev_ after process_frame has run). This is what
    // stock ORB-SLAM3 does every frame; a birth-time or medoid
    // descriptor at a position the track occupied frames ago is the
    // wrong thing to hand to stereo matching or the local map.
    std::vector<cv::KeyPoint> kps;
    kps.reserve(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        const ActiveTrack& t = active_tracks_.at(order[i]);
        if (t.x < 0.0f || t.y < 0.0f ||
            t.x >= static_cast<float>(prev_gray_.cols) ||
            t.y >= static_cast<float>(prev_gray_.rows))
            continue;
        cv::KeyPoint kp(t.x, t.y,
                        static_cast<float>(cfg_.descriptor_patch_size) *
                            std::pow(cfg_.descriptor_scale_factor,
                                     static_cast<float>(t.octave)));
        kp.octave = t.octave;
        kp.class_id = static_cast<int>(i);
        kps.push_back(kp);
    }
    assign_ic_angles(kps, pyr_prev_, cfg_.descriptor_scale_factor, umax_);
    cv::Mat desc;
    orb_->compute(prev_gray_, kps, desc);   // drops border keypoints
    if (desc.empty()) return;

    ids.reserve(kps.size());
    keypoints.reserve(kps.size());
    for (std::size_t r = 0; r < kps.size(); ++r) {
        const int src = kps[r].class_id;
        if (src < 0 || src >= static_cast<int>(order.size())) continue;
        ids.push_back(order[static_cast<std::size_t>(src)]);
        keypoints.push_back(kps[r]);
    }
    descriptors = desc.clone();
}

bool HybridFrontend::set_map_point(std::uint64_t id, MapPointHandle mp) {
    auto it = active_tracks_.find(id);
    if (it == active_tracks_.end()) return false;
    it->second.map_point = mp;
    return true;
}

std::uint64_t HybridFrontend::spawn_track(
        float x, float y, const Descriptor256& desc, int octave,
        std::uint64_t forced_id, bool has_forced_id, std::uint32_t age,
        const Descriptor256* seed_history) {
    std::uint64_t id;
    if (!has_forced_id) {
        id = next_id_++;
    } else {
        id = forced_id;
        next_id_ = std::max(next_id_, id + 1);
    }
    assert(active_tracks_.find(id) == active_tracks_.end() &&
           "Duplicate active track id");
    ActiveTrack t;
    t.id = id;
    t.x = x;
    t.y = y;
    t.birth_descriptor = desc;
    t.octave = octave;
    t.age = age;
    if (cfg_.use_representative_descriptor) {
        // Seed the observation set; on resurrection, seed_history
        // carries the dormant entry's representative across the gap.
        t.descriptor_history.push_back(desc);
        if (seed_history != nullptr)
            t.descriptor_history.push_back(*seed_history);
        t.representative_descriptor =
            representative_descriptor(t.descriptor_history);
        t.has_representative = true;
    }
    active_tracks_.emplace(id, std::move(t));
    active_order_.push_back(id);
    return id;
}

}  // namespace hybrid_frontend
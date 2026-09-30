/**
 * TrackingHybrid.cc — Stage B (§9.2 step 5, takeover).
 *
 * Two Tracking members, kept in their own translation unit so Tracking.cc
 * needs only the small edits listed in STAGE_B_EDITS.md:
 *
 *   HybridFrameInputs()          — turn the frontend's active tracks into
 *                                  the (keypoints, descriptors, ids) the
 *                                  hybrid Frame constructor consumes.
 *   HybridTrackWithMotionModel() — TrackWithMotionModel with the windowed
 *                                  descriptor search replaced by track-ID
 *                                  lookup against mLastFrame (§4: associate
 *                                  by tracking, not by matching). Falls
 *                                  back to stock SearchByProjection when
 *                                  ID matches are scarce. Pose optimisation
 *                                  and outlier discard are verbatim stock.
 */

 #include "Tracking.h"
 #include "ORBmatcher.h"
 #include "Optimizer.h"
 #include "HybridFrontend.h"
 
 #include <algorithm>
 #include <cmath>
 #include <cstring>
 #include <unordered_map>
 
 using namespace std;
 
 namespace ORB_SLAM3
 {
 
 void Tracking::HybridFrameInputs(std::vector<cv::KeyPoint> &vKeys,
                                  cv::Mat &descriptors,
                                  std::vector<std::uint64_t> &vTrackIds) const
 {
     vKeys.clear();
     vTrackIds.clear();
     descriptors.release();
     if(!mpHybridFrontend)
         return;
 
     // FRESH steered descriptors at every track's CURRENT position, with
     // kp.angle / octave / size set by the frontend. Two reasons this is
     // not "pull the stored birth/representative descriptor":
     //  1. Those were computed at positions the track occupied frames
     //     ago; stereo matching and the local-map search want this
     //     frame's appearance, which is what stock re-extraction gives.
     //  2. Steering. cv::ORB::compute with user keypoints does not
     //     compute orientation; the frontend now computes the IC angle
     //     itself, so these descriptors are comparable with
     //     ORBextractor's on the right image.
     // Tracks whose patch falls outside the image are omitted from the
     // Frame this frame (they stay alive in the frontend) - which is also
     // what keeps ComputeStereoMatches' unchecked row index in bounds.
     mpHybridFrontend->describe_current(vTrackIds, vKeys, descriptors);
 }
 
 bool Tracking::HybridTrackWithMotionModel()
 {
     // Update last frame pose according to its reference keyframe
     // Create "visual odometry" points if in Localization Mode
     UpdateLastFrame();
 
     if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
     {
         // Predict state with IMU if it is initialized and it doesnt need reset
         PredictStateIMU();
         return true;
     }
     else
     {
         mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
     }
 
     fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
 
     // ---- HYBRID: association by track id, not by windowed search. ----
     // A keypoint in this frame carries the SAME id as the keypoint KLT
     // tracked it from in the last frame, so map-point continuity is a
     // hash lookup. No descriptor distance, no projection window, no
     // descriptor drift to fail on — that is the §4 payoff.
     std::unordered_map<std::uint64_t,int> lastIdx;
     lastIdx.reserve(mLastFrame.mvnTrackIds.size());
     for(int j=0; j<mLastFrame.N && j<(int)mLastFrame.mvnTrackIds.size(); ++j)
     {
         const std::uint64_t id = mLastFrame.mvnTrackIds[j];
         if(id!=0)
             lastIdx[id] = j;
     }
 
     int nmatches = 0;
     for(int i=0; i<mCurrentFrame.N && i<(int)mCurrentFrame.mvnTrackIds.size(); ++i)
     {
         const std::uint64_t id = mCurrentFrame.mvnTrackIds[i];
         if(id==0)
             continue;
         auto it = lastIdx.find(id);
         if(it==lastIdx.end())
             continue;
         MapPoint* pMP = mLastFrame.mvpMapPoints[it->second];
         if(!pMP || pMP->isBad() || mLastFrame.mvbOutlier[it->second])
             continue;
         mCurrentFrame.mvpMapPoints[i] = pMP;
         ++nmatches;
     }
     mnHybridIdMatches = nmatches;   // diagnostics: how much the ids bought
 
     // Fallback: too few id matches (fresh map, mass track death, first
     // frame after a reset). Stock projection search still works because
     // the Frame carries ordinary keypoints and descriptors.
     if(nmatches<20)
     {
         Verbose::PrintMess("HYBRID: few id matches (" + to_string(nmatches) + "), falling back to projection search", Verbose::VERBOSITY_NORMAL);
         ORBmatcher matcher(0.9,true);
         fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
         int th;
         if(mSensor==System::STEREO)
             th=7;
         else
             th=15;
         nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
         if(nmatches<20)
         {
             fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
             nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
         }
     }
     // -------------------------------------------------------------------
 
     if(nmatches<20)
     {
         Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
         if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
             return true;
         else
             return false;
     }
 
     // Optimize frame pose with all matches
     Optimizer::PoseOptimization(&mCurrentFrame);
 
     // Discard outliers  (verbatim stock)
     int nmatchesMap = 0;
     for(int i =0; i<mCurrentFrame.N; i++)
     {
         if(mCurrentFrame.mvpMapPoints[i])
         {
             if(mCurrentFrame.mvbOutlier[i])
             {
                 MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
 
                 mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                 mCurrentFrame.mvbOutlier[i]=false;
                 if(i < mCurrentFrame.Nleft){
                     pMP->mbTrackInView = false;
                 }
                 else{
                     pMP->mbTrackInViewR = false;
                 }
                 pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                 nmatches--;
             }
             else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                 nmatchesMap++;
         }
     }
 
     if(mbOnlyTracking)
     {
         mbVO = nmatchesMap<10;
         return nmatches>20;
     }
 
     if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
         return true;
     else
         return nmatchesMap>=10;
 }
 
 // =====================================================================
 // Step 6 (§9.2) = §4.1 adult/infant bookkeeping + §4.7 Step 5b
 // =====================================================================
 
 void Tracking::SyncHybridMapPoints()
 {
     // §4.1: tell the frontend which tracks are ADULTS (hold a landmark).
     // Call at the end of Track() once mvbOutlier is final: for STEREO,
     // TrackLocalMap already nulls outlier map points, so mvpMapPoints is
     // the truth. A track whose landmark was culled between frames reads
     // back as nullptr here and reverts to infant automatically - no
     // separate culling hook is needed.
     if(!mpHybridFrontend)
         return;
     const int N = mCurrentFrame.N;
     for(int i=0; i<N && i<(int)mCurrentFrame.mvnTrackIds.size(); ++i)
     {
         const std::uint64_t id = mCurrentFrame.mvnTrackIds[i];
         if(id==0)
             continue;
         MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
         if(pMP && (pMP->isBad() || mCurrentFrame.mvbOutlier[i]))
             pMP = nullptr;
         mpHybridFrontend->set_map_point(id, static_cast<void*>(pMP));
     }
 }
 
 void Tracking::ClearHybridMapPoints()
 {
     // Call from Reset(), ResetActiveMap() and CreateMapInAtlas(). The
     // MapPoints those handles refer to are being freed; every track
     // becomes an infant again.
     if(mpHybridFrontend)
         mpHybridFrontend->clear_all_map_points();
 }
 
 void Tracking::HybridSearchLocalPoints()
 {
     // §4.7 Step 5b. Identical to stock SearchLocalPoints up to the
     // matcher call, which becomes a SpatialDescriptorMatch:
     //   queries    = current-frame keypoints with NO map point (S_k^unmatched),
     //                radius r_TLM(ℓ) = tlm_radius_px * th * 1.2^ℓ
     //   candidates = local map points in frustum, at their projection,
     //                with the REPRESENTATIVE descriptor (MapPoint::GetDescriptor
     //                is ORB-SLAM3's ComputeDistinctiveDescriptors medoid - §5.4)
     //   gates      = θ_TLM, optional δ, unique candidates
     // An accepted match binds the existing KLT track to the landmark:
     // "adult re-acquired" in the §4.1 lifecycle. SyncHybridMapPoints
     // then tells the frontend.
     for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
     {
         MapPoint* pMP = *vit;
         if(pMP)
         {
             if(pMP->isBad())
             {
                 *vit = static_cast<MapPoint*>(NULL);
             }
             else
             {
                 pMP->IncreaseVisible();
                 pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                 pMP->mbTrackInView = false;
                 pMP->mbTrackInViewR = false;
             }
         }
     }
 
     int nToMatch=0;
     for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
     {
         MapPoint* pMP = *vit;
         if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
             continue;
         if(pMP->isBad())
             continue;
         if(mCurrentFrame.isInFrustum(pMP,0.5))
         {
             pMP->IncreaseVisible();
             nToMatch++;
         }
         if(pMP->mbTrackInView)
         {
             mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
         }
     }
 
     mnHybridTlmMatches = 0;
     if(nToMatch<=0)
         return;
 
     // Same search-width policy as stock (th multiplies the radius).
     int th = 1;
     if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
         th=3;
     if(mpAtlas->isImuInitialized())
     {
         if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
             th=2;
         else
             th=6;
     }
     else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
     {
         th=10;
     }
     if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
         th=5;
     if(mState==LOST || mState==RECENTLY_LOST)
         th=15;
 
     const hybrid_frontend::HybridConfig &hc = mpHybridFrontend->config();
     const bool bFarPoints = mpLocalMapper->mbFarPoints;
     const float thFarPoints = mpLocalMapper->mThFarPoints;
 
     // ---- candidates: projected local map points -------------------
     std::vector<hybrid_frontend::PixelCandidate> cands;
     std::vector<MapPoint*> candMP;
     cands.reserve(mvpLocalMapPoints.size());
     candMP.reserve(mvpLocalMapPoints.size());
     for(MapPoint* pMP : mvpLocalMapPoints)
     {
         if(!pMP || pMP->isBad() || !pMP->mbTrackInView)
             continue;
         if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
             continue;                              // already matched this frame
         if(bFarPoints && pMP->mTrackDepth>thFarPoints)
             continue;
         const cv::Mat d = pMP->GetDescriptor();
         if(d.empty() || d.cols!=32)
             continue;
         hybrid_frontend::PixelCandidate c;
         c.x = pMP->mTrackProjX;
         c.y = pMP->mTrackProjY;
         std::memcpy(c.descriptor.data(), d.ptr<uchar>(0), 32);
         cands.push_back(c);
         candMP.push_back(pMP);
     }
     if(cands.empty())
         return;
 
     // ---- queries: unmatched current-frame keypoints ---------------
     std::vector<hybrid_frontend::PixelQuery> queries;
     std::vector<int> queryIdx;
     queries.reserve(mCurrentFrame.N);
     queryIdx.reserve(mCurrentFrame.N);
     for(int i=0; i<mCurrentFrame.N; ++i)
     {
         if(mCurrentFrame.mvpMapPoints[i])
             continue;                              // S_k^unmatched only
         const cv::KeyPoint &kp = mCurrentFrame.mvKeysUn[i];
         hybrid_frontend::PixelQuery q;
         q.x = kp.pt.x;
         q.y = kp.pt.y;
         std::memcpy(q.descriptor.data(), mCurrentFrame.mDescriptors.ptr<uchar>(i), 32);
         // r_TLM(ℓ): per-query radius grows geometrically with the
         // keypoint's octave (§8), widened by th exactly as stock widens
         // its projection window.
         q.radius = hc.tlm_radius_px * static_cast<float>(th) *
                    mCurrentFrame.mvScaleFactors[kp.octave];
         queries.push_back(q);
         queryIdx.push_back(i);
     }
     if(queries.empty())
         return;
 
     hybrid_frontend::MatchOptions opts;
     opts.default_radius = hc.tlm_radius_px * static_cast<float>(th);
     opts.hamming_threshold = hc.tlm_hamming_threshold;   // θ_TLM
     opts.second_best_margin = hc.tlm_second_best_margin;  // δ
     opts.unique_candidates = true;                         // §4.8
 
     const auto matches = hybrid_frontend::SpatialDescriptorMatch(queries, cands, opts);
 
     int nmatches = 0;
     for(size_t qi=0; qi<matches.size(); ++qi)
     {
         if(!matches[qi].has_value())
             continue;
         MapPoint* pMP = candMP[matches[qi]->candidate_index];
         const int i = queryIdx[qi];
         if(hc.tlm_octave_gate)
         {
             // Stock searches levels [predicted-1, predicted]. The
             // primitive has no octave notion, so apply it as a
             // post-filter; a rejected match just leaves its candidate
             // unconsumed this frame.
             const int pred = pMP->mnTrackScaleLevel;
             const int oct  = mCurrentFrame.mvKeysUn[i].octave;
             if(oct < pred-1 || oct > pred+1)
                 continue;
         }
         mCurrentFrame.mvpMapPoints[i] = pMP;
         ++nmatches;
     }
     mnHybridTlmMatches = nmatches;
 }
 
 } //namespace ORB_SLAM3
/**
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
 
 } //namespace ORB_SLAM3
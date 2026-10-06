/**
 * Drop-in alternative to ORBmatcher::SearchByBoW(KeyFrame*, KeyFrame*, ...)
 * for loop-closure / merge verification. Identical contract and identical
 * acceptance logic — only map-pointed keypoints on both sides, TH_LOW,
 * nearest-neighbour ratio, one-to-one assignment, rotation-histogram
 * check — with ONE difference: every keypoint of KF1 is compared against
 * every valid keypoint of KF2, instead of only those sharing a vocabulary
 * node (mFeatVec). If loop closure converges with this and not with
 * SearchByBoW, the offline vocabulary's node grouping was splitting true
 * correspondences apart (OV2SLAM verifies candidates the same way).
 *
 * Cost: |valid KF1| x |valid KF2| Hamming distances, ~1M for two 1000-point
 * keyframes, roughly 10-20 ms with the scalar DescriptorDistance. It runs
 * on the asynchronous loop-closing thread. If that thread falls behind,
 * cv::BFMatcher(NORM_HAMMING) with knnMatch(k=2) is the SIMD alternative.
 *
 */

 #include "ORBmatcher.h"
 #include "KeyFrame.h"
 #include "MapPoint.h"
 
 #include <cassert>
 #include <cmath>
 
 using namespace std;
 
 namespace ORB_SLAM3
 {
 
 int ORBmatcher::SearchBruteForce(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12)
 {
     const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
     const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
     const cv::Mat &Descriptors1 = pKF1->mDescriptors;
 
     const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
     const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
     const cv::Mat &Descriptors2 = pKF2->mDescriptors;
 
     vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(NULL));
     vector<bool> vbMatched2(vpMapPoints2.size(), false);
 
     // Valid KF2 side computed once: keypoints holding a live map point.
     vector<size_t> vValid2;
     vValid2.reserve(vpMapPoints2.size());
     for(size_t idx2 = 0; idx2 < vpMapPoints2.size(); ++idx2)
     {
         MapPoint* pMP2 = vpMapPoints2[idx2];
         if(pMP2 && !pMP2->isBad())
             vValid2.push_back(idx2);
     }
     if(vValid2.empty())
         return 0;
 
     vector<int> rotHist[HISTO_LENGTH];
     for(int i = 0; i < HISTO_LENGTH; i++)
         rotHist[i].reserve(500);
     const float factor = 1.0f / HISTO_LENGTH;
 
     int nmatches = 0;
 
     for(size_t idx1 = 0; idx1 < vpMapPoints1.size(); ++idx1)
     {
         MapPoint* pMP1 = vpMapPoints1[idx1];
         if(!pMP1 || pMP1->isBad())
             continue;
 
         const cv::Mat &d1 = Descriptors1.row(static_cast<int>(idx1));
 
         int bestDist1 = 256;
         int bestDist2 = 256;
         int bestIdx2  = -1;
 
         for(size_t k = 0; k < vValid2.size(); ++k)
         {
             const size_t idx2 = vValid2[k];
             if(vbMatched2[idx2])
                 continue;                       // one-to-one, greedy in idx1 order (as stock)
 
             const int dist = DescriptorDistance(d1, Descriptors2.row(static_cast<int>(idx2)));
             if(dist < bestDist1)
             {
                 bestDist2 = bestDist1;
                 bestDist1 = dist;
                 bestIdx2  = static_cast<int>(idx2);
             }
             else if(dist < bestDist2)
             {
                 bestDist2 = dist;
             }
         }
 
         if(bestIdx2 >= 0 && bestDist1 < TH_LOW &&
            static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
         {
             vpMatches12[idx1] = vpMapPoints2[bestIdx2];
             vbMatched2[bestIdx2] = true;
 
             if(mbCheckOrientation)
             {
                 float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                 if(rot < 0.0f)
                     rot += 360.0f;
                 int bin = static_cast<int>(round(rot * factor));
                 if(bin == HISTO_LENGTH)
                     bin = 0;
                 assert(bin >= 0 && bin < HISTO_LENGTH);
                 rotHist[bin].push_back(static_cast<int>(idx1));
             }
             nmatches++;
         }
     }
 
     if(mbCheckOrientation)
     {
         int ind1 = -1, ind2 = -1, ind3 = -1;
         ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
         for(int i = 0; i < HISTO_LENGTH; i++)
         {
             if(i == ind1 || i == ind2 || i == ind3)
                 continue;
             for(size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
             {
                 vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                 nmatches--;
             }
         }
     }
 
     return nmatches;
 }
 
 } // namespace ORB_SLAM3
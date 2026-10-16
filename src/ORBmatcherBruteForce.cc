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
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

 #ifndef TIMING_H
 #define TIMING_H
 
 #include <chrono>
 
 namespace ORB_SLAM3 {
 
     // Per-stage run timings, printed as a table by System::Shutdown().
     //
     // What each stage measures, and what one sample is:
     //
     //   FRONT_END       one System::TrackStereo / TrackRGBD / TrackMonocular call,
     //                   start to finish (one sample per frame)
     //   KF_CREATION     the part of Tracking::CreateNewKeyFrame() that builds and
     //                   inserts the keyframe (one sample per keyframe created)
     //   MAPPING_THREAD  one pass of LocalMapping::Run() over a keyframe, local BA
     //                   included (one sample per keyframe processed)
     //   LOCAL_BA        one Optimizer::LocalBundleAdjustment / LocalInertialBA call
     //   LC_DETECTION    one LoopClosing::NewDetectCommonRegions() call; nothing is
     //                   recorded when loop closing is switched off
     //
     // Recording costs two clock reads and one vector push_back per sample.
     class Timing {
     public:
         enum eStage {
             FRONT_END = 0,
             KF_CREATION = 1,
             MAPPING_THREAD = 2,
             LOCAL_BA = 3,
             LC_DETECTION = 4,
             NUM_STAGES = 5
         };
 
         typedef std::chrono::steady_clock::time_point TimePoint;
 
         static TimePoint Now() { return std::chrono::steady_clock::now(); }
 
         static double MsSince(const TimePoint& tStart)
         {
             return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(Now() - tStart).count();
         }
 
         // Record one sample, in milliseconds. Can be called from any thread.
         static void Add(eStage stage, double ms);
 
         // Print the table and the frame rate to cout.
         static void PrintSummary();
 
         // Times from its construction to the end of the enclosing scope:
         //     Timing::Scope timer(Timing::LOCAL_BA);
         class Scope {
         public:
             explicit Scope(eStage stage) : mStage(stage), mStart(Now()) {}
             ~Scope() { Add(mStage, MsSince(mStart)); }
         private:
             Scope(const Scope&);
             Scope& operator=(const Scope&);
 
             eStage mStage;
             TimePoint mStart;
         };
     };
 
 } // namespace ORB_SLAM3
 
 #endif // TIMING_H
 
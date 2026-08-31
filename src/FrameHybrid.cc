/**
 * A Frame constructor that takes the LEFT features from the hybrid
 * frontend instead of running ORBextractor on the left image. Cloned
 * from the stereo constructor in Frame.cc with exactly one substitution:
 * the left-image ExtractORB call is replaced by assigning the supplied
 * keypoints / descriptors / track ids and building the left pyramid
 * (ComputeStereoMatches reads mpORBextractorLeft->mvImagePyramid for
 * sub-pixel refinement, so the pyramid must exist even though detection
 * is skipped). Right-image extraction, undistortion, stereo matching,
 * grid assignment and every Frame invariant are unchanged, so the rest
 * of the system cannot tell where the left features came from.
 *
 * Lives in its own translation unit so Frame.cc needs no edit beyond the
 * copy-constructor member (see STAGE_B_EDITS.md). The root CMake GLOB
 * picks this file up automatically.
 *
 * Non-inertial STEREO only. The IMU_STEREO variant is this constructor
 * plus the (Frame* pPrevF, const IMU::Calib&) arguments and the
 * mImuCalib / mpPrevFrame initialisers from the stock IMU constructor.
 */

 #include "Frame.h"
 #include "ORBextractor.h"
 #include "Converter.h"
 
 #include <thread>
 
 namespace ORB_SLAM3
 {
 
 Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight,
              const std::vector<cv::KeyPoint> &vHybridKeys,
              const cv::Mat &hybridDescriptors,
              const std::vector<std::uint64_t> &vHybridTrackIds,
              const double &timeStamp,
              ORBextractor* extractorLeft, ORBextractor* extractorRight,
              ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef,
              const float &bf, const float &thDepth, GeometricCamera* pCamera)
     :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
      mImuCalib(IMU::Calib()), mpImuPreintegrated(NULL), mpPrevFrame(NULL),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
      mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
 {
     // Frame ID
     mnId=nNextId++;
 
     // Scale Level Info — from the extractor, so mvInvLevelSigma2[octave]
     // is indexed by the SAME pyramid the frontend detected on. This is
     // why HybridConfig::descriptor_scale_factor / descriptor_levels must
     // equal ORBextractor.scaleFactor / nLevels.
     mnScaleLevels = mpORBextractorLeft->GetLevels();
     mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
     mfLogScaleFactor = log(mfScaleFactor);
     mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
     mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
     mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
     mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
 
 #ifdef REGISTER_TIMES
     std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
 #endif
     // ---- HYBRID: left features come from the frontend. ----------------
     // The left pyramid is still needed by ComputeStereoMatches (sub-pixel
     // refinement reads mvImagePyramid[kpL.octave]); building it is ~3 ms
     // vs ~60 ms for full extraction, which is the Stage B cost saving.
     // ORBextractor::ComputePyramid must be made public (STAGE_B_EDITS.md).
     thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
     mpORBextractorLeft->ComputePyramid(imLeft);
     mvKeys = vHybridKeys;
     mDescriptors = hybridDescriptors.clone();
     mvnTrackIds = vHybridTrackIds;
     threadRight.join();
     // -------------------------------------------------------------------
 #ifdef REGISTER_TIMES
     std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
 
     mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
 #endif
 
     N = mvKeys.size();
     if(mvKeys.empty())
         return;
 
     UndistortKeyPoints();
 
 #ifdef REGISTER_TIMES
     std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
 #endif
     ComputeStereoMatches();
 #ifdef REGISTER_TIMES
     std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();
 
     mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
 #endif
 
     mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
     mvbOutlier = vector<bool>(N,false);
     mmProjectPoints.clear();
     mmMatchedInImage.clear();
 
     // This is done only for the first Frame (or after a change in the calibration)
     if(mbInitialComputations)
     {
         ComputeImageBounds(imLeft);
 
         mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
         mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
 
         fx = K.at<float>(0,0);
         fy = K.at<float>(1,1);
         cx = K.at<float>(0,2);
         cy = K.at<float>(1,2);
         invfx = 1.0f/fx;
         invfy = 1.0f/fy;
 
         mbInitialComputations=false;
     }
 
     mb = mbf/fx;
 
     mVw.setZero();
 
     mpMutexImu = new std::mutex();
 
     //Set no stereo fisheye information
     Nleft = -1;
     Nright = -1;
     mvLeftToRightMatch = vector<int>(0);
     mvRightToLeftMatch = vector<int>(0);
     mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
     monoLeft = -1;
     monoRight = -1;
 
     AssignFeaturesToGrid();
 }
 
 } //namespace ORB_SLAM3
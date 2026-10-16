# ORB-SLAM3 with a Hybrid KLT–Descriptor Frontend for Turbid Water

A fork of [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), via
[ORB-SLAM3-STEREO-FIXED](https://github.com/zang09/ORB-SLAM3-STEREO-FIXED),
that replaces the stock FAST–ORB frontend with a hybrid frontend built for
low-contrast, self-similar underwater imagery. It was developed for a stereo
inspection ROV with CPU-only compute and evaluated in a Godot/ROS 2 simulator
with a turbidity model.

Everything behind the frontend, including local mapping, bundle adjustment,
the Atlas and loop closing, is stock ORB-SLAM3. Every modification is behind a
configuration key and defaults to off, so an unmodified stock YAML file still
runs stock ORB-SLAM3.

## Link to dataset

https://drive.google.com/file/d/1I3L1O4M5zX-BSWuV8do-dNWiqUMOxStK/view?usp=sharing

Uncompress using 

```bash
tar -xf Dataset_Thesis.tar.gz
```

Datasets are saved as ros bags, use foxglove or

```bash
ros2 run rqt_image_view rqt_image_view
```

## What is modified

**Detector.** Multi-octave Shi–Tomasi detection with rank selection replaces
FAST inside `ORBextractor`. Each pyramid level returns its quota of strongest
corners by the minimum eigenvalue of the structure tensor, with no absolute
threshold, so the detector keeps delivering features as contrast falls.
Orientation, steered-BRIEF description and everything downstream are unchanged.

**Hybrid frontend** (`HybridFrontend`, an OpenCV-only library). Features are
tracked frame to frame by pyramidal Lucas–Kanade optical flow with
forward–backward and RANSAC verification, and carry a persistent identity.
Dead tracks are replaced by targeted detection; tracks that die before being
triangulated are held in a dormant buffer and re-identified by descriptor match
against new detections, with a Hamming ceiling that widens with the gap.

**Tracking integration.** A frame constructor that adopts the frontend's
keypoints, descriptors and track IDs; frame-to-frame association by identity
with a fallback to the stock projection search when fewer than 20 matches are
found; a spatial local-map matcher (Step 5b) for long-lived tracks; and a
lifecycle sync so that a track holding a map point is discarded on death
rather than buffered, which prevents re-triangulation of existing landmarks.

**Loop closing.** Restores the upstream statement that selects the covisible
keyframe with the most BoW matches (commented out in the release because it
does not compile); exposes the Sim(3) inlier threshold; adds an optional
brute-force matcher that bypasses vocabulary-node restriction; adds per-candidate
diagnostics (`[LC]` log lines); and can dump keyframe descriptors for offline
vocabulary training.

**Build.** C++14 → C++17. Tested on Ubuntu 24.04, ROS 2 Jazzy, OpenCV 4.x.

## Repository layout

```
include/HybridFrontend.h            Steps 1–5 engine (KLT, detection, re-ID)
include/SpatialDescriptorMatcher.h  shared matching primitive (Steps 5 and 5b)
include/DormantTrackBuffer.h        dormant-track buffer
src/HybridFrontend.cc
src/SpatialDescriptorMatcher.cc
src/DormantTrackBuffer.cc
src/FrameHybrid.cc                  Frame constructor taking frontend output
src/TrackingHybrid.cc               identity association, Step 5b, lifecycle sync
src/ORBextractor.cc                 Shi–Tomasi path beside stock FAST
src/ORBmatcherBruteForce.cc         brute-force loop-closing matcher
src/LoopClosing.cc                  upstream fix, Sim3 threshold, diagnostics
Vocabulary/train_vocab.cc           offline DBoW2 vocabulary training
Tests/                              unit tests (GoogleTest) and Python parity dump
Examples/ROS2/...                   stereo ROS 2 node
```

## Building

Install the dependencies of upstream ORB-SLAM3 (Eigen3, Pangolin, OpenCV, and
the bundled DBoW2, g2o and Sophus), then:

```bash
git clone <this repository> ORB_SLAM3
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

This builds `lib/libORB_SLAM3.so` and the executables in `Examples/`. Sophus is
built with `-DBUILD_TESTS=OFF`. The ROS 2 node is built separately in its
workspace and loads the shared library at run time.

Unit tests build as a separate project that compiles the frontend sources
directly, without linking ORB-SLAM3:

```bash
cd Tests && mkdir build && cd build
cmake .. && make
./test_HybridFrontend && ./test_DormantTrackBuffer && ./test_SpatialDescriptorMatcher
```

## Running

Stereo, from a ROS 2 bag of the simulator's left/right topics:

```bash
ros2 run <package> stereo <vocabulary.txt> <config.yaml>
ros2 bag play <sequence> --rate 1.0
```

At rate 1.0 the recorded 28.6 Hz exceeds what either frontend processes, so
frames are dropped and the fraction processed is reported at shutdown. A rate
of about 0.35 processes every frame with the hybrid frontend.

At shutdown the node writes `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`
(TUM format) and prints a timing table (front-end, keyframe creation, mapping
thread, local BA; mean, median, std, call count; processing and wall-clock
FPS). The console log carries one `[Hybrid]` line per frame from the frontend
and one `[HybridTrack]` line from the tracking thread; see *Diagnostics*.

## Configuration

All keys live in the run's YAML file next to the stock camera and ORBextractor
keys. Absent keys take the defaults shown. The start-up settings dump prints
every value in use; check it, since a mistyped key is silently ignored.

### Detector

| Key | Default | Meaning |
|---|---|---|
| `ORBextractor.useShiTomasi` | 1 | 1 = Shi–Tomasi rank selection, 0 = stock FAST. `iniThFAST`/`minThFAST` are ignored when 1. |
| `Hybrid.multiscale` | 1 | Detect on every pyramid level with per-level quotas (rank rule); 0 = level 0 only. |

### Frontend mode

| Key | Default | Meaning |
|---|---|---|
| `Hybrid.shadowMode` | 0 | Run the frontend beside stock tracking and log it; nothing reaches the pose. |
| `Hybrid.takeover` | 0 | Frontend supplies the frame's left features and frame-to-frame association. |
| `Hybrid.enableReid` | 1 | Re-identify dormant tracks (Step 5). 0 = dead tracks are discarded. |

The target track count and descriptor pyramid are taken from
`ORBextractor.nFeatures`, `ORBextractor.scaleFactor` and `ORBextractor.nLevels`
so the frontend and extractor cannot disagree.

### Optical flow (Steps 1–3)

| Key | Default | Meaning |
|---|---|---|
| `Hybrid.kltWindow` | 21 | Lucas–Kanade window side, px |
| `Hybrid.kltLevels` | 3 | Lucas–Kanade pyramid levels |
| `Hybrid.kltMinEig` | 1.0e-4 | `minEigThreshold` passed to `calcOpticalFlowPyrLK`. OpenCV's default binds hard on noise-free low-contrast imagery; 1.0e-5 recovered most tracks on the turbid test sequence. Must be written as a real number. |
| `Hybrid.fbThreshold` | 1.0 | Forward–backward error threshold, px |
| `Hybrid.ransacPx` | 3.0 | Fundamental-matrix RANSAC threshold, px |

### Detection and lifecycle (Steps 4–5)

| Key | Default | Meaning |
|---|---|---|
| `Hybrid.maskRadius` | 10 | Occupancy suppression radius around live tracks, px |
| `Hybrid.localQualityScale` | | Response floor multiplier for targeted detection around dormant tracks |
| `Hybrid.minTrackAge` | 3 | Frames a track must survive before it can be buffered |
| `Hybrid.dormantHorizon` | 10 | Frames a dormant track is kept |
| `Hybrid.reidRadius` | 10 | Re-ID search half-width around predicted position, px |
| `Hybrid.thetaBase`, `Hybrid.thetaSlope`, `Hybrid.thetaCap` | 63, 1.2, 86 | Gap-scaled Hamming ceiling θ(g) = min(cap, base + slope·g) |
| `Hybrid.secondBestMargin` | 0 | Distinctiveness margin δ in bits; 0 = off |
| `Hybrid.useRepresentative` | 1 | Accumulate a representative (medoid) descriptor per track |
| `Hybrid.dormantStoreRepresentative` | 0 | Buffer stores the death-time descriptor (0) or the medoid (1) |

### Local-map association (Step 5b)

| Key | Default | Meaning |
|---|---|---|
| `Hybrid.tlmRadius` | 3.0 | Search radius at octave 0, px; scaled by 1.2^ℓ |
| `Hybrid.tlmHamming` | 50 | Hamming ceiling (integer) |
| `Hybrid.tlmMargin` | 0 | Margin δ for Step 5b; 0 = off |
| `Hybrid.tlmOctaveGate` | 1 | Require keypoint octave within 1 of the predicted level, as stock |

### Loop closing

| Key | Default | Meaning |
|---|---|---|
| `loopClosing` | 1 | 0 disables loop detection and map merging (stock key) |
| `LoopClosing.sim3MinInliers` | 15 | Sim(3) RANSAC inlier count required, strictly greater than |
| `LoopClosing.bruteForce` | 0 | Replace the node-restricted BoW matcher with brute-force matching |
| `Vocabulary.dumpDir` | | If set, each keyframe's descriptors are written here as `.desc` files for `train_vocab` |

A reference configuration for the simulator rig is in
`Examples/Stereo/godot_auv_sim.yaml`. `<!-- TODO: path -->`

## Reproducing the thesis configurations

| Configuration | Keys |
|---|---|
| Stock | `useShiTomasi: 0`, `takeover: 0` |
| Stock + CLAHE | as stock, with equalisation applied in the node `<!-- TODO: how it is switched on -->` |
| Detector only | `useShiTomasi: 1`, `takeover: 0` |
| Detector + tracking | `useShiTomasi: 1`, `takeover: 1`, `enableReid: 0` |
| Full hybrid | `useShiTomasi: 1`, `takeover: 1`, `enableReid: 1` |

All thesis tables use `loopClosing: 0` unless stated. The eigenvalue study
varied `Hybrid.kltMinEig` only.

## Diagnostics

Each processed frame prints

```
[Hybrid] f=<n> in=<tracks in> klt=<after KLT> fb=<after FB> ransac=<after RANSAC>
         new=<detected> reid=<succeeded>/<attempted> out=<tracks out> dorm=<buffer size>
         adults=<adult tracks discarded> flow=(dx,dy) ms=<total>(t<track>/d<detect>/r<reid>)
         oct=<per-octave counts> mineig=<F>+<B> left=<n> eigdec=<a>/<b>/<c>
[HybridTrack] f=<n> idmatch=<identity matches> tlmH=<Step 5b matches> tlm=<total local-map matches> N=<features> state=<tracking state>
```

`adults` is non-zero only when the lifecycle sync is live; a zero count over
a long run means the tracking thread is not reporting map points to the
frontend. `mineig`/`left`/`eigdec` classify KLT failures by cause (eigenvalue
test on the forward/backward pass, left the image, and how far below the
threshold). With loop closing on, each candidate prints an `[LC]` line with the
correspondence count, best Sim(3) inliers, samples used and the gate it
failed at.

## Offline vocabulary training

```bash
Vocabulary/train_vocab --dumps <Vocabulary.dumpDir> out.txt 10 5 1 0
```

builds a DBoW2 vocabulary (`k=10`, `L=5`) from dumped keyframe descriptors;
`--images <dir>` trains from images instead. Exclude the evaluation sequence
from the training set.

## Status and known limitations

* Place recognition does not fire for the hybrid frontend on self-similar
  seabed texture: candidates pass retrieval and BoW matching and fail Sim(3)
  verification. The cause and a proposed fix (two-tier feature set) are
  discussed in the thesis.
* `Hybrid.kltMinEig` was tuned on one sequence after the main evaluation; the
  default remains OpenCV's `1.0e-4`.
* The hybrid frontend takes about twice the stock front-end time per frame,
  most of it in the dense corner response, and inserts keyframes at a higher
  rate.
* Monocular and inertial modes are untested with the hybrid frontend; it is
  wired into the stereo path only.

## Acknowledgements

Based on ORB_SLAM3 by Campos, Elvira, Rodríguez, Montiel and Tardós
(UZ-SLAMLab) and the ORB-SLAM3-STEREO-FIXED fork by zang09, which fixed the
stereo rectified-camera start-up error and raised the standard to C++14.
Licensed under GPLv3 as upstream.
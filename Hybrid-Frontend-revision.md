# Hybrid KLT–Descriptor Frontend for ORB-SLAM3 in Turbid Underwater Environments
---

## 1. Problem Statement

### 1.1 Context

The target platform is an inspection ROV operating in turbid water with low and inconsistent ambient lighting. The scenes of interest are static rigid structures; rock walls, structural dolphins, vessel hulls, usually covered in marine biofouling. Visibility ranges from a few meters down to less than one, and the imagery exhibits:

- low global contrast
- spatially inconsistent local illumination (dive lights, scattering, suspended sediment shadows)
- self-similar biological texture (encrusting growth, mussels, weed)
- occasional motion blur from current and station-keeping corrections
- short open-water transits between structures where the camera sees only featureless water and marine snow

The deliverable is a visual SLAM system providing:

1. consistent maps of inspected structures for navigation and revisit
2. relocalization after tracking loss (kidnapped-robot recovery)
3. loop closure to correct accumulated drift when revisiting a previously mapped region

### 1.2 The visual SLAM problem (notation)

Let $\mathcal{I}_k \in \mathbb{R}^{H \times W}$ denote the image captured at time $t_k$ from a calibrated camera with intrinsics $\mathbf{K}$ and distortion parameters $\boldsymbol{\kappa}$. Visual SLAM estimates a camera trajectory $\{\mathbf{T}_{wc_k}\}_{k=0}^{N} \in SE(3)$ and a set of 3D landmarks $\{\mathbf{p}_w^{(j)}\}_{j=1}^{M} \in \mathbb{R}^3$ in the world frame $w$ that jointly minimize reprojection error over the observed image sequence.

ORB-SLAM3 solves this via a three-thread architecture:

1. **Tracking** (frontend) — per-frame data association and pose estimation
2. **Local mapping** — keyframe management, map point triangulation, local bundle adjustment
3. **Loop closing** — place recognition, loop closure detection, pose graph optimization

The tracking thread is the **frontend**. The other two constitute the **backend**.

### 1.3 Hypothesis

ORB-SLAM3's standard frontend performs detection-then-descriptor-matching with ORB features. In our environment this degrades catastrophically because the BRIEF descriptor relies on local intensity comparisons that are unstable under inconsistent lighting and self-similar texture. Empirical evidence from a 3,822-frame sequence confirms this:

| Tracker | Features per frame | Epipolar inlier ratio | Motion consistency |
|---|---|---|---|
| Shi-Tomasi + KLT | 260.1 | 0.818 | 0.841 |
| ORB + descriptor matching | 38.4 | 0.674 | 0.739 |

These are means over the sequence excluding flagged open-water segments. ORB consistently produces fewer matches and lower geometric consistency. Visual inspection shows long cross-image false-match lines characteristic of descriptor aliasing on self-similar biofouling.

We retain ORB-SLAM3's backend (well-tuned and validated) and replace the frontend with a hybrid scheme:

- **Short-baseline data association** between consecutive frames: solved via pyramidal Lucas-Kanade (KLT) optical flow, which does not require descriptors.
- **Long-baseline data association** for re-identification within a pass, loop closure across passes, and relocalization after tracking loss: solved via descriptor matching with geometric verification, against a sparse keyframe-only descriptor database.

This decoupling exploits a key asymmetry: descriptors fail more on short baselines than on long baselines between *well-chosen* keyframes. On short baselines the matcher faces many visually similar candidates and cannot disambiguate the true correspondence. On long baselines between keyframes selected at meaningful viewpoint changes, the matching task is sparser and benefits from the multi-observation representative descriptor strategy (Section 5).

---

## 2. Baseline: The Standard ORB-SLAM3 Frontend

For reference. Given the previous frame $\mathcal{I}_{k-1}$ with landmark observations $\{\mathbf{u}_{k-1}^{(i)}\}$ and the current frame $\mathcal{I}_k$:

1. Extract ORB features in $\mathcal{I}_k$: a set of keypoints $\{\mathbf{u}_k^{(j)}\}$ and binary descriptors $\{\mathbf{d}_k^{(j)} \in \{0,1\}^{256}\}$ via FAST corner detection over a Gaussian pyramid with quadtree spatial distribution, followed by BRIEF descriptor computation oriented by intensity centroid.

2. For each landmark $\mathbf{p}_w^{(i)}$ tracked in $\mathcal{I}_{k-1}$, project it into $\mathcal{I}_k$ using a constant-velocity motion model:
$$\hat{\mathbf{u}}_k^{(i)} = \pi\!\left(\mathbf{K}, \mathbf{T}_{c_kw} \cdot \mathbf{p}_w^{(i)}\right)$$
where $\pi(\cdot)$ is the pinhole projection with distortion.

3. Search for the descriptor match within a window $\mathcal{W}(\hat{\mathbf{u}}_k^{(i)}, r)$ of radius $r$ around the prediction, accepting the candidate $j^*$ with minimum Hamming distance:
$$j^* = \arg\min_{j \in \mathcal{W}} \, d_H\!\left(\mathbf{d}_{k-1}^{(i)},\, \mathbf{d}_k^{(j)}\right)$$

4. Run **TrackLocalMap**: project all map points in the local covisibility graph into the current frame and match them to remaining unmatched keypoints, pulling in additional 2D-3D correspondences to tightly constrain the pose.

5. Estimate the pose $\mathbf{T}_{c_kw}$ by minimizing reprojection error over all surviving correspondences (Motion-Only BA).

**Failure mode in turbid water**: step 1 produces keypoints whose descriptors $\mathbf{d}_k^{(j)}$ are not distinctive. Hamming distances between unrelated patches on biofouling are similar to those between true correspondences, so step 3 produces high-confidence wrong matches. Steps 4 and 5 then operate on a contaminated correspondence set.

---

## 3. The Hybrid Frontend: Overview

The hybrid frontend uses three algorithms with cleanly separated responsibilities:

| Algorithm | Role | Where used |
|---|---|---|
| Shi-Tomasi corner detection | Selects pixel locations where KLT can converge reliably | New track spawning (Step 4) |
| Pyramidal Lucas-Kanade (KLT) | Tracks feature locations frame-to-frame by local optimization | Steps 1-2 |
| BRIEF descriptor | Identifies which physical landmark a feature represents | Steps 5, 5b; backend BoW, loop closure, relocalization |

A single feature is the tuple `(Shi-Tomasi corner location) + (BRIEF descriptor computed at that location) + (KLT-tracked trajectory through subsequent frames)`. One feature, three roles.

### 3.1 Why Shi-Tomasi specifically (not FAST) for new track spawning

The Lucas-Kanade update at iteration $t$ is:
$$\Delta \mathbf{v}_t = \mathbf{H}^{-1} \mathbf{b}_t, \quad \mathbf{H} = \sum_{\mathbf{x} \in \Omega} \nabla \mathcal{I}(\mathbf{x}) \nabla \mathcal{I}(\mathbf{x})^\top$$
where $\mathbf{H} \in \mathbb{R}^{2 \times 2}$ is the structure tensor over the tracking window $\Omega$. Convergence and accuracy depend on the conditioning of $\mathbf{H}$, characterized by its eigenvalues $\lambda_1 \ge \lambda_2 \ge 0$:

- $\lambda_1, \lambda_2 \approx 0$: flat region. $\mathbf{H}$ singular, no solution.
- $\lambda_1 \gg \lambda_2 \approx 0$: edge. $\mathbf{H}$ rank-deficient. Motion along the edge is unobservable (aperture problem).
- $\lambda_1 \ge \lambda_2 \gg 0$: corner. $\mathbf{H}$ well-conditioned, both motion components recoverable.

The Shi-Tomasi criterion explicitly selects points where $\lambda_2 \ge \theta_{ST}$. This is the mathematically optimal criterion for KLT trackability, derived directly from the KLT update equation.

FAST uses a contrast-based intensity comparison test (Bresenham circle) that is fast and repeatable but says nothing about $\mathbf{H}$'s conditioning. FAST fires on:
- one-dimensional intensity ridges where $\lambda_2 \approx 0$
- high-contrast blobs (marine snow particles!) with anisotropic gradient distribution
- texture boundaries that pass the contrast test locally but aren't true corners

For descriptor matching this is fine — BRIEF does not depend on $\mathbf{H}$. But for KLT tracking, FAST corners are a strict superset of Shi-Tomasi corners, and the extra points are precisely where KLT will struggle.

### 3.2 Why BRIEF on Shi-Tomasi points works

BRIEF is computed from intensity comparisons at sampling locations relative to the keypoint center. It cares about local intensity structure but does not require the underlying point to be a FAST corner. Computing BRIEF at Shi-Tomasi corners produces descriptors of identical format (256-bit binary, same sampling pattern) to those produced by stock ORB-SLAM3 — bit-compatible. The DBoW2 vocabulary, loop closure matching, and relocalization all work on these descriptors transparently. We modify the detector; the descriptor format is held fixed.
---

## 4. Per-Frame Pipeline

### 4.1 State

The frontend maintains a set of **active tracks** $\mathcal{T}_k$ at time $k$. Each track $\tau \in \mathcal{T}_k$ is a tuple:
$$\tau = \left(\, \mathrm{id}_\tau,\, \mathbf{u}_k^\tau,\, \mathbf{d}_\tau,\, \ell_\tau,\, a_\tau,\, j_\tau \,\right)$$
where:
- $\mathrm{id}_\tau \in \mathbb{N}$ is a globally unique landmark identifier
- $\mathbf{u}_k^\tau \in \mathbb{R}^2$ is the current pixel observation
- $\mathbf{d}_\tau \in \{0,1\}^{256}$ is the descriptor (initially the birth descriptor; promoted to the representative descriptor of the map point after triangulation — see Section 5)
- $\ell_\tau \in \mathbb{N}$ is the pyramid level (octave) at which the feature was originally detected
- $a_\tau \in \mathbb{N}$ is the track age (frames since birth)
- $j_\tau$ is an optional reference to a 3D map point $\mathbf{p}_w^{(j_\tau)}$ in the global map (None if not yet triangulated)

A **dormant track buffer** $\mathcal{D}_k$ holds **infant tracks only** — tracks where $j_\tau = \emptyset$ (never triangulated to a map point) that died in the last $\Delta_{\text{dormant}}$ frames, keyed by their last observed pixel and birth descriptor. This is used for short-term re-identification (Step 5).

**Adult tracks ($j_\tau \neq \emptyset$) that lose KLT tracking are discarded from the frontend entirely.** They are not added to $\mathcal{D}_k$. The rationale is twofold:

1. The corresponding map point persists in the backend's local map. As long as the landmark remains in the camera frustum, Step 5b (TrackLocalMap) will project it and re-acquire it using its representative descriptor $\bar{\mathbf{d}}^{(i)}$ — which is multi-observation and far more robust than the single-shot birth descriptor stored in the dormant buffer.

2. Allowing the same physical landmark to exist in both $\mathcal{D}_k$ (with its weak birth descriptor) and $\mathcal{L}_k$ (with its strong representative descriptor) creates a race condition between Steps 5 and 5b. A new Shi-Tomasi candidate could match the dormant entry first, inheriting only a track ID with no map point reference — corrupting the data association by creating an "orphan" track that fails to re-bind to its map point.

This makes Steps 5 and 5b operate on disjoint regions of the landmark state space: Step 5 handles infants (no map point); Step 5b handles adults (map point exists). Together they cover all reachable lost-track recoveries.

The complete track lifecycle:

| Phase | Active in $\mathcal{T}_k$? | In $\mathcal{D}_k$? | In local map $\mathcal{L}_k$? | Descriptor used for matching |
|---|---|---|---|---|
| **Infant** (born from Step 4, not yet triangulated) | Yes | No | No | Single-shot birth descriptor |
| **Infant dies** (KLT fails before triangulation) | No | Yes (up to $\Delta_{\text{dormant}}$ frames) | No | Single-shot birth descriptor (used by Step 5) |
| **Infant resurrected** (Step 5 match) | Yes (new track entry, original ID) | No (removed) | No | Single-shot birth descriptor |
| **Adult** (triangulated into map point by local mapping) | Yes | No | Yes | Multi-observation $\bar{\mathbf{d}}^{(i)}$ |
| **Adult dies** (KLT fails after triangulation) | No | **No** (discarded) | Yes | $\bar{\mathbf{d}}^{(i)}$ (used by Step 5b on next visibility) |
| **Adult re-acquired** (Step 5b match against local map) | Yes (new track entry, map point preserved) | No | Yes | $\bar{\mathbf{d}}^{(i)}$ |

Transitions out of the dormant buffer happen via Step 5 (match → active) or via expiration (frame age > $\Delta_{\text{dormant}}$ → permanently discarded).

### 4.2 Step 1: Pyramidal Lucas-Kanade tracking

Build Gaussian pyramids $\mathcal{P}_{k-1} = \{\mathcal{I}_{k-1}^{(0)}, \ldots, \mathcal{I}_{k-1}^{(L)}\}$ and $\mathcal{P}_k$ similarly, where $\mathcal{I}^{(l+1)}$ is $\mathcal{I}^{(l)}$ downsampled by factor 2 after Gaussian smoothing.

For each track $\tau \in \mathcal{T}_{k-1}$, find the displacement $\mathbf{v}_\tau \in \mathbb{R}^2$ that minimizes the SSD over a window $\Omega$ centered on $\mathbf{u}_{k-1}^\tau$:
$$\mathbf{v}_\tau^* = \arg\min_{\mathbf{v}} \sum_{\mathbf{x} \in \Omega} \Big[\, \mathcal{I}_{k-1}(\mathbf{u}_{k-1}^\tau + \mathbf{x}) - \mathcal{I}_k(\mathbf{u}_{k-1}^\tau + \mathbf{x} + \mathbf{v}) \,\Big]^2$$

Solved iteratively from coarsest to finest pyramid level, with the coarse displacement initializing the fine search. The Lucas-Kanade update at each iteration is:
$$\Delta \mathbf{v} = \mathbf{H}^{-1} \mathbf{b}, \qquad \mathbf{H} = \sum_\Omega \nabla \mathcal{I}_k \nabla \mathcal{I}_k^\top, \qquad \mathbf{b} = \sum_\Omega \nabla \mathcal{I}_k \cdot \Delta \mathcal{I}$$

The new observation is $\mathbf{u}_k^\tau = \mathbf{u}_{k-1}^\tau + \mathbf{v}_\tau^*$.

We warp at translation-only resolution within each pyramid level. Affine warping (as in SVO) would improve robustness to viewpoint changes but is significantly more expensive. For our environment the dominant nuisance is appearance change rather than geometric warping, so translational KLT is expected to enough. This is worth revisiting if results don't match.

### 4.3 Step 2: Forward-backward consistency filter

Re-track each $\mathbf{u}_k^\tau$ backward from $\mathcal{I}_k$ to $\mathcal{I}_{k-1}$, obtaining $\tilde{\mathbf{u}}_{k-1}^\tau$. Compute the FB error:
$$\epsilon_\tau^{FB} = \left\| \mathbf{u}_{k-1}^\tau - \tilde{\mathbf{u}}_{k-1}^\tau \right\|_2$$
and reject tracks where $\epsilon_\tau^{FB} > \theta_{FB}$ (typically $\theta_{FB} = 1$ px). This catches the case where KLT converges to a different nearby corner — the round trip would not return to the origin.

### 4.4 Step 3: Geometric outlier rejection

Estimate the fundamental matrix $\mathbf{F}_{k-1,k}$ via RANSAC over the surviving tracks, using:
$$\mathbf{u}_k^\tau{}^\top \, \mathbf{F}_{k-1,k} \, \mathbf{u}_{k-1}^\tau = 0$$
for true correspondences (homogeneous coordinates). Reject tracks whose Sampson distance to the recovered epipolar geometry exceeds $\theta_{RANSAC}$.

For stereo configurations we additionally enforce the stereo epipolar constraint between left and right images via precomputed extrinsics. This is cheap and removes mismatched stereo pairs.

The set of tracks surviving Steps 1–3 is denoted $\mathcal{T}_k^{\text{surv}}$.

### 4.5 Step 4: Track top-up via Shi-Tomasi + BRIEF

Let $N_{\text{target}}$ be the desired active track count (typically 500–1000, matching ORB-SLAM3 default scale). If $|\mathcal{T}_k^{\text{surv}}| < N_{\text{target}}$, spawn new tracks.

**4.5.1 Build occupancy mask.** To avoid clumping new detections near existing tracks, construct:
$$\mathcal{M}_k(\mathbf{x}) = \begin{cases} 0 & \text{if } \exists\, \tau \in \mathcal{T}_k^{\text{surv}} : \|\mathbf{x} - \mathbf{u}_k^\tau\|_\infty < r_{\text{mask}} \\ 1 & \text{otherwise} \end{cases}$$

**4.5.2 Shi-Tomasi detection.** Detect corners in $\mathcal{I}_k$ restricted to $\mathcal{M}_k$ by computing $\lambda_2(\mathbf{H}(\mathbf{x}))$ at every candidate pixel and accepting corners with $\lambda_2 \ge \theta_{ST}$, subject to non-maximum suppression at scale $r_{\text{nms}}$. This is the standard `goodFeaturesToTrack`-style detection.

**4.5.3 Quadtree spatial distribution.** Apply ORB-SLAM3's quadtree distribution on the accepted Shi-Tomasi corners to enforce uniform spatial coverage. The quadtree recursively subdivides the image region until each leaf contains at most one keypoint or the target keypoint count is reached, retaining the highest-response corner per leaf. This is the spatial-spread benefit of ORB-SLAM3's design, decoupled from FAST.

**4.5.4 Orientation.** For each retained corner $\mathbf{u}$, compute orientation via intensity centroid:
$$m_{pq} = \sum_{x,y} x^p y^q \mathcal{I}(x + u_x,\, y + u_y), \qquad \theta = \arctan_2(m_{01},\, m_{10})$$
with sums over a circular patch of radius 15 px (ORB default).

**4.5.5 BRIEF descriptor.** Compute the steered BRIEF descriptor:
$$d_i = \begin{cases} 1 & \text{if } \mathcal{I}(\mathbf{R}_\theta \mathbf{a}_i + \mathbf{u}) < \mathcal{I}(\mathbf{R}_\theta \mathbf{b}_i + \mathbf{u}) \\ 0 & \text{otherwise} \end{cases}, \quad i = 1 \ldots 256$$
where $(\mathbf{a}_i, \mathbf{b}_i)$ are the ORB-SLAM3 BRIEF sampling pattern and $\mathbf{R}_\theta$ is the 2D rotation by $\theta$ for rotation invariance.

**4.5.6 Stereo: per-frame depth assignment for new features (stereo / RGB-D only).** In stereo configurations, ORB-SLAM3's `Frame` constructor expects every keypoint to have an assigned depth value (or "no depth" sentinel) immediately upon extraction. Local mapping uses this for fast triangulation of new map points without needing temporal parallax. Our hybrid pipeline must preserve this contract:

1. Run Steps 4.5.1–4.5.5 on **both** the left and right rectified images, producing two candidate sets $\mathcal{S}_k^L$ and $\mathcal{S}_k^R$. The dormant buffer, KLT tracking, and active-track set $\mathcal{T}_k$ are maintained over the left image only; the right image is used solely for depth computation at extraction time.

2. For each candidate $(\mathbf{u}_k^{(j),L}, \mathbf{d}_k^{(j),L}) \in \mathcal{S}_k^L$, search along its rectified epipolar line in $\mathcal{S}_k^R$ for a match satisfying:
   - Same pyramid level (within ±1 octave)
   - $|v^L - v^R| < \theta_{\text{stereo-row}}$ (rectified row alignment, typically 1 px)
   - $u^L > u^R$ (positive disparity)
   - $d_H(\mathbf{d}_k^{(j),L}, \mathbf{d}_k^{(j'),R}) < \theta_{\text{stereo-desc}}$
   - SAD block match around the predicted disparity to obtain sub-pixel disparity

3. Compute depth from disparity using the stereo baseline $b$ and focal length $f$:
$$z^{(j)} = \frac{f \cdot b}{u^{L,(j)} - u^{R,(j)}}$$
and attach it to the left-image candidate. Candidates with no valid right-image match are kept as monocular features (no depth), exactly as stock ORB-SLAM3 does for points outside the stereo overlap region.

4. KLT tracking (Steps 1–3) operates on left-image positions only. After Step 3, **recompute disparity for each surviving track each frame** via SAD block matching on a small search range around the previous frame's disparity. This is the cheapest way to maintain valid stereo depths through KLT-tracked frames; the alternative of independently KLT-tracking both left and right images is more expensive and offers no accuracy benefit because the rectified epipolar constraint already pins the right position to a 1D search.

The resulting set of new candidate features $\mathcal{S}_k = \{(\mathbf{u}_k^{(j)},\, \ell_k^{(j)},\, \mathbf{d}_k^{(j)},\, z_k^{(j)})\}$ is bit-compatible with stock ORB-SLAM3 stereo features (including the convention that $z = \emptyset$ marks monocular-only features). For monocular configurations, ignore 4.5.6 entirely; the set $\mathcal{S}_k$ has no depth field and triangulation proceeds normally via temporal parallax in local mapping.

### 4.6 Step 5: Re-identification of dormant tracks

For each new candidate $(\mathbf{u}_k^{(j)},\, \mathbf{d}_k^{(j)}) \in \mathcal{S}_k$, search the dormant buffer $\mathcal{D}_k$ for matches:
$$\tau^* = \arg\min_{\tau \in \mathcal{D}_k \,\cap\, \mathcal{W}(\mathbf{u}_k^{(j)},\, r_{\text{reid}})} \, d_H\!\left(\mathbf{d}_\tau, \mathbf{d}_k^{(j)}\right)$$
where $\mathcal{W}(\cdot, r_{\text{reid}})$ is the spatial window around the candidate (optionally propagated by the current motion model from the dormant track's last position).

Accept the re-identification if $d_H(\mathbf{d}_{\tau^*},\, \mathbf{d}_k^{(j)}) < \theta_{\text{reid}}$. If accepted:
- The new feature inherits $\mathrm{id}_{\tau^*}$ (preserving track identity for accumulating parallax toward eventual triangulation)
- $\tau^*$ is removed from $\mathcal{D}_k$
- The re-identified track remains an **infant** ($j_\tau = \emptyset$); it does not inherit any map point reference because by construction (§4.1) dormant tracks have $j_\tau = \emptyset$

Otherwise the candidate is assigned a fresh $\mathrm{id}$ and added to $\mathcal{T}_k$ as a new track.

This step uses descriptors only for **local re-identification of dropped tracks**, never for primary frame-to-frame matching. The candidate set is small (typically O(10–100) dormant tracks within $r_{\text{reid}}$), the spatial prior is strong, and we only need to confirm "same landmark" rather than pick one from thousands. This is a much easier descriptor task than vanilla ORB matching.

**Important asymmetry**: in Step 5, both sides of the match are single-shot descriptors — the dormant track has only its birth descriptor, since it died before being promoted to a map point with multiple observations. Step 5 is therefore the inherently weakest descriptor step in the pipeline, and the system must be designed to tolerate Step 5 failures gracefully. A missed re-ID is not catastrophic: the landmark gets a new ID and is retriangulated, which is wasteful but recoverable.

### 4.7 Step 5b: TrackLocalMap — descriptor matching against the local map

ORB-SLAM3's tracking thread performs two phases of data association per frame: frame-to-frame (Steps 1–3) and TrackLocalMap. Skipping the second phase systematically under-constrains pose estimation, especially on long-tracked surfaces where many landmarks are already in the local map but new ones enter the field of view continuously.

After Step 5, compute a provisional pose $\hat{\mathbf{T}}_{c_kw}$ from the current correspondence set. Then query the local map for the set $\mathcal{L}_k$ of map points satisfying:

1. In the local covisibility graph of recent keyframes
2. Project inside the image bounds under $\hat{\mathbf{T}}_{c_kw}$
3. Within the per-landmark viewing-angle bounds (ORB-SLAM3 maintains the mean observation direction per map point)
4. **Not already being tracked** by any active KLT track in $\mathcal{T}_k$

For each $\mathbf{p}_w^{(i)} \in \mathcal{L}_k$, compute:
- Predicted pixel: $\hat{\mathbf{u}}_k^{(i)} = \pi(\mathbf{K},\, \hat{\mathbf{T}}_{c_kw} \cdot \mathbf{p}_w^{(i)})$
- Predicted pyramid level $\hat{\ell}_k^{(i)}$, estimated from the distance ratio to the map point's reference scale

Then search for a match among the unmatched candidates in $\mathcal{S}_k$:
$$j^* = \arg\min_{j \,\in\, \mathcal{S}_k^{\text{unmatched}} \,\cap\, \mathcal{W}(\hat{\mathbf{u}}_k^{(i)},\, r_{\text{TLM}}(\hat{\ell}_k^{(i)}))} \, d_H\!\left(\bar{\mathbf{d}}^{(i)},\, \mathbf{d}_k^{(j)}\right)$$
where $\bar{\mathbf{d}}^{(i)}$ is the **representative descriptor** of map point $i$ (Section 5) and $r_{\text{TLM}}$ scales with the predicted pyramid level (typically 2–4 px at level 0, growing geometrically).

Accept if $d_H(\bar{\mathbf{d}}^{(i)},\, \mathbf{d}_k^{(j^*)}) < \theta_{\text{TLM}}$. The new correspondence $(\mathbf{u}_k^{(j^*)},\, \mathbf{p}_w^{(i)})$ is added to $\mathcal{T}_k$ with $\mathrm{id}_\tau$ inherited from the map point and a **new KLT track is initialized** at that corner so subsequent frames track it via Step 1.

### 4.8 Steps 5 and 5b share a primitive

Both steps are spatially-gated descriptor matches against a small candidate set:

| Step | Predict location from | Match against | Result if accepted |
|---|---|---|---|
| 5 (Re-ID) | Dormant track's last pixel + motion prior | Recently-died track descriptors | Resurrect track ID; re-link to existing map point if it had one |
| 5b (TrackLocalMap) | Map point reprojection under provisional pose | Local map point representative descriptors | Initialize new KLT track tied to existing map point |

The shared primitive is `SpatialDescriptorMatcher`: given a list of `(prediction_pixel, query_descriptor)` queries and a list of `(candidate_pixel, candidate_descriptor)` candidates, return the best match per query under spatial and Hamming-distance gates. This is implemented as a standalone class, unit-testable on synthetic inputs.

### 4.9 Step 6: Motion-only bundle adjustment

The correspondence set entering pose estimation is the union of:
- KLT-tracked correspondences from Steps 1–3
- Re-identified correspondences from Step 5
- TrackLocalMap correspondences from Step 5b

Estimate $\mathbf{T}_{c_kw}$ by:
$$\mathbf{T}_{c_kw}^* = \arg\min_{\mathbf{T}} \sum_{\tau \,:\, j_\tau \ne \emptyset} \rho\!\left( \left\| \mathbf{u}_k^\tau - \pi(\mathbf{K},\, \mathbf{T} \cdot \mathbf{p}_w^{(j_\tau)}) \right\|_{\Sigma_\tau}^2 \right)$$
where $\rho(\cdot)$ is a Huber robust kernel and $\Sigma_\tau$ is the observation covariance scaled by pyramid level (higher octave → higher uncertainty), as in ORB-SLAM3.

The pose is then passed to local mapping, which performs local BA, triangulates new map points from tracks with sufficient parallax, and decides keyframe insertion. None of this changes from stock ORB-SLAM3.

### 4.10 Step 7: Keyframe insertion and descriptor maintenance

If the current frame is selected as a keyframe (ORB-SLAM3's existing criteria, untouched), then for every track $\tau$ observed in this keyframe that is bound to a map point $\mathbf{p}_w^{(j_\tau)}$:

1. Append the current observation's descriptor (computed at the current Shi-Tomasi corner location, or interpolated from the KLT-tracked location at the appropriate pyramid level — see Section 5.2 for the choice) to the observation set $\mathcal{O}_{j_\tau}$.
2. Recompute $\bar{\mathbf{d}}^{(j_\tau)}$ using the median-Hamming criterion (Section 5.1).

This maintains the representative descriptor in line with the current appearance of the landmark across viewing conditions.

---

## 5. Representative Descriptor Strategy

### 5.1 Multi-observation descriptor set per map point

For each map point $\mathbf{p}_w^{(i)}$, maintain a set of observation descriptors:
$$\mathcal{O}_i = \left\{ \mathbf{d}_{k_1}^{(i)},\, \mathbf{d}_{k_2}^{(i)},\, \ldots,\, \mathbf{d}_{k_n}^{(i)} \right\}$$
where each entry is the descriptor computed at keyframe $k_j$'s observation of landmark $i$. Descriptors are added only at keyframes, not every frame, both for storage efficiency and because non-keyframe descriptors add noise without information (consecutive frames yield highly correlated descriptors).

The **representative descriptor** used for matching (in Steps 5b, loop closure, relocalization) is the observation with minimum median Hamming distance to all other observations:
$$\bar{\mathbf{d}}^{(i)} = \arg\min_{\mathbf{d} \in \mathcal{O}_i} \; \mathrm{median}_{\mathbf{d}' \in \mathcal{O}_i \setminus \{\mathbf{d}\}} \, d_H(\mathbf{d},\, \mathbf{d}')$$

This is exactly ORB-SLAM3's `MapPoint::ComputeDistinctiveDescriptors`. The median-of-pairwise-distances criterion picks the "most central" observation in the cluster, which is robust to outliers from single bad-quality observations (motion blur, transient occlusion, brief turbidity spike).

### 5.2 Where the descriptor at a keyframe observation comes from

Two cases:

**Case A**: at this keyframe, the track was matched in Step 5b (TrackLocalMap), so a fresh Shi-Tomasi corner was detected with an explicit BRIEF descriptor. Use that descriptor directly.

**Case B**: at this keyframe, the track was carried by KLT from a previous frame. KLT only updates pixel position; it does not produce a descriptor. We have two options:

1. **Compute BRIEF on the fly at the current KLT pixel location.** Requires running the orientation step and BRIEF computation, but the corner location is already known. Cost: ~1 ms per track in C++.

2. **Reuse the descriptor from the most recent keyframe where the track was matched in Step 5b.** Cheaper but staler.

We choose option 1. The compute cost at keyframe rate (≪ frame rate) is negligible, and it ensures the keyframe's descriptor reflects the keyframe's actual appearance rather than a prior keyframe's. This is the same choice ORB-SLAM3 implicitly makes (its descriptors are always computed at the current keyframe).

### 5.3 Update schedule

- $\mathcal{O}_i$ grows by one entry each time the corresponding track is observed in a new keyframe.
- $\bar{\mathbf{d}}^{(i)}$ is recomputed whenever $\mathcal{O}_i$ changes.
- For long-lived map points, $|\mathcal{O}_i|$ may be bounded (ORB-SLAM3 does not explicitly bound it; the local map size limits the working set anyway). In our environment, where individual surfaces may be tracked across hundreds of keyframes, an explicit cap (e.g., 50) may be warranted for memory reasons.

### 5.4 The asymmetry between active tracks, dormant tracks, and map points

| State | Descriptor available | Used in |
|---|---|---|
| Active track, not yet a map point | Birth descriptor (single shot) | Carried forward through KLT |
| Dormant track | Birth descriptor (single shot) | Step 5 re-ID |
| Map point with observations | Representative $\bar{\mathbf{d}}^{(i)}$ over multi-observation set | Step 5b TLM, loop closure, relocalization |

Step 5 is the weakest because both sides are single-shot. Step 5b is much stronger because the local map side uses $\bar{\mathbf{d}}^{(i)}$.

---

## 6. Backend Integration (Unchanged from Stock ORB-SLAM3)

### 6.1 What the backend expects

The backend operates on keyframes containing `(KeyPoint, descriptor, octave)` tuples. As long as our frontend produces these in the standard format — which it does, since BRIEF computed at Shi-Tomasi corners is bit-identical to BRIEF computed at FAST corners — the backend is agnostic to the data association method used upstream.

### 6.2 Bag-of-Words for place recognition

DBoW2 converts each keyframe's descriptor set into a compact BoW vector $\mathbf{v}_{\text{BoW}} \in \mathbb{R}^V$ over a pre-trained vocabulary of $V$ visual words (a hierarchical k-means tree built offline). Place recognition queries are nearest-neighbor lookups in BoW space, scored by:
$$s(\mathbf{v}_1,\, \mathbf{v}_2) = 1 - \frac{1}{2} \left\| \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|_1} - \frac{\mathbf{v}_2}{\|\mathbf{v}_2\|_1} \right\|_1$$

Candidate matches above a similarity threshold proceed to geometric verification.

**Vocabulary mismatch is a known risk** for our environment (see Section 7.1). The standard DBoW2 vocabulary is trained on terrestrial imagery; underwater biofouling produces a different distribution of BRIEF responses. **Retraining the vocabulary on a held-out subset of our own footage is mandatory**, not optional. Details in Section 7.1.

### 6.3 Loop closure

When a place recognition candidate $\mathcal{K}_{\text{cand}}$ is identified for the current keyframe $\mathcal{K}_{\text{curr}}$:

1. Descriptor-based 2D-2D matching between $\mathcal{K}_{\text{curr}}$ and $\mathcal{K}_{\text{cand}}$ (using $\bar{\mathbf{d}}$ for both sides).
2. Sim(3) estimation via RANSAC over matched 3D points (reduces to SE(3) for stereo where scale is observable).
3. Acceptance if inlier count exceeds threshold and the optimized transform is geometrically reasonable.
4. Pose graph optimization over the essential graph, distributing the loop closure correction across all intervening keyframes.

### 6.4 Relocalization

If tracking is lost ($|\mathcal{T}_k^{\text{surv}}| < N_{\text{min}}$ for several consecutive frames, or pose estimation diverges), the relocalization routine queries the BoW index over all past keyframes and attempts PnP against each candidate's 3D map points, accepting the first that produces a geometrically consistent pose with sufficient inliers.

The query uses standard ORB extraction (FAST + BRIEF) on the relocalization frame, because the BoW index is built over standard ORB descriptors. This is run independently of the hybrid frontend's normal Step 4 detection — it is a one-off recovery operation, not part of the per-frame pipeline.

**Handoff to the hybrid frontend after successful relocalization.** When PnP accepts, ORB-SLAM3 produces a set of 2D-3D correspondences: rescued map points $\{\mathbf{p}_w^{(i_n)}\}$ matched to keypoints $\{\mathbf{u}_k^{(n)}\}$ in the relocalization frame. The hybrid frontend's contract requires that Step 1 on the *next* frame has a populated $\mathcal{T}_{k-1}$ to KLT-track forward. The handoff is therefore explicit:

1. **Initialize $\mathcal{T}_k$ directly from the rescued correspondences.** For each accepted $(\mathbf{u}_k^{(n)}, \mathbf{p}_w^{(i_n)})$ pair, create a new active track with:
   - Fresh $\mathrm{id}_\tau$
   - Position $\mathbf{u}_k^\tau = \mathbf{u}_k^{(n)}$
   - Map point reference $j_\tau = i_n$ (these are adult tracks from birth — their map point already exists)
   - Descriptor $\mathbf{d}_\tau = \bar{\mathbf{d}}^{(i_n)}$ (inherited from the map point's representative descriptor)
   - Pyramid level $\ell_\tau$ inherited from the matched keypoint
   - Age $a_\tau = 0$

2. **Bypass Steps 1–3 on the relocalization frame** (there is no $\mathcal{T}_{k-1}$ to KLT-track from). Proceed directly to Step 4 to top up with Shi-Tomasi corners, then Steps 5b and 6 as usual. Step 5 is skipped on this frame ($\mathcal{D}_k$ is stale; we purge it on relocalization).

3. **Stereo configurations**: compute disparity / depth for the rescued correspondences using the standard SAD search (§4.5.6) before passing them into Step 6's BA.

4. **On frame $k+1$ onward**: the pipeline runs normally. Step 1 KLT-tracks the rescued features forward.

**Trackability caveat.** The rescued correspondences come from FAST corners (BoW requires standard ORB extraction), which are not guaranteed to be optimal KLT-trackable points per §3.1. Some fraction will be edge-like ($\lambda_2 \approx 0$) and KLT will lose them within a few frames. This is acceptable: those tracks are adult ($j_\tau \neq \emptyset$) so on death they are simply discarded from the frontend, and Step 5b will re-acquire the same map points as Shi-Tomasi corners are spawned nearby on subsequent frames. Recovery is gradual but architecturally clean. If empirical performance proves too slow, a one-time post-relocalization pass to replace FAST-derived seeds with nearby Shi-Tomasi corners (matching by descriptor) is a possible optimization, but we defer it pending evidence that it's needed.

5. **Purge the dormant buffer on relocalization.** Tracks that died before the tracking loss are stale (the dormant buffer's spatial predictions assume continuity of motion, which is broken by the loss). Set $\mathcal{D}_k = \emptyset$ on the relocalization frame. The buffer rebuilds organically from subsequent frame-to-frame KLT failures.

---

## 7. Risks and Mitigations

### 7.1 DBoW2 vocabulary mismatch (mandatory mitigation)

**Risk**: The default ORB-SLAM3 DBoW2 vocabulary is trained on terrestrial imagery (urban, KITTI-style). On underwater biofouling, BRIEF descriptors have a different statistical distribution. Naïve use of the default vocabulary causes **perceptual aliasing**: many features hash to the same few visual words, BoW similarity scores between unrelated underwater frames become uniformly high, and the loop closer fires on false positives. False loop closures inject incorrect constraints into the pose graph, corrupting the entire map.

**Mitigation**: retrain the vocabulary on our own data.

1. Collect ~10,000 frames spanning the diversity of operating conditions (turbid/clear, various surfaces, various lighting).
2. Run the hybrid frontend (or stock ORB-SLAM3's ORB extractor) on these frames and accumulate all extracted BRIEF descriptors.
3. Build the hierarchical k-means tree with DBoW2's `create_voc_step` utility. Standard parameters: branching factor 10, depth 6 → $10^6$ visual words.
4. Save the resulting vocabulary file and point ORB-SLAM3 at it via the config.

This is offline, one-time, roughly 30 minutes of compute. **Skip this step and the system will produce confidently wrong loop closures.**

### 7.2 Descriptor staleness across long viewpoint changes

**Risk**: BRIEF is rotation-invariant via the steered sampling pattern, but not scale-invariant beyond the pyramid octaves and not robust to large illumination changes. When a structure is revisited from a substantially different viewpoint (different range, different lighting, different angle), even the representative descriptor may fail to match.

**Mitigation hierarchy**:

1. The keyframe-update strategy (Section 5) mitigates this for within-pass traversal by bounding the maximum viewpoint difference between adjacent observations.
2. The median-of-medians representative descriptor (rather than freshest or birth) picks the most "central" appearance, which is more likely to match a future observation than an outlier-biased choice.
3. For cross-pass loop closure with very different viewpoints (e.g., revisiting a rock from the opposite side), BRIEF will fail regardless of update strategy. If this proves common in operation, swap BRIEF for a learned descriptor with true scale/illumination invariance (SuperPoint, ALIKE, DISK). This is roughly an order of magnitude more implementation work (GPU inference, descriptor dimensionality changes break DBoW2 compatibility, mandatory vocabulary retraining) and should only be considered if BRIEF-with-retrained-vocabulary genuinely fails.

### 7.3 Open-water sections

**Risk**: when the ROV transits between structures, the camera sees only marine snow on a featureless background. KLT will track suspended-particle motion (which is independent of camera motion), producing tracks that look superficially valid but encode current drift rather than camera ego-motion. ORB will produce few matches because there are no real corners.

**Mitigation**: detect and flag open-water frames using the existing failure criteria (low track count, low motion consistency, low epipolar inlier ratio). During flagged segments:

1. Suppress map updates (no new triangulations, no keyframe insertion).
2. Maintain pose by dead reckoning (constant velocity, or IMU integration if available).
3. Attempt relocalization continuously when track count begins to recover, expecting to re-enter the map upon reaching the next structure.

This is not a SLAM problem so much as a behavior-mode-switching problem, and should be implemented as a separate state in the tracking thread.

### 7.4 Implementation surface and silent map corruption

**Risk**: track ID propagation bugs (the same ID assigned to two different physical features, or a `MapPoint*` dangling after deletion) silently corrupt the map. The symptoms appear hours of footage later and are difficult to diagnose.

**Mitigations**:

**7.4.1 Invariants enforceable by assertion.** All of these are constant-time checks; cost is negligible relative to KLT and BA. Use `assert()` (with `NDEBUG` undefined in debug builds) liberally:

- No two active tracks in the same frame share `id_τ`.
- No active track holds a `MapPoint*` for which `MapPoint::isBad()` is true.
- No `MapPoint` has two observations attributed to the same `(KeyFrame*, KeyPointIndex)` pair (ORB-SLAM3 asserts this internally; we must respect it on insertion).
- The dormant buffer never contains a track whose ID is also in the active set.
- Re-identified tracks correctly inherit the `MapPoint*` from the dormant track, not a freshly allocated map point.

**7.4.2 Modularization.** The two new components are pure data structures with no ORB-SLAM3 dependencies:

- `DormantTrackBuffer`: ring buffer of `(id, last_pixel, descriptor, frame_died, MapPoint*)` tuples. Public interface: `add(track, frame)`, `purge_older_than(frame)`, `query_within(pixel, radius) → vector<entry>`, `remove(id)`. Built and unit-tested standalone.
- `SpatialDescriptorMatcher`: takes lists of queries and candidates, returns best match per query under spatial + Hamming gates. Stateless. Built and unit-tested standalone.

This separates algorithmic correctness (do the modules work in isolation on synthetic inputs?) from integration correctness (does modified `Tracking.cc` call them correctly?).

**7.4.3 Map-corruption canaries.** Add lightweight runtime checks that detect known corruption patterns:

- Sudden jumps in map point count (Δ > some threshold per frame): likely indicates a bad keyframe insertion.
- Average reprojection error rising: BA is failing, likely due to bad correspondences.
- Loop closure inlier ratio < threshold but acceptance still occurring: vocabulary or geometric verification misconfigured.

Log these continuously; if any tripped, dump the relevant frames + correspondences to disk for post-mortem.

### 7.5 KLT track persistence under appearance change

**Risk**: KLT is patch-tracking, not identity-tracking. When patch appearance changes substantially (viewpoint, lighting, partial occlusion), KLT's gradient descent terminates, the track dies, and a new track is spawned in the same region with a new ID. The Step 5 re-identification mechanism is designed to resurrect these tracks, but if Step 5 fails (descriptors too noisy on biofouling) we accumulate "ghost" landmarks — the same physical feature tracked under multiple IDs in the map.

**Mitigation**: this risk is real and inherent to the architecture. The local mapping thread's existing map point fusion logic (which merges map points that project to within a threshold of each other across multiple keyframes) provides a secondary cleanup mechanism. Ghost landmarks should eventually be detected and merged at the next BA pass that covers both observations.

If this becomes a significant problem in practice, a more aggressive map point fusion criterion in local mapping (smaller threshold, more frequent fusion attempts) is a tunable mitigation.

---

## 8. Parameters

Starting values; all are subject to empirical tuning on real data.

| Symbol | Meaning | Start | Notes |
|---|---|---|---|
| $L$ | KLT pyramid levels | 3 | Increase for large frame-to-frame motion |
| $\|\Omega\|$ | KLT window size | 21×21 px | Larger helps low-texture |
| $\theta_{FB}$ | FB error threshold | 1.0 px | Tighter is stricter |
| $\theta_{RANSAC}$ | Sampson distance threshold | 3.0 px | Calibration-dependent |
| $N_{\text{target}}$ | Target active track count | 1000 | Matches ORB-SLAM3 |
| $N_{\text{min}}$ | Track count below which tracking is "lost" | 50 | Triggers relocalization |
| $r_{\text{mask}}$ | Mask radius for new detections | 10 px | Anti-clumping |
| $\theta_{ST}$ | Shi-Tomasi $\lambda_2$ threshold | quality 0.01 | Low for low-contrast |
| $\Delta_{\text{dormant}}$ | Dormant buffer depth | 30 frames | Longer = more re-ID chances |
| $r_{\text{reid}}$ | Re-ID spatial search radius | 20 px | Scales with motion |
| $\theta_{\text{reid}}$ | Re-ID Hamming threshold | 50/256 | Tighter than $\theta_{\text{TLM}}$ |
| $r_{\text{TLM}}(\ell)$ | TLM search radius at octave $\ell$ | $2 \cdot 1.2^\ell$ px | Geometric scaling |
| $\theta_{\text{TLM}}$ | TLM Hamming threshold | 50/256 | Standard ORB-SLAM3 value |
| $\theta_{\text{stereo-row}}$ | Stereo row alignment tolerance (rectified) | 1 px | Tighter for well-calibrated cameras |
| $\theta_{\text{stereo-desc}}$ | Stereo descriptor Hamming threshold | 50/256 | Same as TLM |
| Vocab branching factor | DBoW2 | 10 | Standard |
| Vocab depth | DBoW2 | 6 | Standard, $10^6$ words |

---

## 9. Implementation Plan

### 9.1 ORB-SLAM3 fork structure

We fork `ORB_SLAM3` and modify the following files:

| File | Modification |
|---|---|
| `include/Frame.h`, `src/Frame.cc` | Add tracking state fields (track IDs per keypoint); leave keypoint/descriptor representation unchanged |
| `include/ORBextractor.h`, `src/ORBextractor.cc` | Replace FAST detection with Shi-Tomasi inside the existing pyramid + quadtree structure. Keep BRIEF computation unchanged |
| `include/Tracking.h`, `src/Tracking.cc` | Replace `TrackWithMotionModel` / `TrackReferenceKeyFrame` with the hybrid pipeline (Steps 1–5). Existing `TrackLocalMap` modified to Step 5b semantics (operate on $\mathcal{S}_k^{\text{unmatched}}$). `Relocalization` unchanged |
| `include/MapPoint.h`, `src/MapPoint.cc` | No changes — `ComputeDistinctiveDescriptors` already does what we need |
| (new) `include/DormantTrackBuffer.h`, `src/DormantTrackBuffer.cc` | New module |
| (new) `include/SpatialDescriptorMatcher.h`, `src/SpatialDescriptorMatcher.cc` | New module |
| `include/LoopClosing.h`, `src/LoopClosing.cc` | No changes |
| `include/LocalMapping.h`, `src/LocalMapping.cc` | No changes |
| `Vocabulary/*` | Retrain (offline, see §7.1) |

Estimate: ~500–1000 lines of net new C++ code, plus ~500 lines of modified existing code. Several weeks of focused work.

### 9.2 Suggested order of work

1. **Standalone modules first** (no ORB-SLAM3 dependency). `DormantTrackBuffer`, `SpatialDescriptorMatcher`. Full unit tests on synthetic inputs. Should compile and pass tests outside the ORB-SLAM3 source tree.

2. **Python proof-of-concept of Steps 1–5** on the existing 3,822-frame sequence. Builds on the existing `compare_trackers.py` infrastructure. Validates the matching thresholds and re-ID success rate before committing to C++. Adds a track-ID system to KLT and measures the re-ID precision/recall on synthetic dropouts (force KLT failures and see whether Step 5 resurrects the right track).

3. **Modify `ORBextractor`** to detect Shi-Tomasi instead of FAST. Run stock ORB-SLAM3 with this change on standard datasets (KITTI, TUM). Sanity check: performance should be roughly equivalent — Shi-Tomasi is not dramatically different from FAST on terrestrial imagery, so this swap alone shouldn't break anything. If KITTI/TUM results regress significantly, we have a bug.

4. **Retrain vocabulary** on collected footage. Run loop closure regression tests against held-out underwater sequences.

5. **Integrate Steps 1–5 into `Tracking.cc`**. Maintain a feature flag (`USE_HYBRID_FRONTEND`) to allow A/B comparison with stock ORB-SLAM3.

6. **Integrate Step 5b** (modify TrackLocalMap to operate on $\mathcal{S}_k^{\text{unmatched}}$).

7. **End-to-end testing** on full sequences. Compare ATE/RPE against stock ORB-SLAM3 on our own data and on KITTI as a non-regression check.

### 9.3 Tests

Beyond unit tests on the new modules, end-to-end tests should include:

- **Non-regression**: hybrid frontend on KITTI / TUM-RGBD sequences should produce ATE within 10% of stock ORB-SLAM3.
- **Improvement metric**: hybrid frontend on our underwater sequences should produce successful tracking through segments where stock ORB-SLAM3 loses tracking, quantified by tracking success rate (fraction of frames with valid pose) and ATE on segments with ground truth (if available; otherwise inspection).
- **Loop closure regression**: synthetic loop closure sequences (revisit known structures) should produce loop closure events with stable inlier ratios. False loop closures on unrelated segments should be zero or near-zero.

---

## 10. Open Validation Questions

These cannot be settled from theory and need to be answered with data before committing fully:

1. **Does descriptor matching work at keyframe scale on our imagery?** We have evidence it fails frame-to-frame, but we have not directly tested keyframe-to-keyframe matching (separated by 10–30 frames). If this also fails, the loop closure mechanism will not work regardless of vocabulary retraining, and we need to revisit Section 7.2's escalation to learned descriptors. **Concrete test**: from our existing sequence, sample frame pairs separated by 10, 20, 30 frames. Run ORB matching with geometric verification. Plot inlier ratio vs separation. If it stays high (> 0.5) at 30 frames, we're in good shape. If it collapses by 20, we have a problem.

2. **What is the empirical re-identification rate of Step 5 on our data?** Forced-failure test: in the Python proof-of-concept, deliberately kill 10% of KLT tracks each frame and measure what fraction Step 5 correctly resurrects (i.e., assigns the original ID) versus incorrectly (assigns a different ID or fails to resurrect at all). Target: > 80% correct resurrection, < 1% incorrect resurrection.

3. **How frequent are open-water sections in practice?** Determines whether mode-switching is a marginal concern or a central design constraint. Inspect existing footage and report fraction of frames flagged as open-water under the current criteria.

---

## 11. Summary Table of the Pipeline

| Step | Operation | Algorithm | Inputs | Outputs |
|---|---|---|---|---|
| 1 | KLT-track active features | Pyramidal LK | $\mathcal{T}_{k-1}$, $\mathcal{I}_{k-1}$, $\mathcal{I}_k$ | Provisional new positions |
| 2 | FB consistency filter | KLT round-trip | Provisional positions | $\mathcal{T}_k^{\text{surv}}$ |
| 3 | Geometric outlier rejection | 8-pt F + RANSAC | $\mathcal{T}_k^{\text{surv}}$ | Geometrically consistent tracks |
| 4 | Top up via Shi-Tomasi + BRIEF | $\lambda_2$ detection, quadtree, BRIEF | $\mathcal{I}_k$, occupancy mask | New candidates $\mathcal{S}_k$ |
| 5 | Re-ID against dormant buffer | `SpatialDescriptorMatcher` | $\mathcal{S}_k$, $\mathcal{D}_k$ | Resurrected track IDs |
| 5b | TrackLocalMap | `SpatialDescriptorMatcher` | $\mathcal{S}_k^{\text{unmatched}}$, $\mathcal{L}_k$, $\hat{\mathbf{T}}_{c_kw}$ | New tracks bound to existing map points |
| 6 | Motion-only BA | LM + Huber | All correspondences | Refined $\mathbf{T}_{c_kw}$ |
| 7 | Keyframe insertion + descriptor update | ORB-SLAM3 + §5.1 | Map points observed at keyframe | Updated $\bar{\mathbf{d}}^{(i)}$ |

End of frame; loop to next frame.

---
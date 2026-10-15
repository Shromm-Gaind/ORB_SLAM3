# Hybrid KLT–Descriptor Frontend for ORB-SLAM3 in Turbid Underwater Environments

> **Revision 3 (September 2026).** Records the C++ integration into ORB-SLAM3 (§9.2 steps 2b–6) and what it showed on a Godot stereo simulation. The new **§13** holds the in-system results; edits elsewhere are marked *(rev 3)*. Principal changes:
>
> - §4.1 is now enforced in code. The first C++ port violated it — the frontend never learned which tracks were adults — and produced exactly the duplicate landmarks §4.1 predicts (§13.8). §4.8, §7.4.1 and §7.4.2 contradicted §4.1 in revision 2 and are corrected.
> - Frontend detection is multi-octave with the §4.5.2 rank rule; the PoC and first port detected at level 0 only. Stall-frame KLT retention rose from 51% to 92% (§13.4).
> - Frontend descriptors were unsteered BRIEF: `cv::ORB::compute` does not compute orientation for caller-supplied keypoints. Fixed with an explicit IC angle; §12.1–12.2 should be re-measured (§13.3).
> - §7.1 vocabulary retraining is not required on the simulation: BoW is not the loop-closure bottleneck (§13.9).
> - An upstream `LoopClosing.cc` defect that discards the best-BoW keyframe is fixed; the first genuine loop closure followed (§13.9).
> - Step 5b (§4.7) and §4.1 bookkeeping are implemented as step 6; in-system validation is pending (§13.2).

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

$$
\hat{\mathbf{u}}_k^{(i)} = \pi\!\left(\mathbf{K}, \mathbf{T}_{c_kw} \cdot \mathbf{p}_w^{(i)}\right)
$$

where $\pi(\cdot)$ is the pinhole projection with distortion.

3. Search for the descriptor match within a window $\mathcal{W}(\hat{\mathbf{u}}_k^{(i)}, r)$ of radius $r$ around the prediction, accepting the candidate $j^*$ with minimum Hamming distance:

$$
j^* = \arg\min_{j \in \mathcal{W}} \, d_H\!\left(\mathbf{d}_{k-1}^{(i)},\, \mathbf{d}_k^{(j)}\right)
$$

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

$$
\Delta \mathbf{v}_t = \mathbf{H}^{-1} \mathbf{b}_t, \quad \mathbf{H} = \sum_{\mathbf{x} \in \Omega} \nabla \mathcal{I}(\mathbf{x}) \nabla \mathcal{I}(\mathbf{x})^\top
$$

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

$$
\tau = \left(\, \mathrm{id}_\tau,\, \mathbf{u}_k^\tau,\, \mathbf{d}_\tau,\, \ell_\tau,\, a_\tau,\, j_\tau \,\right)
$$

where:
- $\mathrm{id}_\tau \in \mathbb{N}$ is a globally unique landmark identifier
- $\mathbf{u}_k^\tau \in \mathbb{R}^2$ is the current pixel observation
- $\mathbf{d}_\tau \in \{0,1\}^{256}$ is the descriptor. **Revised (§12.1):** this is the descriptor recomputed at the track's *most recent verified observation*, not the birth descriptor. Measurement showed the birth descriptor drifts to a median Hamming distance of 48/256 from the track's own current appearance, which alone capped Step 5 recall at ~47%. After triangulation it is promoted to the representative descriptor of the map point (Section 5)
- $\ell_\tau \in \mathbb{N}$ is the pyramid level (octave) at which the feature was originally detected
- $a_\tau \in \mathbb{N}$ is the track age (frames since birth)
- $j_\tau$ is an optional reference to a 3D map point $\mathbf{p}_w^{(j_\tau)}$ in the global map (None if not yet triangulated)

A **dormant track buffer** $\mathcal{D}_k$ holds **infant tracks only** — tracks where $j_\tau = \emptyset$ (never triangulated to a map point) that died in the last $\Delta_{\text{dormant}}$ frames, keyed by their last observed pixel and their **death-time descriptor** (§12.1). Entries additionally store $f_\tau^{\text{died}}$ (the frame of death, used for the gap-scaled acceptance threshold of §4.6) and $a_\tau^{\text{died}}$ (age at death, carried back on resurrection).

Only tracks with $a_\tau \ge a_{\min}$ are buffered. Short-lived tracks are predominantly detector noise, and admitting them in bulk floods the buffer with impostors for $\Delta_{\text{dormant}}$ frames; in the unilluminated section of our sequence this inflated the buffer from ~1,000 to ~7,400 entries (§12.3).

**Adult tracks ($j_\tau \neq \emptyset$) that lose KLT tracking are discarded from the frontend entirely.** They are not added to $\mathcal{D}_k$. The rationale is twofold:

1. The corresponding map point persists in the backend's local map. As long as the landmark remains in the camera frustum, Step 5b (TrackLocalMap) will project it and re-acquire it using its representative descriptor $\bar{\mathbf{d}}^{(i)}$ — which is multi-observation and far more robust than the single-shot birth descriptor stored in the dormant buffer.

2. Allowing the same physical landmark to exist in both $\mathcal{D}_k$ (with its weak birth descriptor) and $\mathcal{L}_k$ (with its strong representative descriptor) creates a race condition between Steps 5 and 5b. A new Shi-Tomasi candidate could match the dormant entry first, inheriting only a track ID with no map point reference — corrupting the data association by creating an "orphan" track that fails to re-bind to its map point.

This makes Steps 5 and 5b operate on disjoint regions of the landmark state space: Step 5 handles infants (no map point); Step 5b handles adults (map point exists). Together they cover all reachable lost-track recoveries.

The complete track lifecycle:

| Phase | Active in $\mathcal{T}_k$? | In $\mathcal{D}_k$? | In local map $\mathcal{L}_k$? | Descriptor used for matching |
|---|---|---|---|---|
| **Infant** | Yes | No | No | Last-observation descriptor |
| **Infant dies** | No | Yes | No | Death-time descriptor (§12.1) |
| **Infant resurrected** | Yes | No | No | Fresh descriptor; dormant entry's descriptor retained as an observation |
| **Adult** | Yes | No | Yes | Multi-observation representative descriptor |
| **Adult dies** | No | No | Yes | Representative descriptor used by Step 5b |
| **Adult re-acquired** | Yes | No | Yes | Representative descriptor |

Transitions out of the dormant buffer happen via Step 5 (match $\rightarrow$ active) or via expiration (frame age > $\Delta_{\text{dormant}} \rightarrow$ permanently discarded).

*(rev 3)* **Enforcement.** The adult/infant distinction is only as good as the frontend's knowledge of which tracks hold a map point. In the C++ system Tracking reports this every frame after `TrackLocalMap`, and clears it on every map reset (§13.2). Without that report every track looks like an infant, adults enter $\mathcal{D}_k$, and resurrections come back as the orphan tracks described above — observed in the first port as duplicated landmarks (§13.8).

### 4.2 Step 1: Pyramidal Lucas-Kanade tracking

Build Gaussian pyramids $\mathcal{P}_{k-1} = \{\mathcal{I}_{k-1}^{(0)}, \ldots, \mathcal{I}_{k-1}^{(L)}\}$ and $\mathcal{P}_k$ similarly, where $\mathcal{I}^{(l+1)}$ is $\mathcal{I}^{(l)}$ downsampled by factor 2 after Gaussian smoothing.

> **On the two pyramid scale factors (factor 2 here vs 1.2 in §4.5).** These are deliberately different and do not need to be unified. The KLT pyramid is internal scratch space for coarse-to-fine flow: it consumes and returns level-0 pixel coordinates, and no octave label passes through it. The 1.2 pyramid of §4.5 defines the scale at which descriptors are computed and is what $\ell_\tau$ indexes. Forcing KLT onto a 1.2 pyramid would require ~8 levels to reach the displacement coverage that 3 levels at factor 2 provide (level 3 at 1.2 is only $1.73\times$ downsampled), materially weakening large-motion tracking for no benefit. $\ell_\tau$ is metadata describing a feature's characteristic scale — consumed by descriptor extraction, $r_{\text{TLM}}(\ell)$, and the BA information matrix — not a tracking parameter.

For each track $\tau \in \mathcal{T}_{k-1}$, find the displacement $\mathbf{v}_\tau \in \mathbb{R}^2$ that minimizes the SSD over a window $\Omega$ centered on $\mathbf{u}_{k-1}^\tau$:

$$
\mathbf{v}_\tau^* = \arg\min_{\mathbf{v}} \sum_{\mathbf{x} \in \Omega} \Big[\, \mathcal{I}_{k-1}(\mathbf{u}_{k-1}^\tau + \mathbf{x}) - \mathcal{I}_k(\mathbf{u}_{k-1}^\tau + \mathbf{x} + \mathbf{v}) \,\Big]^2
$$

Solved iteratively from coarsest to finest pyramid level, with the coarse displacement initializing the fine search. The Lucas-Kanade update at each iteration is:

$$
\Delta \mathbf{v} = \mathbf{H}^{-1} \mathbf{b}, \qquad \mathbf{H} = \sum_\Omega \nabla \mathcal{I}_k \nabla \mathcal{I}_k^\top, \qquad \mathbf{b} = \sum_\Omega \nabla \mathcal{I}_k \cdot \Delta \mathcal{I}
$$

The new observation is $\mathbf{u}_k^\tau = \mathbf{u}_{k-1}^\tau + \mathbf{v}_\tau^*$.

We warp at translation-only resolution within each pyramid level. Affine warping (as in SVO) would improve robustness to viewpoint changes but is significantly more expensive. For our environment the dominant nuisance is appearance change rather than geometric warping, so translational KLT is expected to enough. This is worth revisiting if results don't match.

### 4.3 Step 2: Forward-backward consistency filter

Re-track each $\mathbf{u}_k^\tau$ backward from $\mathcal{I}_k$ to $\mathcal{I}_{k-1}$, obtaining $\tilde{\mathbf{u}}_{k-1}^\tau$. Compute the FB error:

$$
\epsilon_\tau^{FB} = \left\| \mathbf{u}_{k-1}^\tau - \tilde{\mathbf{u}}_{k-1}^\tau \right\|_2
$$

and reject tracks where $\epsilon_\tau^{FB} > \theta_{FB}$ (typically $\theta_{FB} = 1$ px). This catches the case where KLT converges to a different nearby corner — the round trip would not return to the origin.

### 4.4 Step 3: Geometric outlier rejection

Estimate the fundamental matrix $\mathbf{F}_{k-1,k}$ via RANSAC over the surviving tracks, using:

$$
\mathbf{u}_k^\tau{}^\top \, \mathbf{F}_{k-1,k} \, \mathbf{u}_{k-1}^\tau = 0
$$

for true correspondences (homogeneous coordinates). Reject tracks whose Sampson distance to the recovered epipolar geometry exceeds $\theta_{RANSAC}$.

For stereo configurations we additionally enforce the stereo epipolar constraint between left and right images via precomputed extrinsics. This is cheap and removes mismatched stereo pairs.

The set of tracks surviving Steps 1–3 is denoted $\mathcal{T}_k^{\text{surv}}$.

### 4.5 Step 4: Track top-up via Shi-Tomasi + BRIEF

Let $N_{\text{target}}$ be the desired active track count (typically 500–1000, matching ORB-SLAM3 default scale). If $|\mathcal{T}_k^{\text{surv}}| < N_{\text{target}}$, spawn new tracks.

**4.5.1 Build occupancy mask.** To avoid clumping new detections near existing tracks, construct:

$$
\mathcal{M}_k(\mathbf{x}) = \begin{cases} 0 & \text{if } \exists\, \tau \in \mathcal{T}_k^{\text{surv}} : \|\mathbf{x} - \mathbf{u}_k^\tau\|_\infty < r_{\text{mask}} \\ 1 & \text{otherwise} \end{cases}
$$

**4.5.2 Multi-scale Shi-Tomasi detection and the octave rule.** Detection runs at **every level** of an ORB-SLAM3-style pyramid with scale factor $s_p = 1.2$ and $n_{\text{lev}} = 8$ levels. Multi-scale detection is mandatory, not optional: detecting only at level 0 would emit descriptors at a single scale, whereas DBoW2's vocabulary is populated from descriptors across scales, and a revisit from a different standoff distance requires the matching octave to exist (§12.6).

For each level $\ell$, compute $\lambda_2(\mathbf{H}(\mathbf{x}))$ over the level image restricted to $\mathcal{M}_k$, apply non-maximum suppression at radius $r_{\text{nms}}$, and retain candidates above a nominal noise floor.

**Selection rule (revised — see §12.4).** Allocate a per-level target using ORB-SLAM3's geometric distribution,

$$
N^{(\ell)} = N_{\text{target}} \cdot \frac{1 - s_p^{-1}}{1 - s_p^{-n_{\text{lev}}}} \cdot s_p^{-\ell},
$$

then select $N^{(\ell)}$ corners per level by **quadtree distribution ranked on $\lambda_2$**. The absolute value of $\theta_{ST}$ is therefore almost irrelevant — it acts only as a noise floor, and rank selection determines the output.

This replaces the adaptive per-level threshold $\theta_{ST}^{(\ell)} = q \cdot \max_\ell \lambda_2$ that an earlier revision of this document proposed. That proposal rested on the premise that $\lambda_2$ decays at higher octaves because Gaussian smoothing suppresses gradients. **Measurement on our footage shows the opposite**: $\lambda_2$ *rises* monotonically with level, with the p99 response $2.44\times$ higher at level 7 than at level 0 (§12.4). Downsampling concentrates coarse structure into sharper per-pixel gradients faster than the anti-alias blur destroys fine texture. A max-normalised threshold is also fragile: it divides by a single-pixel outlier, so one specular highlight sets the threshold for an entire octave. Rank selection is insensitive to the absolute response scale in either direction.

Each retained corner is lifted to level-0 coordinates and stamped with $\ell_\tau = \ell$ and patch size $31 \cdot s_p^{\ell}$, so that BRIEF is computed at the scale the feature was detected at, and so $r_{\text{TLM}}(\ell)$ and the BA weight $\sigma^{-2}_\ell$ behave as stock ORB-SLAM3 expects.

**Octave is frozen at birth for infants (§12.5).** A KLT track is detected once and then followed — median lifetime 206 frames in the well-lit portion of our sequence — so its octave label ages. Measurement confirms real drift accumulates (mean $|\Delta\ell|$ rises $+0.79$ octaves above the age-1 noise floor by age 121+), but also that no image-response estimate of the current octave is good enough to correct it: re-describing at the instantaneous best octave is *worse* than the frozen label by 12–22 bits at every age, because the estimator's own noise floor (mean $|\Delta\ell| = 1.08$, disagreeing by $\ge 1$ octave in 54% of samples at age 1, where the true scale has not yet changed) exceeds the drift being corrected. Turbid coral has no dominant characteristic scale, so a per-level argmax is near a coin flip. Once a track is triangulated, ORB-SLAM3's `PredictScale` derives the octave from geometry instead, which is the principled fix and is available from stereo depth at every frame.

**4.5.3 Quadtree spatial distribution.** The quadtree recursively subdivides each level's image region until each leaf contains at most one keypoint or the per-level target $N^{(\ell)}$ is reached, retaining the highest-$\lambda_2$ corner per leaf. This is the spatial-spread benefit of ORB-SLAM3's design, decoupled from FAST.

**4.5.4 Orientation.** For each retained corner $\mathbf{u}$, compute orientation via intensity centroid:

$$
m_{pq} = \sum_{x,y} x^p y^q \mathcal{I}(x + u_x,\, y + u_y), \qquad \theta = \arctan_2(m_{01},\, m_{10})$$

with sums over a circular patch of radius 15 px (ORB default).

**4.5.5 BRIEF descriptor.** Compute the steered BRIEF descriptor:

$$
d_i = \begin{cases} 1 & \text{if } \mathcal{I}(\mathbf{R}_\theta \mathbf{a}_i + \mathbf{u}) < \mathcal{I}(\mathbf{R}_\theta \mathbf{b}_i + \mathbf{u}) \\ 0 & \text{otherwise} \end{cases}, \quad i = 1 \ldots 256
$$

where $(\mathbf{a}_i, \mathbf{b}_i)$ are the ORB-SLAM3 BRIEF sampling pattern and $\mathbf{R}_\theta$ is the 2D rotation by $\theta$ for rotation invariance.

**4.5.6 Stereo: per-frame depth assignment for new features (stereo / RGB-D only).** In stereo configurations, ORB-SLAM3's `Frame` constructor expects every keypoint to have an assigned depth value (or "no depth" sentinel) immediately upon extraction. Local mapping uses this for fast triangulation of new map points without needing temporal parallax. Our hybrid pipeline must preserve this contract:

1. Run Steps 4.5.1–4.5.5 on **both** the left and right rectified images, producing two candidate sets $\mathcal{S}_k^L$ and $\mathcal{S}_k^R$. The dormant buffer, KLT tracking, and active-track set $\mathcal{T}_k$ are maintained over the left image only; the right image is used solely for depth computation at extraction time.

2. For each candidate $(\mathbf{u}_k^{(j),L}, \mathbf{d}_k^{(j),L}) \in \mathcal{S}_k^L$, search along its rectified epipolar line in $\mathcal{S}_k^R$ for a match satisfying:
   - Same pyramid level (within $\pm$1 octave)
   - $|v^L - v^R| < \theta_{\text{stereo-row}}$ (rectified row alignment, typically 1 px)
   - $u^L > u^R$ (positive disparity)
   - $d_H(\mathbf{d}_k^{(j),L}, \mathbf{d}_k^{(j'),R}) < \theta_{\text{stereo-desc}}$
   - SAD block match around the predicted disparity to obtain sub-pixel disparity

3. Compute depth from disparity using the stereo baseline $b$ and focal length $f$:

$$
z^{(j)} = \frac{f \cdot b}{u^{L,(j)} - u^{R,(j)}}
$$

and attach it to the left-image candidate. Candidates with no valid right-image match are kept as monocular features (no depth), exactly as stock ORB-SLAM3 does for points outside the stereo overlap region.

4. KLT tracking (Steps 1–3) operates on left-image positions only. After Step 3, **recompute disparity for each surviving track each frame** via SAD block matching on a small search range around the previous frame's disparity. This is the cheapest way to maintain valid stereo depths through KLT-tracked frames; the alternative of independently KLT-tracking both left and right images is more expensive and offers no accuracy benefit because the rectified epipolar constraint already pins the right position to a 1D search.

The resulting set of new candidate features $\mathcal{S}_k = \{(\mathbf{u}_k^{(j)},\, \ell_k^{(j)},\, \mathbf{d}_k^{(j)},\, z_k^{(j)})\}$ is bit-compatible with stock ORB-SLAM3 stereo features (including the convention that $z = \emptyset$ marks monocular-only features). For monocular configurations, ignore 4.5.6 entirely; the set $\mathcal{S}_k$ has no depth field and triangulation proceeds normally via temporal parallax in local mapping.

### 4.6 Step 5: Re-identification of dormant tracks

For each new candidate $(\mathbf{u}_k^{(j)},\, \mathbf{d}_k^{(j)}) \in \mathcal{S}_k$, search the dormant buffer $\mathcal{D}_k$ for matches:

$$
\tau^* = \arg\min_{\tau \in \mathcal{D}_k \,\cap\, \mathcal{W}(\mathbf{u}_k^{(j)},\, r_{\text{reid}})} \, d_H\!\left(\mathbf{d}_\tau, \mathbf{d}_k^{(j)}\right)
$$

where $\mathcal{W}(\cdot, r_{\text{reid}})$ is the spatial window around the candidate. **Motion compensation is mandatory, not optional (§12.1):** every dormant entry's stored pixel is advanced each frame by the dominant image motion,

$$
\mathbf{u}_\tau \leftarrow \mathbf{u}_\tau + \operatorname{median}_{\tau' \in \mathcal{T}_k^{\text{surv}}} \left( \mathbf{u}_k^{\tau'} - \mathbf{u}_{k-1}^{\tau'} \right),
$$

taken over RANSAC-inlier survivors, so $\mathbf{u}_\tau$ is a *prediction* in the current frame rather than a stale position at death. This permits $r_{\text{reid}}$ to be halved (20 → 10 px), which reduces the spatial candidate count and suppresses coincidental matches quadratically.

**Gap-scaled acceptance threshold (§12.2).** A single Hamming ceiling is the wrong shape for this problem, because descriptor drift grows with the dormancy gap $g = k - f_\tau^{\text{died}}$. Accept if

$$
d_H\!\left(\mathbf{d}_{\tau^*},\, \mathbf{d}_k^{(j)}\right) \;\le\; \theta_{\text{reid}}(g) \;=\; \min\!\left(\theta_{\text{cap}},\; \theta_{\text{base}} + \beta \, g\right),
$$

with $\theta_{\text{base}}$, $\beta$ and $\theta_{\text{cap}}$ calibrated from the measured drift-vs-gap curve of §12.2 (values in §8).

**Distinctiveness gate.** Optionally require the best candidate to beat the second-best *spatially-gated* candidate by a margin $\delta$:
$d_H^{(2)} - d_H^{(1)} \ge \delta$. On self-similar texture this is the standard defence against descriptor aliasing; a query that cannot be resolved is better left unmatched, since §4.6's failure mode is benign. Ablation on our data found $\delta$ contributes nothing once the gap-scaled threshold and local detection are in place (§12.1), so it defaults to off.

If accepted:
- The new feature inherits $\mathrm{id}_{\tau^*}$ (preserving track identity for accumulating parallax toward eventual triangulation)
- $\tau^*$ is removed from $\mathcal{D}_k$
- The re-identified track remains an **infant** ($j_\tau = \emptyset$); it does not inherit any map point reference because by construction (§4.1) dormant tracks have $j_\tau = \emptyset$

Otherwise the candidate is assigned a fresh $\mathrm{id}$ and added to $\mathcal{T}_k$ as a new track.

This step uses descriptors only for **local re-identification of dropped tracks**, never for primary frame-to-frame matching. The candidate set is small (typically O(10–100) dormant tracks within $r_{\text{reid}}$), the spatial prior is strong, and we only need to confirm "same landmark" rather than pick one from thousands. This is a much easier descriptor task than vanilla ORB matching.

**Important asymmetry**: in Step 5, both sides of the match are single-observation descriptors — the dormant track died before being promoted to a map point with multiple observations. Step 5 is therefore the inherently weakest descriptor step in the pipeline, and the system must tolerate its failures gracefully. A missed re-ID is not catastrophic: the landmark gets a new ID and is retriangulated, which is wasteful but recoverable.

**4.6.1 Targeted local detection in dormant windows (new — §12.1).** Step 5 can only re-identify a landmark if Step 4 happened to place a candidate near it. Because Step 4's budget is the global deficit $N_{\text{target}} - |\mathcal{T}_k^{\text{surv}}|$ — often only ~20 corners spread across the whole image — a recently-died landmark's location usually receives no candidate at all. This was the single largest failure mode measured: 12,385 of 41,544 forced-kill events failed because no corner was ever detected at the location, versus 10,201 that failed in the matcher.

We therefore add a second, targeted detection pass. Within each dormant prediction window $\mathcal{W}(\mathbf{u}_\tau, r_{\text{reid}})$, take the strongest $\lambda_2$ response above a **relaxed** threshold $\kappa \cdot \theta_{ST}$ with $\kappa < 1$, respecting the occupancy mask $\mathcal{M}_k$ and a minimum separation from already-selected corners. This is directly analogous to ORB-SLAM3's search-by-projection, which likewise looks where the map predicts a landmark rather than relying on global detection to find it.

These corners are **re-ID candidates only**. If a local corner fails to match a dormant track it is discarded, never spawned as a new track — so the relaxed threshold cannot pollute the map with weak corners. Two invariants follow, and both are enforced in the implementation:

1. Total additions to $\mathcal{T}_k$ in one frame (resurrections plus fresh spawns) may not exceed the deficit, so $N_{\text{target}}$ is honoured exactly.
2. Resurrections are applied **before** fresh spawns when the budget binds. Recovering a landmark's identity is worth more than adding an anonymous corner: a fresh corner can be spawned next frame, whereas a dormant entry expires and its identity is lost permanently.

Measured effect: re-ID recall 59.6% → 78.6%, no-corner failures 6,147 → 1,388, and — counter-intuitively — *fewer* wrong associations, because giving the true landmark a candidate lets it claim its own dormant entry before an impostor can (§12.1).

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

$$
j^* = \arg\min_{j \,\in\, \mathcal{S}_k^{\text{unmatched}} \,\cap\, \mathcal{W}(\hat{\mathbf{u}}_k^{(i)},\, r_{\text{TLM}}(\hat{\ell}_k^{(i)}))} \, d_H\!\left(\bar{\mathbf{d}}^{(i)},\, \mathbf{d}_k^{(j)}\right)
$$

where $\bar{\mathbf{d}}^{(i)}$ is the **representative descriptor** of map point $i$ (Section 5) and $r_{\text{TLM}}$ scales with the predicted pyramid level (typically 2–4 px at level 0, growing geometrically).

Accept if $d_H(\bar{\mathbf{d}}^{(i)},\, \mathbf{d}_k^{(j^*)}) < \theta_{\text{TLM}}$. The new correspondence $(\mathbf{u}_k^{(j^*)},\, \mathbf{p}_w^{(i)})$ is added to $\mathcal{T}_k$ with $\mathrm{id}_\tau$ inherited from the map point and a **new KLT track is initialized** at that corner so subsequent frames track it via Step 1.

### 4.8 Steps 5 and 5b share a primitive

Both steps are spatially-gated descriptor matches against a small candidate set:

| Step | Predict location from | Match against | Result if accepted |
|---|---|---|---|
| 5 (Re-ID) | Dormant track's last pixel + motion prior | Recently-died **infant** descriptors | Resurrect track ID. No map point to re-link: $\mathcal{D}_k$ holds infants only (§4.1). *(rev 3 — revision 2 read "re-link to existing map point if it had one", contradicting §4.1)* |
| 5b (TrackLocalMap) | Map point reprojection under provisional pose | Local map point representative descriptors | Initialize new KLT track tied to existing map point |

The shared primitive is `SpatialDescriptorMatcher`: given a list of `(prediction_pixel, query_descriptor)` queries and a list of `(candidate_pixel, candidate_descriptor)` candidates, return the best match per query under spatial and Hamming-distance gates. This is implemented as a standalone class, unit-testable on synthetic inputs.

The primitive carries four gates, all exercised by the Python reference implementation and required for C++ parity:

| Gate | Purpose | Used by |
|---|---|---|
| Per-query radius $r$ | $L_\infty$ spatial window; per-query override supports $r_{\text{TLM}}(\ell)$ without forcing every caller to grow it | 5, 5b |
| Per-candidate Hamming ceiling | Each candidate judged against its own threshold; carries the gap-scaled $\theta_{\text{reid}}(g)$ of §4.6 | 5 |
| Distinctiveness margin $\delta$ | Best must beat second-best *spatially gated* candidate by $\delta$ bits, including candidates above their own ceiling (a near competitor is evidence of ambiguity regardless) | 5, 5b |
| Unique candidates | Each candidate awarded to at most one query; lower Hamming wins, ties break to lower query index | 5b (and 5, where it also has a useful side effect — a correct re-ID consumes both the dormant entry and the corner an impostor would otherwise claim) |

### 4.9 Step 6: Motion-only bundle adjustment

The correspondence set entering pose estimation is the union of:
- KLT-tracked correspondences from Steps 1–3
- Re-identified correspondences from Step 5
- TrackLocalMap correspondences from Step 5b

Estimate $\mathbf{T}_{c_kw}$ by:

$$
\mathbf{T}_{c_kw}^* = \arg\min_{\mathbf{T}} \sum_{\tau \,:\, j_\tau \ne \emptyset} \rho\!\left( \left\| \mathbf{u}_k^\tau - \pi(\mathbf{K},\, \mathbf{T} \cdot \mathbf{p}_w^{(j_\tau)}) \right\|_{\Sigma_\tau}^2 \right)
$$

where $\rho(\cdot)$ is a Huber robust kernel and $\Sigma_\tau$ is the observation covariance scaled by pyramid level (higher octave $\rightarrow$ higher uncertainty), as in ORB-SLAM3.

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

$$
\mathcal{O}_i = \left\{ \mathbf{d}_{k_1}^{(i)},\, \mathbf{d}_{k_2}^{(i)},\, \ldots,\, \mathbf{d}_{k_n}^{(i)} \right\}
$$

where each entry is the descriptor computed at keyframe $k_j$'s observation of landmark $i$. Descriptors are added only at keyframes, not every frame, both for storage efficiency and because non-keyframe descriptors add noise without information (consecutive frames yield highly correlated descriptors).

The **representative descriptor** used for matching (in Steps 5b, loop closure, relocalization) is the observation with minimum median Hamming distance to all other observations:

$$
\bar{\mathbf{d}}^{(i)} = \arg\min_{\mathbf{d} \in \mathcal{O}_i} \; \mathrm{median}_{\mathbf{d}' \in \mathcal{O}_i \setminus \{\mathbf{d}\}} \, d_H(\mathbf{d},\, \mathbf{d}')
$$

This is exactly ORB-SLAM3's `MapPoint::ComputeDistinctiveDescriptors`. The median-of-pairwise-distances criterion picks the "most central" observation in the cluster, which is robust to outliers from single bad-quality observations (motion blur, transient occlusion, brief turbidity spike).

### 5.2 Where the descriptor at a keyframe observation comes from

Two cases:

**Case A**: at this keyframe, the track was matched in Step 5b (TrackLocalMap), so a fresh Shi-Tomasi corner was detected with an explicit BRIEF descriptor. Use that descriptor directly.

**Case B**: at this keyframe, the track was carried by KLT from a previous frame. KLT only updates pixel position; it does not produce a descriptor. We have two options:

1. **Compute BRIEF on the fly at the current KLT pixel location.** Requires running the orientation step and BRIEF computation, but the corner location is already known. Cost: ~1 ms per track in C++.
2. **Reuse the descriptor from the most recent keyframe where the track was matched in Step 5b.** Cheaper but staler.

We choose option 1. The compute cost at keyframe rate ($\ll$ frame rate) is negligible, and it ensures the keyframe's descriptor reflects the keyframe's actual appearance rather than a prior keyframe's. This is the same choice ORB-SLAM3 implicitly makes (its descriptors are always computed at the current keyframe).

### 5.3 Update schedule

- $\mathcal{O}_i$ grows by one entry each time the corresponding track is observed in a new keyframe.
- $\bar{\mathbf{d}}^{(i)}$ is recomputed whenever $\mathcal{O}_i$ changes.
- For long-lived map points, $|\mathcal{O}_i|$ may be bounded (ORB-SLAM3 does not explicitly bound it; the local map size limits the memory working set anyway). In our environment, where individual surfaces may be tracked across hundreds of keyframes, an explicit cap (e.g., 50) may be warranted for memory reasons.

### 5.4 The asymmetry between active tracks, dormant tracks, and map points

| State | Descriptor available | Used in |
|---|---|---|
| Active track, not yet a map point | Last-observation descriptor | Carried forward through KLT |
| Dormant track | Death-time descriptor (§12.1) | Step 5 re-ID |
| Map point with observations | Representative (medoid) descriptor | Step 5b TLM, loop closure, relocalization |

Step 5 is the weakest because both sides are single-observation. Step 5b is much stronger because the local map side uses the multi-observation representative descriptor.

**The representative descriptor is right for wide baselines and wrong for Step 5 (§12.1, §12.5).** This is worth stating explicitly because it is counter-intuitive. Substituting the medoid for the death-time snapshot in the dormant buffer produced *no* improvement in re-ID (recall 78.6% vs 79.3%) and a *worse* drift curve (gap-1 p95 of 59 vs 57 bits). The reason is that Step 5's dormancy gaps are 1–2 frames, so the comparison is against a corner detected essentially *now*: a snapshot taken one frame ago is closer to "now" than a medoid averaged over up to 24 frames of a track's life, during which the vehicle moved and the viewpoint changed. The medoid is central to a track's *history*; Step 5 needs its *most recent* appearance.

The opposite holds at wide baselines, and the same measurements make the case. A track's birth descriptor drifts to a mean Hamming distance of ~50/256 from its own appearance by age 121+ (§12.5) — at ORB-SLAM3's `TH_LOW` and therefore effectively unmatchable. Any consumer matching against long-lived tracks (Step 5b, relocalization, loop closure) must use the representative descriptor, never a single observation. Sections 5.1–5.3 are therefore mandatory for §4.7 and §6, and deliberately *not* used by §4.6.

---

## 6. Backend Integration (Unchanged from Stock ORB-SLAM3)

### 6.1 What the backend expects

The backend operates on keyframes containing `(KeyPoint, descriptor, octave)` tuples. As long as our frontend produces these in the standard format — which it does, since BRIEF computed at Shi-Tomasi corners is bit-identical to BRIEF computed at FAST corners — the backend is agnostic to the data association method used upstream.

### 6.2 Bag-of-Words for place recognition

DBoW2 converts each keyframe's descriptor set into a compact BoW vector $\mathbf{v}_{\text{BoW}} \in \mathbb{R}^V$ over a pre-trained vocabulary of $V$ visual words (a hierarchical k-means tree built offline). Place recognition queries are nearest-neighbor lookups in BoW space, scored by:

$$
s(\mathbf{v}_1,\, \mathbf{v}_2) = 1 - \frac{1}{2} \left\| \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|_1} - \frac{\mathbf{v}_2}{\|\mathbf{v}_2\|_1} \right\|_1
$$

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
3. Build the hierarchical k-means tree with DBoW2's `create_voc_step` utility. Standard parameters: branching factor 10, depth 6 $\rightarrow$ $10^6$ visual words.
4. Save the resulting vocabulary file and point ORB-SLAM3 at it via the config.

This is offline, one-time, roughly 30 minutes of compute. **Skip this step and the system will produce confidently wrong loop closures.**

**Status (rev 3): not required on the simulation dataset.** Instrumenting every loop-closure gate (§13.9) showed Shi-Tomasi descriptors producing 200–290 BoW matches per candidate against the stock vocabulary — four times FAST's 24–73 — and candidates being proposed readily. Loop closure fails at geometric verification, where Sim3 correctly rejects aliased candidates. The perceptual-aliasing risk described above is real, but it is a property of uniform biofouling, and retraining would not remove it; a vocabulary trained on uniform texture could plausibly make it worse. Retraining remains an option for real data and should then use the same preprocessing as runtime (CLAHE included), left-camera frames only, balanced across water types, and exclude evaluation sequences. Parameter note: $10^6$ words suits a ~10,000-image corpus; with a few thousand images use $k = 10, L = 5$ and aim for ≥ 30 descriptors per word.

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
- *(rev 3)* No dormant entry holds a `MapPoint*` (§4.1): adults are discarded on death, never buffered. Revision 2 listed the opposite invariant ("re-identified tracks inherit the `MapPoint*` from the dormant track"). Enforced in `DormantTrackBuffer::add` by assertion.
- *(rev 3)* No active track is outside the image. KLT can report success for a point that has drifted off-image; downstream stereo matching indexes rows by `pt.y` unchecked (§13.3).
- *(rev 3)* Per frame, `idmatch` ≤ KLT survivors. An id-matched map point must come from a track that survived KLT; exceeding the survivor count implies id aliasing.

**7.4.2 Modularization.** The two new components are pure data structures with no ORB-SLAM3 dependencies:

- `DormantTrackBuffer`: ring buffer of `(id, last_pixel, descriptor, frame_died, octave, age_at_death)` entries — infants only, so the `MapPoint*` slot is always null (§4.1; *rev 3* correction). Public interface: `add`, `purge_older_than`, `translate_all` (motion compensation), `query_within`, `remove`, `gap_frames`. Built and unit-tested standalone; byte-identical to the Python reference on a shared parity harness.
- `SpatialDescriptorMatcher`: takes lists of queries and candidates, returns best match per query under spatial + Hamming gates. Stateless. Built and unit-tested standalone.

This separates algorithmic correctness (do the modules work in isolation on synthetic inputs?) from integration correctness (does modified `Tracking.cc` call them correctly?).

**7.4.3 Map-corruption canaries.** Add lightweight runtime checks that detect known corruption patterns:

- Sudden jumps in map point count ($\Delta$ > some threshold per frame): likely indicates a bad keyframe insertion.
- Average reprojection error rising: BA is failing, likely due to bad correspondences.
- Loop closure inlier ratio < threshold but acceptance still occurring: vocabulary or geometric verification misconfigured.
- *(rev 3)* The settings dump at startup, checked against the yaml. A numeric key read with the boolean pattern (`readParameter<int>(...) != 0`) silently becomes 1 (§13.3).
- *(rev 3)* `adults` (dead tracks discarded as adults) at zero over many frames: Tracking is not reporting map points to the frontend, and §4.1 is silently off.

Log these continuously; if any tripped, dump the relevant frames + correspondences to disk for post-mortem.

### 7.5 KLT track persistence under appearance change

**Risk**: KLT is patch-tracking, not identity-tracking. When patch appearance changes substantially (viewpoint, lighting, partial occlusion), KLT's gradient descent terminates, the track dies, and a new track is spawned in the same region with a new ID. The Step 5 re-identification mechanism is designed to resurrect these tracks, but if Step 5 fails (descriptors too noisy on biofouling) we accumulate "ghost" landmarks — the same physical feature tracked under multiple IDs in the map.

**Mitigation**: this risk is real and inherent to the architecture. The local mapping thread's existing map point fusion logic (which merges map points that project to within a threshold of each other across multiple keyframes) provides a secondary cleanup mechanism. Ghost landmarks should eventually be detected and merged at the next BA pass that covers both observations.

If this becomes a significant problem in practice, a more aggressive map point fusion criterion in local mapping (smaller threshold, more frequent fusion attempts) is a tunable mitigation.

*(rev 3)* **Observed.** Ghost landmarks appeared in the first C++ port, from the §4.1 violation rather than from Step 5 failing (§13.8). Separately, the return pass of a loop duplicated structure even without re-ID. Fusion does not catch that case: after ~60 s of drift the old landmarks project outside the search window and the revisit keyframes share no covisibility with the originals, so only a verified loop closure fuses them.

---

## 8. Parameters

Starting values; all are subject to empirical tuning on real data.

| Symbol | Meaning | Start | Notes |
|---|---|---|---|
| $L$ | KLT pyramid levels | 3 | Increase for large frame-to-frame motion |
| $\|\Omega\|$ | KLT window size | 21×21 px | Larger helps low-texture |
| $\theta_{FB}$ | FB error threshold | 1.0 px | Tighter is stricter |
| $\theta_{RANSAC}$ | Sampson distance threshold | 3.0 px | Calibration-dependent |
| $N_{\text{target}}$ | Target active track count | 1000 | Matches ORB-SLAM3. *(rev 3)* Bounded above by scene capacity: at 2000 the frontend churned ~1000 untrackable corners per frame on the simulation (§13.4) |
| $N_{\text{min}}$ | Track count below which tracking is "lost" | 50 | Triggers relocalization |
| $r_{\text{mask}}$ | Mask radius for new detections | 10 px | Anti-clumping |
| $\theta_{ST}$ | Shi-Tomasi $\lambda_2$ noise floor | quality 0.01 | Acts only as a floor; rank selection dominates (§12.4) |
| $s_p$ | Descriptor pyramid scale factor | 1.2 | ORB-SLAM3 standard; distinct from the KLT pyramid's factor 2 (§4.2) |
| $n_{\text{lev}}$ | Descriptor pyramid levels | 8 | Gives a $1.2^7 = 3.58\times$ standoff window (§12.6) |
| $a_{\min}$ | Min track age to enter dormant buffer | 3 frames | Noise suppression (§4.1) |
| $\Delta_{\text{dormant}}$ | Dormant buffer depth | **10 frames** | Revised from 30. Resurrection latency is overwhelmingly 1–2 frames; a shorter horizon roughly halves the false-association exposure at a cost of ~5 points of recall (§12.1) |
| $r_{\text{reid}}$ | Re-ID spatial search radius | **10 px** | Halved from 20, enabled by motion compensation (§4.6) |
| $\theta_{\text{base}}$ | Re-ID Hamming ceiling at gap 1 | **63/256** | Fitted to the p95 of the measured same-point drift curve (§12.2) |
| $\beta$ | Gap scaling of $\theta_{\text{reid}}$ | **1.2 bits/frame** | From the same fit |
| $\theta_{\text{cap}}$ | Ceiling on $\theta_{\text{reid}}(g)$ | **86/256** | Beyond this the same-point and different-point modes overlap. *(rev 3)* Unreachable with $\Delta_{\text{dormant}} = 10$: $63 + 1.2 \cdot 10 = 75$, and saturation needs $g \approx 19$, so $\beta$ is the live knob. Fit taken on unsteered descriptors (§13.3) |
| $\delta$ | Distinctiveness margin | **0 (off)** | Ablation showed no measurable benefit once §4.6.1 is in place |
| $\kappa$ | Local-detection quality scale | 0.3 | Fraction of $\theta_{ST}$ inside dormant windows (§4.6.1) |
| $r_{\text{TLM}}(\ell)$ | TLM search radius at octave $\ell$ | $2 \cdot 1.2^\ell$ px | Geometric scaling. *(rev 3)* Implemented as $r_0 \cdot th \cdot 1.2^\ell$ with $r_0 = 3.0$ and $th$ stock's search-width multiplier, matching stock's effective 2.5–4 px window |
| $\theta_{\text{TLM}}$ | TLM Hamming threshold | 50/256 | Standard ORB-SLAM3 value |
| $\theta_{\text{stereo-row}}$ | Stereo row alignment tolerance (rectified) | 1 px | Tighter for well-calibrated cameras |
| $\theta_{\text{stereo-desc}}$ | Stereo descriptor Hamming threshold | 50/256 | Same as TLM |

*(rev 3)* Implementation parameters added during integration (yaml key in brackets):

| Parameter | Default | Meaning |
|---|---|---|
| multi-octave detection [`Hybrid.multiscale`] | on | Off gives the PoC's level-0-only ablation (§13.4) |
| deficit gate [`Hybrid.minDeficitFraction`] | 0 | Skip Step 4/5 until the deficit reaches this fraction of $N_{\text{target}}$; 0.10 amortises detection. Also skips re-ID (§13.5) |
| dormant descriptor [`Hybrid.dormantStoreRepresentative`] | 0 | 0 stores the death-time snapshot per §5.4; 1 stores the medoid (PoC behaviour) |
| $r_0$ [`Hybrid.tlmRadius`] | 3.0 px | Step 5b base radius |
| $\theta_{\text{TLM}}$ [`Hybrid.tlmHamming`] | 50 | Step 5b Hamming ceiling |
| Step 5b $\delta$ [`Hybrid.tlmMargin`] | 0 | Distinctiveness margin for 5b; 0 off |
| octave gate [`Hybrid.tlmOctaveGate`] | on | Require $|\ell_{kp} - \hat\ell| \le 1$, as stock |
| descriptor pyramid | $s_p$, $n_{\text{lev}}$ from `ORBextractor` | Must equal the extractor's, since `mvInvLevelSigma2[octave]` is indexed with frontend octaves |

---

## 9. Implementation Plan

### 9.1 ORB-SLAM3 fork structure

We fork `ORB_SLAM3` and modify the following files:

| File | Modification |
|---|---|
| `include/Frame.h`, `src/Frame.cc` | Add tracking state fields (track IDs per keypoint); leave keypoint/descriptor representation unchanged |
| `include/ORBextractor.h`, `src/ORBextractor.cc` | Replace FAST detection with multi-scale Shi-Tomasi inside the existing pyramid + quadtree structure, using the per-level target + rank selection rule of §4.5.2 (**not** an adaptive per-level threshold — see §12.4). Stamp `kp.octave` and `kp.size` = $31 \cdot 1.2^\ell$. Keep BRIEF computation unchanged |
| `include/Tracking.h`, `src/Tracking.cc` | Replace `TrackWithMotionModel` / `TrackReferenceKeyFrame` with the hybrid pipeline (Steps 1–5). Existing `TrackLocalMap` modified to Step 5b semantics (operate on $\mathcal{S}_k^{\text{unmatched}}$). `Relocalization` unchanged |
| `include/MapPoint.h`, `src/MapPoint.cc` | No changes — `ComputeDistinctiveDescriptors` already does what we need |
| (new) `include/DormantTrackBuffer.h`, `src/DormantTrackBuffer.cc` | New module |
| (new) `include/SpatialDescriptorMatcher.h`, `src/SpatialDescriptorMatcher.cc` | New module. Must carry all four gates of §4.8 |
| `include/LoopClosing.h`, `src/LoopClosing.cc` | *(rev 3)* One fix: use the computed `nIndexMostBoWMatchesKF` in `DetectCommonRegionsFromBoW` (upstream defect, §13.3) |
| `include/LocalMapping.h`, `src/LocalMapping.cc` | No changes |
| `Vocabulary/*` | *(rev 3)* Unchanged; retraining not required on the simulation (§7.1) |
| *(rev 3)* (new) `include/HybridFrontend.h`, `src/HybridFrontend.cc` | Steps 1–5 engine, OpenCV only (§13.1) |
| *(rev 3)* (new) `src/FrameHybrid.cc` | `Frame` constructor taking left features from the frontend |
| *(rev 3)* (new) `src/TrackingHybrid.cc` | `HybridTrackWithMotionModel`, `HybridSearchLocalPoints` (Step 5b), `SyncHybridMapPoints`, `ClearHybridMapPoints` |
| *(rev 3)* `include/Settings.h`, `src/Settings.cc` | `Hybrid.*` yaml keys and settings-dump lines |
| *(rev 3)* `include/ORBextractor.h` | `ComputePyramid` made public (left pyramid without detection) |
| *(rev 3)* root `CMakeLists.txt` | C++17, keeping `COMPILEDWITHC14` defined alongside `COMPILEDWITHC17` |

Estimate: ~500–1000 lines of net new C++ code, plus ~500 lines of modified existing code. Several weeks of focused work.

### 9.2 Suggested order of work

1. **Standalone modules first** — **COMPLETE.** (no ORB-SLAM3 dependency). `DormantTrackBuffer`, `SpatialDescriptorMatcher`. Full unit tests on synthetic inputs. Should compile and pass tests outside the ORB-SLAM3 source tree.
2. ~~**Python proof-of-concept of Steps 1–5**~~ — **COMPLETE (§12).** Implemented in `hybrid_frontend.py` with a track-ID system over KLT, a forced-failure harness with a permutation-controlled chance floor, and diagnostics for descriptor drift, track lifetime, octave selection, octave staleness and scale invariance. Both §10 targets met. Outcomes that changed the design before any C++ was written: death-time descriptors (§4.1), motion-compensated dormant predictions and gap-scaled thresholds (§4.6), targeted local detection in dormant windows (§4.6.1), rank-based octave selection (§4.5.2), frozen octave for infants (§12.5), and representative descriptors confined to wide-baseline consumers (§5.4).

2b. **Bring the C++ modules to parity with the Python reference** — **COMPLETE.** `translate_all`, `age_at_death`, `octave` and `gap_frames` added to `DormantTrackBuffer`; per-candidate Hamming ceiling and distinctiveness margin added to `SpatialDescriptorMatcher`. A cross-language parity harness produces byte-identical output against the unmodified Python modules, covering the boundary-inclusive horizon, the $L_\infty$ window, the underflow-safe gap, and every matcher gate.

3. **Modify `ORBextractor`** to detect Shi-Tomasi instead of FAST. Run stock ORB-SLAM3 with this change on standard datasets (KITTI, TUM). Sanity check: performance should be roughly equivalent — Shi-Tomasi is not dramatically different from FAST on terrestrial imagery, so this swap alone shouldn't break anything. If KITTI/TUM results regress significantly, we have a bug. *(rev 3)* **COMPLETE.** On the simulation the Shi-Tomasi extractor avoids the track loss stock FAST suffers on the same sequence.
4. ~~**Retrain vocabulary**~~ — *(rev 3)* **NOT REQUIRED on the simulation** (§7.1, §13.9). Revisit for real data.
5. **Integrate Steps 1–5 into `Tracking.cc`**. Maintain a feature flag (`USE_HYBRID_FRONTEND`) to allow A/B comparison with stock ORB-SLAM3. *(rev 3)* **COMPLETE** as shadow mode (Stage A) and takeover (Stage B), switched by `Hybrid.shadowMode` / `Hybrid.takeover` (§13.1).
6. **Integrate Step 5b** (modify TrackLocalMap to operate on $\mathcal{S}_k^{\text{unmatched}}$). *(rev 3)* **IMPLEMENTED, validation pending**, together with §4.1 adult/infant bookkeeping (§13.2).
7. **End-to-end testing** on full sequences. Compare ATE/RPE against stock ORB-SLAM3 on our own data and on KITTI as a non-regression check. *(rev 3)* **IN PROGRESS** on the simulation; matched repeats outstanding (§13.7, §13.11).

### 9.3 Tests

Beyond unit tests on the new modules, end-to-end tests should include:

- **Non-regression**: hybrid frontend on KITTI / TUM-RGBD sequences should produce ATE within 10% of stock ORB-SLAM3.
- **Improvement metric**: hybrid frontend on our underwater sequences should produce successful tracking through segments where stock ORB-SLAM3 loses tracking, quantified by tracking success rate (fraction of frames with valid pose) and ATE on segments with ground truth (if available; otherwise inspection).
- **Loop closure regression**: synthetic loop closure sequences (revisit known structures) should produce loop closure events with stable inlier ratios. False loop closures on unrelated segments should be zero or near-zero.
- **Loop closure baseline comparison**: stock ORB-SLAM3 in stereo mode already closes the loop on our primary sequence. The hybrid frontend must be benchmarked directly against that run — same sequence, same parameters — since it is the only ground truth we have for open question 4 (keypoint repeatability on a genuine revisit). *(rev 3)* On the simulation (§13.9), stock's recovery was a **map merge after track loss**, not a loop closure; every stock loop candidate failed Sim3. Confirm the event type on the real sequence from its log (`Loop detected` against `*Merge detected`) before using it as a baseline.
- **Python/C++ parity**: the C++ `DormantTrackBuffer` and `SpatialDescriptorMatcher` must reproduce the Python reference implementation's behaviour on the shared unit-test corpus, including the boundary-inclusive horizon, the $L_\infty$ window shape, tie-breaking to the lower query index, and the per-candidate threshold and margin gates.

---

## 10. Validation Questions — Status

The three questions posed in the original revision, and where they now stand. Full method and numbers in §12.

| # | Question | Status | Result |
|---|---|---|---|
| 1 | Does descriptor matching work at keyframe scale on our imagery? | **Superseded** | The premise was wrong. Stock ORB-SLAM3 in stereo mode closes the loop on this dataset over ~2,000 frames, so ORB matching demonstrably works at keyframe *and* loop scale. The live question became whether **Shi-Tomasi** features preserve that, answered in §12.4 and §12.6: scale allocation is near-identical to FAST (TVD 0.064) and wide-baseline matching is equivalent (at $2\times$ standoff, 185 correct matches vs ORB's 161). *(rev 3: this refers to the real sequence. On the simulation, stock's recovery was a merge after track loss — §13.9.)* |
| 2 | What is the empirical re-ID rate of Step 5? | **Answered — both targets met** | Recall **80.8%** on recoverable kills (target > 80%); true mis-association **0.25%** falling to **0.02%** with all fixes, statistically indistinguishable from the chance floor of the scoring protocol ($z = +0.1$) (target < 1%). §12.1. |
| 3 | How frequent are open-water sections? | **Partially answered** | Not measured directly, but the RANSAC-survival proxy classifies ~5% of frames as degraded. Those frames dominate the failure statistics out of all proportion to their number (§12.3), so §7.3's mode-switching is a central concern, not a marginal one. |

Two questions that were not in the original list have been added by the experimental work and remain open:

4. **Does Shi-Tomasi fire on the same physical structures on a genuine revisit?** Everything measured so far concerns descriptors and scale statistics. Keypoint *repeatability* under real appearance change — illumination, viewing angle, backscatter, biological change between passes — cannot be simulated and needs the C++ integration, benchmarked against the existing stereo ORB-SLAM3 run. *(rev 3)* **Partially answered on the simulation:** revisits yield 200–290 BoW matches per candidate, and the one genuine revisit passed full geometric verification (§13.9). Real-data repeatability remains open.
5. **What standoff variation does the survey actually contain?** §12.6 quantifies matching as a function of the standoff ratio $s$. Converting that into a prediction for our dataset requires the empirical distribution of $s$ between first and second pass, obtainable from the stereo depth of the existing run.
6. *(rev 3)* **Is appearance-based place recognition viable on uniform biofouling?** On the simulation BoW proposes abundant, overwhelmingly aliased candidates, and Sim3 correctly rejects nearly all of them (§13.9). This is now the main limit on loop closure, and it is environmental.
7. *(rev 3)* **How do §12.1–12.2 change with steered descriptors?** The PoC's descriptors were very likely unsteered (§13.3); the drift fit behind $\theta_{\text{base}}$, $\beta$ and $\theta_{\text{cap}}$ needs re-measuring.

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
| 7 | Keyframe insertion + descriptor update | ORB-SLAM3 + §5.1 | Map points observed at keyframe | Updated configuration arrays |

End of frame; loop to next frame.

---

---

## 12. Experimental Validation

All experiments in this section were run on a 2,213-frame stereo survey of a coral wall (CLAHE-preprocessed, 1280×720, right camera), traversed right-to-left, revisiting the opening view at the end. Frames ~850–1400 are an unilluminated section imaging the reverse side of the structure; where a result is materially different there, both segments are reported. Except where stated, the well-lit segment [0, 800) is used.

*(rev 3)* **Caveat on descriptors.** Descriptors here were computed with OpenCV's ORB on caller-supplied keypoints, which does not estimate orientation, so they were very likely unsteered BRIEF (§13.3). Results comparing descriptors of the same point over short baselines (§12.1–12.2) are internally consistent but should be re-measured with steered descriptors before being treated as final. The in-system results on the simulation are in §13.

The Python reference implementation and the harnesses described here are:
`hybrid_frontend.py`, `dormant_buffer.py`, `spatial_descriptor_matcher.py` (pipeline);
`run_hybrid_poc.py` (`--mode normal|forced-fail|lifetime`);
`multiscale_shitomasi.py`, `octave_rule_test.py`, `octave_staleness_test.py`, `scale_invariance_test.py` (octave and scale studies).

### 12.1 Step 5 re-identification: forced-failure test

**Method.** Each frame, a random 2% of healthy active tracks are force-killed into the dormant buffer, simulating KLT failure while retaining ground truth: we know the identity and location of every killed landmark. An event resolves as *correct* (the original ID returns near its motion-propagated expected location), *incorrect* (a foreign ID claims that location, or the right ID returns outside tolerance), *missed*, or *left-FOV* (the expected location has moved outside the usable image area — unrecoverable by construction and therefore excluded from the recall denominator).

Two methodological points matter for interpreting the numbers:

- **Scoring geometry must match matcher geometry.** With the scoring tolerance at 5 px while $r_{\text{reid}}$ was 10 px, 2,463 events were scored "incorrect" that were in fact the correct landmark returning inside the matcher's legitimate window. Aligning the two eliminated that category entirely (2,463 → 0).
- **The chance floor must be measured, not assumed.** RANSAC-free coincidence — an unrelated resurrection happening to land in a kill window — is not negligible at these densities. We pair every real kill with a **decoy** event placed at a kill location sampled from ≥ 1 horizon earlier in the sequence, closed with the same exposure window as its partner and scored with the same tolerance. A decoy has no identity, so every hijack it collects is chance. This is a permutation control against the sequence's real spatial clustering, replacing an earlier uniform-density model that saturated (predicting more coincidences than were observed) and could not rank configurations.

**Results.** Cumulative effect of the design changes, at $\Delta_{\text{dormant}} = 10$, $r_{\text{reid}} = 10$ px:

| Configuration | Recall (recoverable) | Missed: no corner | Missed: matcher | True mis-association (excess over chance) |
|---|---|---|---|---|
| Birth descriptor, fixed $\theta = 50$, $r = 20$ | 47.2% | — | — | not separable |
| + death-time descriptor, motion comp., $a_{\min}$ | 59.8% | 13.0% | 12.1% | — |
| + gap-scaled $\theta_{\text{reid}}(g)$ | 62.6% | 12.7% | 9.4% | — |
| + aligned scoring, permutation control | 62.2% | 15.8% | 16.4% | 0.25% ($z = +2.0$) |
| **+ local detection in dormant windows (§4.6.1)** | **80.8%** | **3.4%** | **11.8%** | **0.02% ($z = +0.1$)** |

The final configuration meets both §10 targets. Precision of attempted re-IDs is 92.8%.

**Ablations at the final configuration:**

| Ablation | Recall | Missed: no corner | True mis-association |
|---|---|---|---|
| Full | 80.8% | 3.4% | 0.02% ($z=+0.1$) |
| Without local detection (§4.6.1) | 61.6% | 15.4% | 0.65% ($z=+5.4$) |
| Without representative descriptor | 81.5% | 3.5% | 0.06% ($z=+0.5$) |
| Distinctiveness margin $\delta = 15$ | 80.7% | 3.5% | 0.02% (unchanged) |

Local detection is the dominant contribution and *improves* precision as well as recall. The representative descriptor and the distinctiveness margin are both neutral-to-negative here and are disabled in Step 5 (see §5.4 for why the former is nonetheless mandatory elsewhere).

**Caveat for reporting.** Forced kills sample *healthy* tracks. Real KLT failures occur precisely because appearance or geometry has broken down, so genuinely-dead tracks are harder to re-detect and match. 80.8% is an upper bound on real-world re-ID performance and should be quoted as "on synthetically-failed tracks".

### 12.2 Descriptor drift as a function of dormancy gap

**Method.** The threshold $\theta_{\text{reid}}$ must be calibrated on the population Step 5 actually faces: tracks that *died*, matched against a freshly detected corner. Calibrating instead on KLT *survivors* — features stable enough to keep tracking — understates the drift substantially and led to an early threshold of 32 that was far too strict.

We therefore instrument the forced-failure run with **drift probes**: up to 300 concurrent killed landmarks are followed for the full horizon regardless of whether their event resolved (sampling only unresolved events would bias toward the hard cases). Each frame, the stored dormant descriptor is compared against a descriptor recomputed at the motion-propagated location and against the nearest freshly detected corner.

The at-corner measurement is contaminated: when the true landmark was not re-detected, the nearest corner is a *different* physical point whose descriptor sits near the random-pair distance (~128 bits). Taking a high percentile over that mixture measures the wrong-point mode. We therefore require a sample to lie within 3 px *and* below 90 bits to count as same-point evidence, and report the contamination fraction.

**Results** (same-point mode, well-lit segment, $n = 12{,}147$; contamination 13%):

| Gap (frames) | mean | median | p90 | **p95** |
|---|---|---|---|---|
| 1 | 23.3 | 20 | 46 | **56** |
| 2 | 25.8 | 21 | 53 | **64** |
| 3–4 | 28.8 | 24 | 58 | **72** |
| 5–8 | 31.9 | 26 | 66 | **77** |
| 9–15 | 40.7 | 36 | 79 | **85** |
| 16–30 | 47.2 | 48 | 83 | **86** |

Fitting $\theta(g) = \theta_{\text{base}} + \beta g$ to the p95 column gives $\theta_{\text{base}} \approx 63$, $\beta \approx 1.2$, saturating near 86 — the values adopted in §8. Two contrasts are worth recording:

*(rev 3)* With $\Delta_{\text{dormant}} = 10$ the effective ceiling tops out at $63 + 1.2 \cdot 10 = 75$, so $\theta_{\text{cap}}$ never binds. The fit should be repeated with steered descriptors (§13.3).

- **Birth vs death-time descriptor.** Over the same run, the birth descriptor sits at mean 55.0 / median 48 bits from a track's current appearance, while consecutive-frame drift is mean 12.0 / median 11. Storing the birth descriptor placed the true match *above* a threshold of 50 roughly half the time, which is exactly why recall was pinned near 47%.
- **The 13% contamination figure is itself a useful result**: when the prediction is good, Shi-Tomasi re-detects the same physical corner ~87% of the time. Detector repeatability at short baselines is therefore not the bottleneck.

The p95 at long gaps presses against the 90-bit contamination filter, meaning the same-point and different-point distributions begin to overlap by gap ~16. This, not the fit, is the justification for capping $\theta_{\text{cap}}$ at 86 and for shortening $\Delta_{\text{dormant}}$ to 10.

### 12.3 Track lifetime: what re-ID buys the backend

Re-ID recall is an internal metric. What the backend consumes is track *identity*: a landmark held under one ID for 40 frames yields a 40-observation constraint for bundle adjustment, whereas the same landmark fragmented into four 10-frame IDs yields four weak constraints and four duplicate map points. We therefore run the same footage twice, with Step 5 enabled and disabled, and compare span distributions (span = last frame seen − first frame seen + 1, so bridging a gap counts as one landmark).

**Well-lit segment [0, 800):**

| Metric | re-ID ON | re-ID OFF | Change |
|---|---|---|---|
| Distinct track IDs created | 3,344 | 3,990 | −16.2% |
| Median span (frames) | **206** | 137 | **+50.4%** |
| Mean observations per ID | 253.5 | 212.4 | +19.3% |
| Fraction of IDs surviving ≥ 20 frames | **0.907** | 0.787 | +15.2% |
| Fraction of IDs surviving ≥ 30 frames | 0.887 | 0.752 | +17.9% |
| Fraction of IDs surviving ≥ 60 frames | 0.834 | 0.699 | +19.2% |

**Whole sequence**, for contrast: 100,104 vs 130,856 IDs (−23.5%), median span 1.0 frame in *both* configurations, ≥ 20 frames 0.115 vs 0.081 (+42%).

The two tables tell different stories and both belong in any write-up. In nominal conditions the frontend produces long-lived landmarks and re-ID extends them by half again. In the unilluminated section the frontend creates ~300 new IDs per frame against a target of 1,000 active — churning its entire feature set every three frames — and roughly 75,000 of the sequence's 100,000 IDs originate there. Re-ID's *relative* benefit is larger in that section (+42% vs +15% at ≥ 20 frames), i.e. it helps most where conditions are worst, but it cannot rescue a segment in which the detector itself is producing noise. The honest summary is: **median track lifetime 206 frames in nominal conditions, collapsing to ~1 frame in the unilluminated section.**

### 12.4 The octave rule

**Q1 — does $\lambda_2$ decay with pyramid level?** No; it rises. Averaged over 30 frames of the well-lit segment, with $s_p = 1.2$:

| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| p99 $\lambda_2$ relative to L0 | 1.00 | 1.09 | 1.26 | 1.47 | 1.69 | 1.92 | 2.19 | **2.44** |

This refutes the premise behind adaptive per-level thresholding. On turbid coral the level-0 image is dominated by fine, low-contrast texture carrying little corner energy; downsampling concentrates coarser structure into sharper per-pixel gradients faster than the anti-alias blur removes detail. Note also that the median $\lambda_2$ is ~0 at every level: the response distribution is extremely heavy-tailed, which is a second reason to avoid normalising by $\max \lambda_2$.

**Q2 — which selection rule fills the octaves?** None starves: every level clears its target several times over under all three rules (absolute, relative, rank). Rank selection nonetheless produces 2.8× more candidates at L0 and 4.4× more at L1, giving the quadtree a much richer pool to distribute from for the same output count, and requires no per-dataset threshold tuning. Adopted (§4.5.2).

**Q3 — scale allocation vs FAST.** With rank selection, Shi-Tomasi and ORB/FAST allocate features across octaves almost identically:

| Octave | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Shi-Tomasi (fraction) | 0.210 | 0.194 | 0.178 | 0.132 | 0.086 | 0.074 | 0.066 | 0.059 |
| ORB / FAST (fraction) | 0.243 | 0.180 | 0.148 | 0.112 | 0.098 | 0.085 | 0.074 | 0.060 |

Total variation distance **0.064**. Because DBoW2's vocabulary is populated from the descriptor population FAST produces across scales, this is direct evidence that the §7.1 vocabulary-mismatch risk is *not* driven by scale statistics. If loop closure degrades after the port, keypoint repeatability on revisit (open question 4) is the more likely cause.

### 12.5 Octave staleness over a track's life

ORB-SLAM3 re-detects every frame, so octave is re-estimated every frame; a KLT track is detected once and followed for a median of 206 frames. We measure whether the frozen birth label goes stale, and whether correcting it helps.

**Method.** Track features from six anchors for up to 200 frames. Each frame, estimate the instantaneous best octave, and re-describe the track twice — at the frozen birth octave and at the current best octave — comparing both to the birth descriptor. Because §12.4 established that $\lambda_2$ rises with level, a raw cross-level argmax would be biased toward coarse octaves; we instead compare each level's response to that level's own p99.

**Results** ($n = 69{,}605$ track-age samples):

| Age (frames) | mean $\lvert\Delta\ell\rvert$ | excess over age-1 floor | Hamming (frozen) | Hamming (adaptive) |
|---|---|---|---|---|
| 1 | 1.08 | — (noise floor) | 7.7 | 30.2 |
| 2–5 | 1.13 | +0.04 | 11.1 | 33.1 |
| 6–15 | 1.24 | +0.16 | 15.3 | 37.2 |
| 16–30 | 1.29 | +0.21 | 21.1 | 41.0 |
| 31–60 | 1.41 | +0.32 | 28.2 | 46.3 |
| 61–120 | 1.60 | +0.52 | 36.9 | 52.1 |
| 121+ | 1.87 | **+0.79** | **49.4** | 61.7 |

Two conclusions, and a third finding that matters more than either:

1. **Real drift exists.** Excess above the age-1 noise floor climbs monotonically to +0.79 octaves; the fraction differing by ≥ 2 octaves rises from 0.25 to 0.43. Over a 200-frame track the landmark's apparent scale moves by roughly one octave.
2. **But it cannot be corrected from image response.** Adaptive loses at every age. The decisive figure is age 1, where the true scale has barely changed yet switching octave costs 22 bits: the estimator's noise floor (mean $|\Delta\ell| = 1.08$, 54% flipping by ≥ 1) exceeds the drift it is chasing. Turbid coral has no dominant characteristic scale. Hence §4.5.2's rule: **freeze at birth for infants; use `PredictScale` from stereo depth once triangulated.**
3. **The birth descriptor is unusable at long track ages.** The frozen-Hamming column reaches 49.4 bits by age 121+ — at ORB-SLAM3's `TH_LOW` of 50. A long-lived track's birth descriptor no longer matches its own current appearance. This is the quantitative case for §5's representative descriptor in every wide-baseline consumer.

### 12.6 Scale invariance: how far can standoff change?

**Method.** Rescaling a frame by $1/s$ simulates viewing the scene from $s$ times the distance and gives exact ground truth (a feature at $(x,y)$ maps to $(x/s, y/s)$), so detector repeatability, octave shift and descriptor matchability can be separated without RANSAC in the loop. The governing relation from `MapPoint::PredictScale` is

$$
\ell_{\text{far}} = \ell_{\text{near}} - \frac{\log s}{\log 1.2}, \qquad s = d_{\text{far}} / d_{\text{near}}.
$$

**Results** (well-lit segment, 15 frames, $N_{\text{target}} = 1000$; "#correct" is geometrically verified correct matches):

| $s$ | predicted $\Delta\ell$ | measured $\Delta\ell$ | repeatability | median Hamming | #correct (Shi-Tomasi) | #correct (ORB) |
|---|---|---|---|---|---|---|
| 1.00 | 0.00 | −0.00 | 0.93 | 0.0 | 1134 | 1090 |
| 1.20 | −1.00 | −0.79 | 0.68 | 2.6 | 697 | 715 |
| 1.44 | −2.00 | −1.40 | 0.57 | 5.7 | 474 | 536 |
| 1.70 | −2.91 | −1.80 | 0.48 | 30.6 | 241 | 208 |
| 2.00 | −3.80 | −2.15 | 0.43 | 35.9 | **185** | 161 |
| 2.50 | −5.03 | −2.47 | 0.38 | 47.0 | 122 | 124 |
| 3.00 | −6.03 | −2.62 | 0.33 | 58.1 | 86 | 93 |

The $s = 1.00$ row is a control confirming the harness (zero octave shift, zero Hamming, 0.93 repeatability limited only by border effects).

**The pyramid wall is real.** Measured shift tracks prediction near $s = 1$ then bends away and saturates near −2.6 by $s = 3$ against a predicted −6.03. Features that "should" move to level −3 have nowhere to go: there are no negative pyramid levels. A landmark born at level $\ell_{\text{birth}}$ carries a usable standoff window of $[\,d_{\text{birth}} \cdot 1.2^{-(n_{\text{lev}}-1-\ell_{\text{birth}})},\ d_{\text{birth}} \cdot 1.2^{\ell_{\text{birth}}}\,]$ — so a feature born at level 0 can only be re-observed *closer*, one born at level 7 only *further*, and only the high-octave subset survives a large increase in standoff. With ~36% of features at levels 4–7 (§12.4), a $2\times$ backing-off retains roughly a third of them.

**But degradation is graceful, and Shi-Tomasi matches ORB.** At $2\times$ standoff, 185 correct matches remain at 60% precision against ORB-SLAM3's ~20-inlier loop requirement; at $3\times$, 86. Shi-Tomasi and ORB are equivalent end-to-end, splitting the difference in an interesting way: ORB holds better detector repeatability across scale (0.50 vs 0.43 at $s=2$), while Shi-Tomasi produces markedly better descriptors on the points it does re-detect (median Hamming 35.9 vs 57.2; 65% under `TH_LOW` vs 44%). Together with §12.4's TVD of 0.064, this is the second independent line of evidence that **replacing FAST with Shi-Tomasi should not break loop closure.**

**Asymmetry.** Moving *closer* is worse than moving away: at $s = 0.5$ repeatability is 0.21 and precision 0.34, versus 0.43 / 0.60 at $s = 2$. Upsampling cannot recover detail the near view never captured. Mapping from close and revisiting from further is the favourable direction.

**Caveat.** Synthetic rescaling omits perspective change, illumination, viewing angle and backscatter. These figures are an upper bound; if matching failed here it would certainly fail on a real revisit.

---


## 13. C++ Integration and In-System Results *(rev 3)*

§12 validated the design offline on real footage. This section records what happened when the pipeline was built into ORB-SLAM3 and run end to end (§9.2 steps 2b–6). **The dataset is different from §12** and the two should not be compared directly:

- **Godot AUV simulation**, stereo PinHole, 1280×720 at 20 Hz, baseline $b = 0.10$ m, $f_x = 379.35$ px. Ground truth at ~10 Hz over ~78 s.
- **Synthetic coral with deliberately uniform, biofouling-like texture** — representative of real hull and structure fouling, and close to a worst case for appearance-based place recognition (§13.9).
- **Physically-motivated turbidity post-process**: per-channel Beer–Lambert attenuation, Koschmieder airlight veil, albedo-driven haze, mip-blur forward scatter, marine snow. Homogeneous water, single scattering, screen-space. The first and last thirds of the sequence are illuminated; **the middle third is effectively unilluminated and heavily blurred**.
- Standoff stays **below 4 m** throughout, so `Stereo.ThDepth = 40` ($40\,b = 4.0$ m) classifies every stereo point as close; stereo depth is well conditioned at these ranges.
- **Unlike §12, frames reach the frontend without CLAHE** (§13.11).

**Replay rate.** At real-time replay the system does not keep up (§13.5) and frames are dropped nondeterministically, which inflates apparent inter-frame motion: mean flow was 8.87 px/frame at rate 1 against 2.43 at `--rate 0.2`, where every frame is processed. Numbers below are labelled with their rate; **rate-1 and rate-0.2 numbers are not comparable.**

### 13.1 Integration architecture

| Component | Role |
|---|---|
| `HybridFrontend` (`include/`, `src/`) | C++ port of `hybrid_frontend.py`, Steps 1–5. OpenCV only, no ORB-SLAM3 dependency. Unit-tested with GTest. Now multi-octave (§13.4) with explicit IC orientation (§13.3) |
| **Stage A — shadow mode** (`Hybrid.shadowMode`) | Frontend runs on every left frame alongside stock tracking and logs a `FrameResult`; drives nothing. Validated the engine on real sequences before it could affect pose |
| **Stage B — takeover** (`Hybrid.takeover`) | `FrameHybrid.cc`: a `Frame` built from the frontend's tracks — left detection skipped, left pyramid still built because `ComputeStereoMatches` reads it for sub-pixel refinement; right extraction, undistortion, stereo matching and grid assignment stock. `mvnTrackIds` carries the track id per keypoint. `HybridTrackWithMotionModel`: map-point association by **track-id lookup against `mLastFrame`** instead of windowed descriptor search, falling back to stock `SearchByProjection` below 20 id matches. `PoseOptimization`, LocalMapping and loop closing unchanged |
| **Step 6** (this revision) | §4.1 enforcement and §4.7 Step 5b — §13.2 |

Per-frame diagnostics: `[Hybrid]` (stage counts, `adults`, `dorm`, flow, per-stage `ms`, octave histogram) and `[HybridTrack]` (`idmatch`, `tlmH`, `tlm`, `N`, `state`).

### 13.2 Step 6: §4.1 enforcement and Step 5b

**§4.1 in code.** Every frame, after `TrackLocalMap`, `Tracking::SyncHybridMapPoints` reports to the frontend which tracks hold a non-outlier map point. The frontend discards those *adults* on death instead of buffering them, counting them in `FrameResult::adults_discarded`; a track whose landmark was culled reads back as `nullptr` and reverts to infant. `ClearHybridMapPoints` runs first in `Reset`, `ResetActiveMap` and `CreateMapInAtlas`, since the handles die with the map. Dormant entries never carry a map point, per §4.1 — **not** the carry-through that §4.8, §7.4.1 and §7.4.2 of revision 2 described; those are corrected in this revision.

**Step 5b (§4.7).** `Tracking::HybridSearchLocalPoints` is stock `SearchLocalPoints` up to the matcher, which becomes a `SpatialDescriptorMatch`:

- queries: current keypoints with no map point ($\mathcal{S}_k^{\text{unmatched}}$), radius $r_{\text{TLM}}(\ell) = r_0 \cdot th \cdot 1.2^{\ell}$ with $r_0$ = `Hybrid.tlmRadius` (3.0) and $th$ stock's search-width multiplier;
- candidates: local map points in the frustum at their projection, carrying `MapPoint::GetDescriptor()` — ORB-SLAM3's `ComputeDistinctiveDescriptors` medoid, so §5.4's representative descriptor comes for free;
- gates: $\theta_{\text{TLM}} = 50$, optional $\delta$, unique candidates (§4.8), and stock's octave band $|\ell_{kp} - \hat\ell| \le 1$ as a post-filter.

An accepted match binds the existing KLT track to the landmark — "adult re-acquired" in the §4.1 lifecycle.

**§5.4 policy.** The dormant buffer now stores the death-time snapshot by default. The PoC and the first port overrode it with the medoid, contrary to §5.4's own measurement; `Hybrid.dormantStoreRepresentative: 1` restores the PoC behaviour for ablation.

**Status: implemented; in-system validation pending.** Acceptance criteria for the first run: `adults` nonzero (proves the sync reaches the frontend); `dorm` and re-ID counts fall (adults were a large share of buffered deaths — expected, not a regression); `tlm` close to `idmatch + tlmH` (Step 5b matches survive pose optimisation).

### 13.3 Defects found and fixed during integration

| Defect | Symptom | Fix |
|---|---|---|
| **Unsteered descriptors.** `cv::ORB::compute` with caller-supplied keypoints does not compute orientation; it uses `kp.angle` as given, and the frontend left the default −1. Every frontend descriptor — in the first C++ port and, by the same call pattern, very likely in the Python PoC — was effectively plain BRIEF | Invisible while frontend descriptors were compared only with each other (re-ID). In Stage B they met `ORBextractor`'s steered descriptors in `ComputeStereoMatches`: **112 stereo points at map initialisation against 433** on the extractor path | Frontend computes the intensity-centroid angle itself — the same `IC_Angle` as `ORBextractor`, on the unblurred pyramid level named by `kp.octave` — before every `compute`. After the fix, `tlm` reached 585 |
| **Off-image KLT tracks.** KLT can report success for a point that has drifted just outside the image | Random-timed segfault: `ComputeStereoMatches` indexes rows by `pt.y` with an unchecked float→`size_t` conversion | Tracks leaving the image are killed in the KLT step; the Frame builder filters again |
| **Adults never flagged.** `set_map_point` was never called in Stage B, so every track looked like an infant | Adults were buffered and resurrected without their map point — the §4.1 orphan race — and re-triangulated as duplicate landmarks (§13.8) | Step 6 (§13.2) |
| **Upstream `LoopClosing.cc`.** In `DetectCommonRegionsFromBoW`, `nIndexMostBoWMatchesKF` is computed and discarded; the line meant to use it, `pMostBoWMatchesKF = vpCovKFi[pMostBoWMatchesKF]`, indexes a vector with a `KeyFrame*`, cannot compile, and is commented out in released ORB-SLAM3 | The Sim3 solver is aimed at the BoW-ranked candidate rather than the keyframe holding most of the matches | Assign `vpCovKFi[nIndexMostBoWMatchesKF]` (guarded). Follow-on: the later `OptimizeSim3` call passes `pKFi`, which now differs from the solver's keyframe and should be changed to match. Worth reporting upstream |
| **Instrumentation trap.** `Sim3Solver::iterate` assigns `nInliers` only on convergence | Every failed attempt prints `inliers=0`, which reads as "no geometric support" but carries no information | Expose `mnBestInliers` through a getter to see how close failures come |
| **Silent configuration corruption.** Copying the boolean read pattern `readParameter<int>(...) != 0` onto numeric keys stores `1` for any nonzero value (`tlmHamming: 50` → 1); a `readParameter` type mismatch against the yaml literal aborts at startup | Would have disabled Step 5b with no warning | Numeric keys read without `!= 0`, with types matching the yaml. The settings dump is the check — add it to the §7.4.3 canaries |

### 13.4 Feature budget and the octave rule in the frontend

The PoC and the first port detected new tracks at pyramid level 0 only and stamped `octave = 0`, even though §4.5.2 and `ORBextractor` already implemented the multi-octave rank rule. The frontend now ports `detect_multiscale_shi_tomasi`: a 1.2 pyramid, per-level geometric allocation, the rank rule with a low relative noise floor, quadtree per level, octave and `size` stamped, descriptors computed at the detected level. Resurrection restores a track's **original** octave — identity includes scale.

The clean control is **stall frames**, where the simulator delivered the identical image repeatedly and KLT should retain everything:

| Configuration | Stall-frame KLT retention | New corners per frame (steady) |
|---|---|---|
| $N_{\text{target}} = 2000$, level 0 only | 973/1905 = **51%** | ~990 |
| $N_{\text{target}} = 1000$, level 0 only | 738/977 = **76%** | ~250 |
| $N_{\text{target}} = 1000$, multi-octave | 919/1000 = **92%** | ~100 |

Two findings:

- **$N_{\text{target}}$ is bounded by scene capacity.** At 2000, once the occupancy mask covered the real corners, the relative floor admitted ~1000 near-flat corners per frame that failed KLT on the next frame even against an identical image. This scene supports roughly 700–900 trackable features. Churn fell ~10× between the first and last rows.
- **The track population drifts to coarse octaves.** The initial histogram matches the allocation (203, 168, 145, 126, 112, 103, 84, 66 against a predicted 217, 181, 151, 126, 105, 87, 73, 60), but by mid-sequence octaves 4–6 hold ~570 of 1000 tracks against an allocation of ~265, and octave 0 falls below octave 5. Coarse features survive KLT roughly twice as well — this footage carries its trackable structure at coarse scales, which is the case `multiscale_shitomasi.py` made. It also means `mvInvLevelSigma2[octave]` does substantial work in Stage B's pose optimisation: had octaves stayed 0, the majority of observations would have claimed level-0 precision, and the chi² test would have culled coarse features as outliers.

### 13.5 Latency

Frontend only (`process_frame`), rate 0.2, 2,160 frames:

| Stage | Mean ± std (ms) | Share |
|---|---|---|
| Steps 1–3 (KLT, FB, RANSAC, bookkeeping) | 19.60 ± 2.78 | 30% |
| Step 4 (multi-octave detection + BRIEF) | 42.59 ± 5.36 | 66% |
| Step 5 (re-ID) | 2.57 ± 1.13 | 4% |
| **Total** | **64.76 ± 7.62** (≈15.4 Hz) | |

- **Re-ID costs 4% of the frontend.** No cost argument against §4.6 survives this measurement.
- **Detection cost is fixed per image, not per feature.** The eight-level 1.2 pyramid is ~3.05× the base image, with level 0 at 33% of pixels and levels 6–7 together at ~4% — so trimming top levels saves almost nothing, and halving $N_{\text{target}}$ barely moved throughput. Sharing one `cornerMinEigenVal` pass per frame, reusing the occupancy mask, and describing only buffer-eligible deaths and stride-due survivors were applied; vectorising the candidate scan made no measurable difference.
- `min_deficit_fraction` (default 0, off) skips Step 4/5 until the deficit reaches a fraction of $N_{\text{target}}$; 0.10 runs detection roughly every third frame in steady state. Skipped frames also skip re-ID, so keep it at 0 for any run whose re-ID figures are reported.
- **`ms_total` excludes `Frame` construction and the backend**, so it is not system throughput. Shadow mode additionally runs two full extractors (~4 Hz observed at 2000 features). Stage B removes the left extractor, so it is expected to be roughly cost-neutral against the extractor path; this was not measured end to end.

### 13.6 Stage B data association

Rate 1, 603 frames, `state = 2` (tracking OK) on every frame:

| Quantity | Mean ± std | Peak |
|---|---|---|
| `idmatch` — map points associated purely by track id | 296.2 ± 141.9 | 521 |
| `tlm` — local-map inliers after pose optimisation | 342.7 ± 132.0 | 585 |
| KLT survivors after RANSAC | 585.4 ± 265.4 | — |

**ID association carries ~86% of tracked local-map points** in normal operation, and the scatter of `tlm` against `idmatch` hugs the 1:1 line. This is the §4 claim — association by tracking rather than by matching — measured in the running system.

**Two stress windows** came within margin of failure. Frames 295–305 (peak flow 36 px/frame): survivors 20–38, `idmatch` 3–7, `tlm` 76–112. Frames 336–348: `tlm` minimum 55, survivors minimum 8 with `new = 1000`, `idmatch` minimum 1. For STEREO, `TrackLocalMap` fails at `tlm < 30` (50 within 20 frames of a relocalisation), which triggers `RECENTLY_LOST` and relocalisation — not `LOST`, which follows only after 3 s of failed relocalisation.

**Below 20 id matches the fallback replaces id associations wholesale** with stock `SearchByProjection`; `idmatch` is recorded before the fallback runs. So in both stress windows Stage B was running as stock ORB-SLAM3. That is graceful degradation to a well-tested path — a design property — but `state = 2` there demonstrates the fallback, not robust id association. A per-frame fallback counter would turn "fraction of frames tracked on ids alone" into a measured number.

**What the collapses are.** In shadow mode, the extractor path held tracking through the same frames with the same Shi-Tomasi detector, re-detecting each frame and re-associating against the map. The corners there are real; what fails is frame-to-frame KLT persistence *without a map anchor* — exactly the gap §4.7 exists to fill. Note that `median_flow` is computed over RANSAC survivors, so during a collapse it is biased toward the least-moving points and is not an independent measure of motion.

### 13.7 Trajectory accuracy

SE3-aligned ATE on the same sequence:

| Configuration | ATE RMSE (m) | Runs | Notes |
|---|---|---|---|
| Stock FAST/ORB, CLAHE | 0.1324 | 1 | Lost tracking ~85 frames, new map, merged back (§13.9) |
| Shi-Tomasi extractor swap only | 0.2519 | 1 | No frontend |
| Stage B | 0.1740 / 0.2123 / 0.1951 (mean 0.194) | 3 | Re-ID on, before Step 6 |
| Stage B + `LoopClosing` fix | 0.1771 | 1 | One genuine loop closure (§13.9) |
| Stage B, worst observed | 0.5220 | 1 | §13.8 |

- **Stage B improves ~23% on the extractor swap alone** (0.194 vs 0.252), with all three runs better — but the swap has one run and run-to-run spread is ±20%. Matched repeats are required before this is a result.
- **The CLAHE row is confounded three ways** (preprocessing, detector, map merge) and cannot be attributed to any one of them. Its error profile also differs in kind: near zero at the start, peaking at 0.32 m through the dark section.
- **The loop-closure run sits inside the Stage B spread.** The correction is visible — APE falls to ~0.09 m just after the closure at ~76% of the sequence, then drift resumes — but one late closure improves consistency more than ATE after global alignment.

### 13.8 Re-identification and landmark duplication

Before Step 6, runs with re-ID enabled produced the worst trajectory (0.522 m) and duplicated map structure more often than the extractor swap alone. That run's signature was unusual: RPE median 0.020 m/s and 0.24°/s — locally as good as the best runs — but a U-shaped APE (0.69 m at the start, minimum 0.24 m near t ≈ 43 s, 0.70 m at the end) around a single 16°/s RPE spike at t ≈ 37 s, with no merge, relocalisation, map reset or `Fail to track local map!` logged. That is a globally rotated but internally consistent trajectory; the likeliest source is local BA reconciling duplicate landmarks, though this was not isolated.

There are **two distinct duplication mechanisms**, with different remedies:

1. **Frame scale — the §4.1 violation.** An adult dies at frame $k$ and, because the frontend did not know it was an adult, enters the buffer. Step 5 resurrects its id at $k+5$. The id lookup only reaches $k+4$, where the track was absent, so it gets no map point; at the next keyframe `CreateNewKeyFrame` triangulates a second landmark beside the surviving original. Both enter local BA as slightly inconsistent constraints on one physical point, and `MapPointCulling` graduates the duplicate normally. This is precisely the "orphan track" §4.1 predicts. **Step 6 removes it**: adults are discarded, and their landmark is re-acquired through Step 5b.
2. **Loop scale.** The extractor swap *without* re-ID or takeover still duplicated structure on the return pass, displaced by roughly the accumulated drift ("closer depth"). After ~60 s of 0.2–0.5 m drift, old landmarks project outside the search window, and revisit keyframes share no covisibility with the originals, so `LocalMapping::SearchInNeighbors`/`Fuse` — the §7.5 cleanup — cannot reach them. Only a verified loop closure (`CorrectLoop`/merge with `SearchAndFuse` under the corrected Sim3) fuses structure across that gap. **No frontend change addresses this**; it is a place-recognition outcome (§13.9).

### 13.9 Place recognition on uniform biofouling

ORB-SLAM3's path is `NewDetectCommonRegions` → `DetectNBestCandidates` (the **top 3 by BoW score, with no absolute threshold**) → `DetectCommonRegionsFromBoW`, whose gates are: ≥ 20 BoW matches pooled over the candidate's covisibility group; a Sim3 RANSAC with ≥ 15 consistent inliers; ≥ 50 projection matches; Sim3 optimisation; ≥ 80 projection matches after optimisation; and consistency across 3 covisible keyframes. Instrumenting each gate:

| | FAST/ORB (stock) | Shi-Tomasi / Stage B |
|---|---|---|
| BoW matches per candidate | 24–73 | 200–290 in illuminated sections; 25–75 in the dark section |
| Candidates proposed | yes | yes — the same few keyframes for hundreds of queries |
| Loop candidates passing Sim3 | **0** | 0 before the `LoopClosing` fix; **1 after** (KF 170 → KF 4) |
| Other recovery | Merge after ~85 frames of track loss and a new map | none needed — no track loss |

Findings:

- **The vocabulary is not the bottleneck.** Shi-Tomasi descriptors produce four times the BoW matches of FAST against the stock vocabulary. §7.1's retraining is not required on this dataset (§7.1 updated).
- **Sim3 rejects aliased candidates correctly.** Failing candidates reach RANSAC with 25–75 valid 3D–3D correspondences and cannot find 15 geometrically consistent ones. The one success used 95 BoW matches, 50 correspondences and 29 inliers, then 133 projection matches and 47 after optimisation — fewer putatives than most failures, but mutually consistent. That contrast is the signature of a true loop versus perceptual aliasing on repetitive texture.
- **Stock's advantage on this sequence was a merge, not a loop closure.** Every stock loop candidate failed Sim3. Its one success was a *merge candidate*, available only because FAST lost tracking for ~85 frames and the Atlas created a second map.
- **The `LoopClosing` fix admitted the first genuine loop closure.** It fired on the correct revisit (the start of the traverse); with a single run this is indicative.

Stated for the thesis: *on uniform biofouling, appearance-based place recognition proposes abundant candidates that are overwhelmingly aliased; geometric verification rejects them correctly, and genuine loops are rare but detectable. The limitation is the environment, not the detector or the vocabulary.*

### 13.10 Evaluation methodology notes

- **Nondeterminism.** Concurrent LocalMapping and interrupted local BA make ORB-SLAM3 nondeterministic; at rate 1, which frames are dropped also varies. Observed ATE spread is ±20%. Report mean ± std over at least 3 runs per configuration; rate 0.2 separates backend variance from input sampling.
- **SE3 alignment does not anchor the start.** Nonzero error at $t = 0$ is expected. A U-shaped APE with small RPE indicates a global rotation between segments; drift produces monotonic growth.
- **Pose association.** One evaluation associated only 320 of 746 estimated poses (median $|dt|$ = 10 ms at ~10 Hz on both streams). Widen the association window and confirm the associated subset is representative.
- **Systematic radial offset.** Every configuration's loop lies 0.2–0.3 m inside ground truth, while best-fit scale is ~0.997. A lever-arm error in the camera-to-base extrinsic would produce exactly this, and would inflate every ATE here roughly equally. Check with Sim3 alignment and without the extrinsic.

### 13.11 Open items

1. **Validate Step 6** against the §13.2 criteria, then repeat the re-ID ablation (`Hybrid.enableReid` 0 vs 1, 3 runs each). With adults discarded this should turn positive — the result that makes §4.6 contribute rather than cost.
2. **Matched repeats**: extractor swap ×3 against Stage B ×3 at the same rate, to support the §13.7 comparison.
3. **Re-measure §12.1–12.2 with steered descriptors.** The drift curve and the $\theta_{\text{base}} = 63$, $\beta = 1.2$, $\theta_{\text{cap}} = 86$ fit were very likely taken on unsteered BRIEF (§13.3). Steering adds rotation invariance but also orientation-estimate noise, so the distribution will shift. Verify first by printing `kp.angle` before `compute` in the Python harness.
4. **CLAHE into the frontend.** §12 was measured on CLAHE-preprocessed frames; the C++ frontend runs on raw grayscale. Contrast normalisation matters for KLT and is a plausible reason CLAHE helped stock in the dark section. Apply the same preprocessing in `GrabImageStereo` before the frontend.
5. **Sim3 near-miss distribution.** With `mnBestInliers` exposed, record how close failing candidates come to 15. Resist lowering the threshold: on this dataset a false loop closure costs far more than a missed one.
6. **Evaluation fixes** from §13.10: association window, extrinsic check.
7. **Future work.** Loop-scale duplication and place recognition on uniform biofouling are properties of the environment, not open bugs — a genuine open problem (e.g. geometry-first or learned place recognition).

### 13.12 Hypotheses tested and rejected

Recorded because each was plausible, and ruling it out constrains the explanation.

| Hypothesis | Evidence against |
|---|---|
| Loop closure fails because the stock vocabulary does not suit Shi-Tomasi descriptors | 200–290 BoW matches per candidate, four times FAST's |
| Solver starved of correspondences (< 15 valid pairs) | 25–75 valid correspondences on failures; RANSAC runs |
| Collapse regions are textureless, and the relative $\lambda_2$ floor manufactures noise corners | The extractor path tracked through the same frames with the same detector against the map. An absolute $\lambda_2$ floor was prototyped and withdrawn |
| Collapses are caused by dropped frames | They persisted at rate 0.2 (mean flow 2.43 px/frame) |
| The 16°/s jump in the worst run was a false merge | No merge, relocalisation or map creation logged in that run |
| Map-point scatter comes from stereo depth error at range | Standoff < 4 m throughout: disparity ≥ 9.5 px, depth error ~0.05 m at 2 m |

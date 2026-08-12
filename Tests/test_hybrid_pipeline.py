"""
test_hybrid_pipeline.py

End-to-end smoke test of the HybridFrontend pipeline on a synthetic
sequence. The sequence is a textured noise image shifted by a few
pixels per frame, so KLT should track essentially every feature
through every frame (zero forced kills).

Validates:
  - initialize() populates active tracks
  - process_frame() runs without exceptions
  - track IDs are preserved across frames when KLT succeeds
  - dormant buffer stays empty when KLT never fails
  - force_kill() correctly moves tracks to dormant
  - re-ID works: after force-killing a track, the next frame's Shi-Tomasi
    detection in the same region resurrects the original ID
"""

import numpy as np
import pytest

from hybrid_frontend import HybridConfig, HybridFrontend


def make_textured_image(seed: int = 0, h: int = 480, w: int = 640) -> np.ndarray:
    """Generate a textured grayscale image with enough corners for
    Shi-Tomasi to fire. Pure uniform noise won't produce stable corners,
    so we blur slightly to make the corners somewhat coherent."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w), dtype=np.uint8)
    import cv2
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift by integer pixels with zero-padding."""
    h, w = img.shape
    out = np.zeros_like(img)
    sx0 = max(0, dx); sx1 = w + min(0, dx)
    dx0 = max(0, -dx); dx1 = w + min(0, -dx)
    sy0 = max(0, dy); sy1 = h + min(0, dy)
    dy0 = max(0, -dy); dy1 = h + min(0, -dy)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


# Use a smaller config so the test runs fast.
def make_test_config() -> HybridConfig:
    return HybridConfig(
        target_active_tracks=100,
        shi_tomasi_quality=0.01,
        shi_tomasi_min_distance=10,
        dormant_horizon_frames=30,
        reid_radius_px=15.0,
        reid_hamming_threshold=80,  # synthetic noise => relax slightly
    )


class TestSmoke:
    def test_initialize_populates_tracks(self):
        front = HybridFrontend(make_test_config())
        img = make_textured_image()
        front.initialize(img)
        assert len(front.active_tracks) > 0, "Expected some Shi-Tomasi corners on textured noise"
        assert front.frame_index == 0

    def test_process_runs_without_exception(self):
        front = HybridFrontend(make_test_config())
        img = make_textured_image()
        front.initialize(img)
        # Shift by 2 px to give KLT a non-trivial but easy task.
        shifted = shift_image(img, dx=2, dy=1)
        res = front.process_frame(shifted)
        assert res.frame_index == 1
        assert res.tracks_in > 0

    def test_track_ids_persist_under_easy_motion(self):
        front = HybridFrontend(make_test_config())
        img = make_textured_image()
        front.initialize(img)
        initial_ids = set(front.active_tracks.keys())
        # 3 frames of easy motion; most tracks should survive.
        prev = img
        for k in range(1, 4):
            curr = shift_image(prev, dx=1, dy=0)
            front.process_frame(curr)
            prev = curr
        # Some IDs from frame 0 should still be alive.
        surviving = initial_ids.intersection(front.active_tracks.keys())
        # On synthetic noise that is shifted exactly, the very large
        # majority should survive. Use a generous lower bound to keep
        # the test robust to OpenCV version drift.
        assert len(surviving) > len(initial_ids) * 0.5, (
            f"Only {len(surviving)}/{len(initial_ids)} IDs survived "
            f"3 frames of trivial motion — KLT is dropping tracks unexpectedly"
        )

    def test_force_kill_moves_to_dormant(self):
        front = HybridFrontend(make_test_config())
        img = make_textured_image()
        front.initialize(img)
        # Process one frame so we have valid current positions.
        front.process_frame(shift_image(img, 1, 0))
        # Pick some tracks to kill.
        ids = list(front.active_tracks.keys())[:5]
        before = len(front.dormant_buffer)
        front.force_kill(ids)
        assert len(front.dormant_buffer) == before + len(ids)
        # Those IDs should be gone from active.
        for tid in ids:
            assert tid not in front.active_tracks

    def test_reid_resurrects_after_force_kill(self):
        """Force-kill a few tracks; on the very next frame, Step 4 should
        spawn new corners in roughly the same locations, and Step 5
        should resurrect the original IDs.

        Two-frame scenario: by force-killing AFTER process_frame at
        frame 1, the next process_frame at frame 2 will run Step 1 on
        the survivors (skipping the killed ones, which are now in the
        dormant buffer), then Step 4 will spawn new corners. Some of
        those should land near the killed locations and re-ID should
        match them.
        """
        front = HybridFrontend(make_test_config())
        img0 = make_textured_image()
        front.initialize(img0)
        img1 = shift_image(img0, 1, 0)
        front.process_frame(img1)

        # Kill some tracks.
        all_ids = list(front.active_tracks.keys())
        # Limit to 20 to avoid emptying the active set.
        kill_ids = all_ids[: min(20, len(all_ids) // 4)]
        killed_locations = {
            tid: (front.active_tracks[tid].x, front.active_tracks[tid].y)
            for tid in kill_ids
        }
        front.force_kill(kill_ids)
        assert len(front.dormant_buffer) >= len(kill_ids)

        # Run the next frame. Step 4 will spawn corners; Step 5 will try
        # to re-ID them against the dormant buffer.
        img2 = shift_image(img1, 1, 0)
        res = front.process_frame(img2)

        # At least one of the kills should be resurrected. (We can't
        # demand ALL, because Step 4 might not pick a corner in every
        # killed region depending on quadtree distribution.)
        resurrected = set(res.resurrected_ids)
        overlap = resurrected.intersection(kill_ids)
        assert len(overlap) > 0, (
            f"No force-killed tracks were resurrected. Killed: {len(kill_ids)}, "
            f"resurrected: {len(resurrected)}, intersection: {len(overlap)}. "
            f"Re-ID is not finding any candidates near killed locations."
        )

    def test_dormant_buffer_stays_small_when_klt_succeeds(self):
        front = HybridFrontend(make_test_config())
        img = make_textured_image()
        front.initialize(img)
        prev = img
        for k in range(1, 10):
            curr = shift_image(prev, 1, 0)
            res = front.process_frame(curr)
            prev = curr
        # On clean synthetic motion, KLT failures should be rare so the
        # dormant buffer should remain small. (Some KLT failures will
        # happen at frame borders where texture moves out of frame; that's
        # expected. Hence "small", not "zero".)
        assert len(front.dormant_buffer) < front.cfg.target_active_tracks // 2, (
            f"Dormant buffer is suspiciously large ({len(front.dormant_buffer)}) "
            f"on clean synthetic motion"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


class TestNewFeatures:
    """Representative descriptor + local dormant-window detection."""

    def _front(self, **kw):
        cfg = make_test_config()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return HybridFrontend(cfg)

    def test_representative_accumulates_and_is_an_observation(self):
        front = self._front(use_representative_descriptor=True,
                            representative_sample_stride=1)
        img = make_textured_image()
        front.initialize(img)
        prev = img
        for _ in range(6):
            curr = shift_image(prev, 1, 0)
            front.process_frame(curr)
            prev = curr
        tracks = [t for t in front.active_tracks.values() if t.age >= 3]
        assert tracks, "expected surviving tracks"
        t = tracks[0]
        assert len(t.descriptor_history) > 1, "observations should accumulate"
        assert t.representative_descriptor is not None
        assert any(np.array_equal(t.representative_descriptor, o)
                   for o in t.descriptor_history)

    def test_history_capped(self):
        front = self._front(use_representative_descriptor=True,
                            representative_sample_stride=1,
                            representative_max_observations=4)
        img = make_textured_image()
        front.initialize(img)
        prev = img
        for _ in range(12):
            curr = shift_image(prev, 1, 0)
            front.process_frame(curr)
            prev = curr
        for t in front.active_tracks.values():
            assert len(t.descriptor_history) <= 4

    def test_disabled_leaves_history_empty(self):
        front = self._front(use_representative_descriptor=False)
        img = make_textured_image()
        front.initialize(img)
        front.process_frame(shift_image(img, 1, 0))
        for t in front.active_tracks.values():
            assert t.descriptor_history == []
            assert t.representative_descriptor is None

    def test_local_detect_does_not_inflate_active_set(self):
        """Local-only corners are re-ID candidates; unmatched ones must be
        discarded, so the active set never exceeds N_target."""
        front = self._front(local_detect_in_dormant_windows=True,
                            local_detect_quality_scale=0.01)  # very relaxed
        img = make_textured_image()
        front.initialize(img)
        front.process_frame(shift_image(img, 1, 0))
        ids = list(front.active_tracks.keys())[:30]
        front.force_kill(ids)
        prev = shift_image(img, 1, 0)
        for _ in range(3):
            curr = shift_image(prev, 1, 0)
            front.process_frame(curr)
            prev = curr
            assert len(front.active_tracks) <= front.cfg.target_active_tracks, (
                "local dormant-window corners must not be spawned as tracks"
            )

    def test_local_detect_runs_with_no_global_deficit(self):
        """Step 5 must still be reachable when Step 4 produces nothing."""
        front = self._front(local_detect_in_dormant_windows=True)
        img = make_textured_image()
        front.initialize(img)
        res = front.process_frame(shift_image(img, 1, 0))
        assert res.frame_index == 1
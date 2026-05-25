"""
dormant_buffer.py

Python port of the C++ DormantTrackBuffer (see ../cpp/include/DormantTrackBuffer.h).

Holds recently-died "infant" tracks (per design doc §4.1) so that a freshly
spawned Shi-Tomasi corner in a later frame can resurrect them with the same
track ID (Step 5 / §4.6).

Semantics mirror the C++ implementation exactly:
  - L-infinity (square window) spatial gate
  - boundary-inclusive horizon: an entry expires when
    frame_died < current_frame - horizon (strict inequality)
  - infant-only invariant: map_point must be None
  - no duplicate ids allowed in the buffer
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class DormantTrack:
    """A track that died (KLT lost it) but might still be resurrectable.

    `descriptor` is the 32-byte BRIEF descriptor as np.uint8 of shape (32,).
    """
    id: int
    last_x: float
    last_y: float
    descriptor: np.ndarray  # uint8 shape (32,)
    frame_died: int
    octave: int = 0
    map_point: object = None  # MUST be None for dormant tracks (§4.1)


class DormantTrackBuffer:
    """Short-term buffer of recently-died infant tracks.

    Operations are O(n) in buffer size; n is bounded by horizon * spawn_rate
    and is small in practice (tens to low hundreds).
    """

    def __init__(self, dormant_horizon_frames: int) -> None:
        if dormant_horizon_frames < 0:
            raise ValueError("dormant_horizon_frames must be non-negative")
        self._horizon = int(dormant_horizon_frames)
        self._entries: deque[DormantTrack] = deque()

    # ---- properties ------------------------------------------------

    @property
    def horizon(self) -> int:
        return self._horizon

    def __len__(self) -> int:
        return len(self._entries)

    def empty(self) -> bool:
        return not self._entries

    # ---- mutation --------------------------------------------------

    def add(self, track: DormantTrack) -> None:
        """Add an infant track that died this frame (or earlier).

        Enforces the §4.1 invariant (map_point is None) and the §7.4.2
        invariant (no duplicate ids in the buffer).
        """
        if track.map_point is not None:
            raise AssertionError(
                "DormantTrackBuffer holds infant tracks only; map_point must be None"
            )
        # Cheap duplicate check (linear). Buffer is small.
        for e in self._entries:
            if e.id == track.id:
                raise AssertionError(
                    f"Duplicate track id {track.id} in dormant buffer"
                )
        if track.descriptor.shape != (32,) or track.descriptor.dtype != np.uint8:
            raise ValueError(
                "descriptor must be np.uint8 of shape (32,)"
            )
        self._entries.append(track)

    def purge_older_than(self, current_frame: int) -> None:
        """Drop entries with frame_died < current_frame - horizon.

        Underflow-safe: if current_frame <= horizon, nothing is dropped.
        """
        if current_frame <= self._horizon:
            return
        cutoff = current_frame - self._horizon
        # Rebuild deque keeping only fresh entries. Faster than repeated popleft
        # when entries aren't in strict death order.
        self._entries = deque(e for e in self._entries if e.frame_died >= cutoff)

    def remove(self, track_id: int) -> bool:
        """Remove the entry with the given id. Returns True if found."""
        for i, e in enumerate(self._entries):
            if e.id == track_id:
                # deque doesn't support index removal directly; rotate-pop-rotate.
                # For small n this is fine.
                del self._entries[i]
                return True
        return False

    def clear(self) -> None:
        """Drop all entries (used on relocalization, §6.4)."""
        self._entries.clear()

    # ---- query -----------------------------------------------------

    def query_within(self, x: float, y: float,
                     radius: float) -> list[DormantTrack]:
        """Return all entries within L-infinity `radius` of (x, y).

        Returns shallow copies of the matching entries' references; callers
        should not mutate the descriptor arrays in place.
        """
        r = max(0.0, float(radius))
        hits = []
        for e in self._entries:
            if abs(e.last_x - x) <= r and abs(e.last_y - y) <= r:
                hits.append(e)
        return hits

    def all_entries(self) -> Iterable[DormantTrack]:
        """For inspection / testing only."""
        return tuple(self._entries)

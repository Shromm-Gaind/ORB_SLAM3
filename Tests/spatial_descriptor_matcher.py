"""
spatial_descriptor_matcher.py

Python port of the C++ SpatialDescriptorMatch (see
../cpp/include/SpatialDescriptorMatcher.h).

Shared primitive for Step 5 (dormant-track re-ID, §4.6) and Step 5b
(TrackLocalMap, §4.7). Both reduce to: given queries (prediction pixel +
descriptor) and candidates (pixel + descriptor), return the best
candidate per query under (a) L-infinity radius gate and (b) Hamming
distance gate.

No state. Mirrors the C++ semantics exactly, including the
unique_candidates conflict-resolution rule (lower Hamming wins; ties
break to lower query index).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---- Hamming distance ---------------------------------------------

# Precomputed popcount for all 256 uint8 values. Vectorized via numpy
# indexing — fast enough for our scale.
_POPCOUNT_TABLE = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint16
)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two BRIEF descriptors.

    Inputs are uint8 arrays of shape (32,). Returns an int in [0, 256].
    """
    if a.shape != (32,) or b.shape != (32,):
        raise ValueError("descriptors must be uint8 arrays of shape (32,)")
    xor = np.bitwise_xor(a, b)
    return int(_POPCOUNT_TABLE[xor].sum())


def hamming_distance_batch(query: np.ndarray,
                           candidates: np.ndarray) -> np.ndarray:
    """Compute Hamming distance from one query to many candidates.

    query: uint8 shape (32,)
    candidates: uint8 shape (C, 32)
    Returns: int32 shape (C,)
    """
    xor = np.bitwise_xor(candidates, query[None, :])  # (C, 32)
    return _POPCOUNT_TABLE[xor].sum(axis=1).astype(np.int32)


# ---- Matcher -------------------------------------------------------

@dataclass
class PixelQuery:
    x: float
    y: float
    descriptor: np.ndarray            # uint8 (32,)
    radius: float = -1.0              # override default_radius if > 0


@dataclass
class PixelCandidate:
    x: float
    y: float
    descriptor: np.ndarray            # uint8 (32,)


@dataclass
class MatchOptions:
    default_radius: float = 20.0      # px — r_reid (§4.6) or r_TLM (§4.7)
    hamming_threshold: int = 50       # inclusive (§8)
    unique_candidates: bool = False   # see C++ header doc
    # Ambiguity (distinctiveness) gate, ORB-SLAM3-style. If > 0, a query
    # is only matched when its best candidate beats the second-best
    # spatially-gated candidate by at least this many Hamming bits:
    #     second_best - best >= second_best_margin
    # A query with a single spatial candidate passes trivially (nothing
    # to be confused with). 0 disables the gate (legacy behaviour).
    # This is the standard defence against descriptor aliasing on
    # self-similar texture: when two nearby candidates look equally
    # good, refusing to match (fresh ID; §4.6 says a missed re-ID is
    # recoverable) beats guessing.
    # NOTE (C++ parity): SpatialDescriptorMatcher.h does not have this
    # option yet; add it there before porting the frontend.
    second_best_margin: int = 0


@dataclass
class Match:
    candidate_index: int
    hamming_distance: int


def spatial_descriptor_match(
    queries: Sequence[PixelQuery],
    candidates: Sequence[PixelCandidate],
    opts: MatchOptions,
) -> list[Match | None]:
    """Return per-query best candidate match (or None) under the gates."""
    Q = len(queries)
    C = len(candidates)
    out: list[Match | None] = [None] * Q
    if Q == 0 or C == 0:
        return out

    # Pre-stack candidate positions and descriptors for vectorized inner loop.
    cand_xy = np.array([[c.x, c.y] for c in candidates], dtype=np.float32)
    cand_desc = np.stack([c.descriptor for c in candidates], axis=0)  # (C, 32)

    for i, q in enumerate(queries):
        radius = q.radius if q.radius > 0.0 else opts.default_radius
        # Spatial gate first (cheap) — L-infinity.
        dx = np.abs(cand_xy[:, 0] - q.x)
        dy = np.abs(cand_xy[:, 1] - q.y)
        spatial_mask = (dx <= radius) & (dy <= radius)
        if not spatial_mask.any():
            continue
        # Hamming on the spatially-gated subset only.
        idxs = np.flatnonzero(spatial_mask)
        dists = hamming_distance_batch(q.descriptor, cand_desc[idxs])
        best_local = int(np.argmin(dists))
        best_dist = int(dists[best_local])
        # Gate 1: Hamming threshold (inclusive).
        if best_dist > opts.hamming_threshold:
            continue
        # Gate 2: distinctiveness — the best must beat the second-best
        # spatially-gated candidate by the configured margin. The
        # second-best is taken over ALL spatial candidates (not only
        # those under the threshold): a competitor just above the
        # threshold is still evidence of ambiguity.
        if opts.second_best_margin > 0 and len(dists) > 1:
            second_best = int(np.partition(dists, 1)[1])
            if second_best - best_dist < opts.second_best_margin:
                continue
        out[i] = Match(
            candidate_index=int(idxs[best_local]),
            hamming_distance=best_dist,
        )

    # Optional second pass: enforce unique candidates.
    if opts.unique_candidates:
        # candidate -> (query_idx, hamming) of current winner
        winner: dict[int, tuple[int, int]] = {}
        for i in range(Q):
            m = out[i]
            if m is None:
                continue
            j = m.candidate_index
            d = m.hamming_distance
            prev = winner.get(j)
            if prev is None or d < prev[1]:
                if prev is not None:
                    out[prev[0]] = None  # evict
                winner[j] = (i, d)
            else:
                out[i] = None  # loser

    return out
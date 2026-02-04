"""Track representation: centerline, boundaries, and progress computation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Track:
    """2D racing track defined by centerline waypoints and width."""

    waypoints: np.ndarray  # (N, 2) — (x, y) centerline points
    width: float
    loop: bool = True  # True for closed loop (oval/circuit)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64)
        self._precompute()

    def _precompute(self) -> None:
        """Precompute segment vectors and cumulative arc lengths."""
        n = len(self.waypoints)
        # Segment vectors
        self._segments = np.diff(self.waypoints, axis=0, append=self.waypoints[0:1] if self.loop else None)
        if not self.loop:
            self._segments = np.vstack([self._segments, [0, 0]])  # last segment zero

        # Segment lengths
        seg_lens = np.linalg.norm(self._segments, axis=1)
        seg_lens[np.isclose(seg_lens, 0)] = 1e-8
        self._seg_lens = seg_lens

        # Cumulative arc length from start (total track length)
        self._cumulative = np.concatenate([[0], np.cumsum(seg_lens)])
        self.length = float(np.sum(seg_lens))

    def nearest_segment(self, point: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Find nearest point on centerline and return segment index, nearest point, and arc length.

        Returns:
            segment_idx: index of segment containing nearest point
            nearest_pt: (2,) closest point on centerline
            arc_length: distance along track from start to nearest point (0 to length)
        """
        point = np.asarray(point, dtype=np.float64).reshape(2)
        best_dist = np.inf
        best_idx = 0
        best_pt = self.waypoints[0].copy()
        best_t = 0.0

        n = len(self.waypoints)
        for i in range(n - 1 if self.loop else n):
            if i >= len(self._segments) or np.linalg.norm(self._segments[i]) < 1e-10:
                continue
            a = self.waypoints[i]
            b = self.waypoints[(i + 1) % n] if self.loop else self.waypoints[i + 1]
            # Project point onto segment [a, b]
            ab = b - a
            ab_len = np.linalg.norm(ab)
            t = np.clip(np.dot(point - a, ab) / (ab_len**2 + 1e-10), 0, 1)
            pt = a + t * ab
            d = np.linalg.norm(point - pt)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_pt = pt
                best_t = t

        arc_length = self._cumulative[best_idx] + best_t * self._seg_lens[best_idx]
        if self.loop and arc_length >= self.length:
            arc_length -= self.length
        return best_idx, best_pt, arc_length

    def progress(self, point: np.ndarray) -> float:
        """Progress along lap as fraction 0..1 (0 = start, 1 = full lap)."""
        _, _, arc = self.nearest_segment(point)
        return (arc / self.length) if self.length > 0 else 0.0

    def lateral_offset(self, point: np.ndarray) -> float:
        """Signed distance to centerline: positive = left, negative = right."""
        point = np.asarray(point, dtype=np.float64).reshape(2)
        idx, nearest_pt, _ = self.nearest_segment(point)
        # Tangent direction at nearest point
        n = len(self.waypoints)
        next_idx = (idx + 1) % n if self.loop else min(idx + 1, n - 1)
        tangent = self.waypoints[next_idx] - self.waypoints[idx]
        tangent_len = np.linalg.norm(tangent)
        if tangent_len < 1e-10:
            return 0.0
        tangent = tangent / tangent_len
        # Normal pointing "left" (counterclockwise): (-ty, tx)
        normal = np.array([-tangent[1], tangent[0]])
        offset = np.dot(point - nearest_pt, normal)
        return float(offset)

    def heading_at_progress(self, progress: float) -> float:
        """Track direction (angle in radians) at given progress."""
        progress = progress % 1.0
        arc = progress * self.length
        # Find segment
        idx = np.searchsorted(self._cumulative, arc, side="right") - 1
        idx = max(0, min(idx, len(self._segments) - 1))
        seg = self._segments[idx]
        return float(np.arctan2(seg[1], seg[0]))

    def is_on_track(self, point: np.ndarray) -> bool:
        """True if point is within track boundaries (centerline ± width/2)."""
        return abs(self.lateral_offset(point)) <= self.width / 2

    def sample_start_pose(self, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        """Sample random starting position and heading on track."""
        # Pick random progress
        p = rng.random()
        arc = p * self.length
        idx = np.searchsorted(self._cumulative, arc, side="right") - 1
        idx = max(0, min(idx, len(self.waypoints) - 1))
        t = (arc - self._cumulative[idx]) / (self._seg_lens[idx] + 1e-10)
        t = np.clip(t, 0, 1)
        pos = self.waypoints[idx] + t * self._segments[idx]
        heading = self.heading_at_progress(p)
        return pos, heading


def make_oval_track(
    length: float = 20.0,
    width: float = 4.0,
    straight_ratio: float = 0.5,
    num_points: int = 64,
) -> Track:
    """Create a simple oval track (two straights + two semicircles)."""
    straight_len = length * straight_ratio / 2
    curve_radius = (length * (1 - straight_ratio)) / (2 * np.pi)
    half_w = straight_len / 2

    points: List[Tuple[float, float]] = []
    # Start at left end of bottom straight
    # Bottom straight (y = -curve_radius)
    n_straight = max(2, num_points // 4)
    for i in range(n_straight + 1):
        x = -half_w + (2 * half_w * i / n_straight)
        points.append((x, -curve_radius))

    # Right semicircle
    n_curve = max(2, num_points // 4)
    for i in range(1, n_curve + 1):
        angle = -np.pi / 2 + (np.pi * i / n_curve)
        points.append((half_w + curve_radius * np.cos(angle), curve_radius * np.sin(angle)))

    # Top straight
    for i in range(1, n_straight + 1):
        x = half_w - (2 * half_w * i / n_straight)
        points.append((x, curve_radius))

    # Left semicircle
    for i in range(1, n_curve + 1):
        angle = np.pi / 2 + (np.pi * i / n_curve)
        points.append((-half_w + curve_radius * np.cos(angle), curve_radius * np.sin(angle)))

    return Track(waypoints=np.array(points), width=width, loop=True)

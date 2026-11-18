"""Geodesy helpers.

Trips are city-scale (< ~50 km) but spread over the whole globe, so a single
global projection is a poor fit. Each trip gets its own local equirectangular
(tangent-plane) projection centred on its GPS points: x = R cos(lat0) dlon,
y = R dlat. Over a few tens of km the distance error is well under 1 %, which
is far below GPS noise, and it lets us use fast Euclidean geometry in metres.
"""
import numpy as np

from .config import EARTH_R


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    @classmethod
    def for_track(cls, track: np.ndarray) -> "LocalProjection":
        return cls(np.median(track[:, 0]), np.median(track[:, 1]))

    def fwd(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx, (lonlat[:, 1] - self.lat0) * self.ky])

    def inv(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised distance from points p to segments a-b (all (K,2)).

    Returns (distance, fraction along segment in [0,1])."""
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(L2 > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.hypot(*(p - proj).T), t


def angle_diff(a, b):
    """Absolute difference between two compass bearings (radians) in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

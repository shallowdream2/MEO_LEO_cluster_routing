from typing import Dict, Tuple, List
import math

from .satellites import LEOSatellite, MEOSatellite


def distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def find_nearest_available_leo(
    ground_pos: Tuple[float, float, float],
    leos: Dict[int, LEOSatellite],
    load_threshold: int,
) -> LEOSatellite:
    """Return the nearest LEO with load below threshold."""
    candidates = [leo for leo in leos.values() if leo.load < load_threshold]
    if not candidates:
        raise ValueError("No available LEO satellite")
    return min(candidates, key=lambda l: distance(ground_pos, (l.latitude, l.longitude, l.altitude)))


def get_leo_by_id(leos: Dict[int, LEOSatellite], leo_id: int) -> LEOSatellite:
    if leo_id not in leos:
        raise ValueError(f"LEO {leo_id} not found")
    return leos[leo_id]


def get_meo_by_id(meos: Dict[int, MEOSatellite], meo_id: int) -> MEOSatellite:
    if meo_id not in meos:
        raise ValueError(f"MEO {meo_id} not found")
    return meos[meo_id]

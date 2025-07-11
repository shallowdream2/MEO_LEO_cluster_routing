from dataclasses import dataclass, field
from typing import List

@dataclass
class Satellite:
    """Base class for a satellite."""
    id: int
    latitude: float
    longitude: float
    altitude: float

@dataclass
class LEOSatellite(Satellite):
    """Low Earth Orbit satellite."""
    load: int = 0
    neighbors: List[int] = field(default_factory=list)
    meo_id: int = 0

@dataclass
class MEOSatellite(Satellite):
    """Medium Earth Orbit satellite."""
    cluster_leos: List[int] = field(default_factory=list)

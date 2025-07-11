"""Example simulation of MEO controlled LEO routing."""
import random
from typing import Dict

from .satellites import LEOSatellite, MEOSatellite
from .environment import find_nearest_available_leo
from .rl_agent import RLAgent
from .routing import route_request


def build_example_network() -> tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite]]:
    """Create a very small network for demonstration purposes."""
    # Create MEO satellites
    meos = {
        1: MEOSatellite(id=1, latitude=0, longitude=0, altitude=20000, cluster_leos=[1, 2]),
        2: MEOSatellite(id=2, latitude=10, longitude=10, altitude=20000, cluster_leos=[3, 4]),
    }

    # Create LEO satellites
    leos = {
        1: LEOSatellite(id=1, latitude=0, longitude=0, altitude=500, neighbors=[2], meo_id=1),
        2: LEOSatellite(id=2, latitude=1, longitude=1, altitude=500, neighbors=[1], meo_id=1),
        3: LEOSatellite(id=3, latitude=10, longitude=10, altitude=500, neighbors=[4], meo_id=2),
        4: LEOSatellite(id=4, latitude=11, longitude=11, altitude=500, neighbors=[3], meo_id=2),
    }
    return leos, meos


def main():
    leos, meos = build_example_network()
    agent = RLAgent()

    ground_pos = (0.5, 0.5, 0)
    load_threshold = 5
    src_leo = find_nearest_available_leo(ground_pos, leos, load_threshold)

    # Example: send request to LEO 4
    path = route_request(src_leo.id, 4, leos, meos, agent)
    print("Routing path:", path)


if __name__ == "__main__":
    random.seed(0)
    main()

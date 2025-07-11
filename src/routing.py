from typing import Dict, List

from .satellites import LEOSatellite, MEOSatellite
from .environment import get_leo_by_id, get_meo_by_id
from .rl_agent import RLAgent


def route_request(
    src_leo_id: int,
    dst_leo_id: int,
    leos: Dict[int, LEOSatellite],
    meos: Dict[int, MEOSatellite],
    agent: RLAgent,
) -> List[int]:
    """Return list of LEO ids representing the path."""
    src_leo = get_leo_by_id(leos, src_leo_id)
    dst_leo = get_leo_by_id(leos, dst_leo_id)
    path = [src_leo.id]

    # If in the same MEO cluster, use RL to find path via neighbors
    if src_leo.meo_id == dst_leo.meo_id:
        current = src_leo.id
        while current != dst_leo.id:
            current_leo = get_leo_by_id(leos, current)
            actions = current_leo.neighbors
            state = (current, dst_leo.id)
            next_action = agent.choose_action(state, actions)
            path.append(next_action)
            current = next_action
    else:
        # Different clusters: src -> src edge -> cross cluster -> dest
        src_meo = get_meo_by_id(meos, src_leo.meo_id)
        dst_meo = get_meo_by_id(meos, dst_leo.meo_id)
        # choose edge nodes (simplified as first LEO in cluster list)
        edge_src = src_meo.cluster_leos[0]
        edge_dst = dst_meo.cluster_leos[0]
        # path within src cluster
        current = src_leo.id
        while current != edge_src:
            current_leo = get_leo_by_id(leos, current)
            actions = current_leo.neighbors
            state = (current, edge_src)
            next_action = agent.choose_action(state, actions)
            path.append(next_action)
            current = next_action
        # cross cluster via MEO satellites (abstracted as direct link)
        path.append(edge_dst)
        current = edge_dst
        while current != dst_leo.id:
            current_leo = get_leo_by_id(leos, current)
            actions = current_leo.neighbors
            state = (current, dst_leo.id)
            next_action = agent.choose_action(state, actions)
            path.append(next_action)
            current = next_action

    return path

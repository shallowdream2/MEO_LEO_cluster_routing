from typing import Dict, List, Tuple
import math

from .satellites import LEOSatellite, MEOSatellite
from .environment import get_leo_by_id, get_meo_by_id
from .rl_agent import RLAgent


def _check_cluster_connectivity(src_leo_id: int, dst_leo_id: int, cluster_leos: List[int], leos: Dict[int, LEOSatellite]) -> bool:
    """
    检查cluster内两个LEO之间是否有连通性
    """
    if src_leo_id == dst_leo_id:
        return True
    
    from collections import deque
    
    # 使用BFS检查连通性，只在cluster内搜索
    visited = {src_leo_id}
    queue = deque([src_leo_id])
    
    while queue:
        current = queue.popleft()
        
        if current == dst_leo_id:
            return True
            
        current_leo = get_leo_by_id(leos, current)
        
        for neighbor in current_leo.neighbors:
            if neighbor in cluster_leos and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False


def calculate_geographic_distance(leo1: LEOSatellite, leo2: LEOSatellite) -> float:
    """计算两个LEO卫星之间的地理距离。"""
    return math.sqrt(
        (leo1.latitude - leo2.latitude) ** 2 +
        (leo1.longitude - leo2.longitude) ** 2 +
        (leo1.altitude - leo2.altitude) ** 2
    )


def find_optimal_edge_nodes(
    src_cluster_leos: List[int],
    dst_cluster_leos: List[int],
    leos: Dict[int, LEOSatellite]
) -> Tuple[int, int]:
    """
    在两个cluster之间找到地理上距离最近的边缘节点对。
    
    Args:
        src_cluster_leos: 源cluster中的LEO卫星ID列表
        dst_cluster_leos: 目标cluster中的LEO卫星ID列表
        leos: 所有LEO卫星的字典
    
    Returns:
        (src_edge_leo_id, dst_edge_leo_id): 最优边缘节点对的ID
    """
    min_distance = float('inf')
    best_src_edge = src_cluster_leos[0]
    best_dst_edge = dst_cluster_leos[0]
    
    for src_leo_id in src_cluster_leos:
        for dst_leo_id in dst_cluster_leos:
            src_leo = get_leo_by_id(leos, src_leo_id)
            dst_leo = get_leo_by_id(leos, dst_leo_id)
            distance = calculate_geographic_distance(src_leo, dst_leo)
            
            if distance < min_distance:
                min_distance = distance
                best_src_edge = src_leo_id
                best_dst_edge = dst_leo_id
    
    return best_src_edge, best_dst_edge


def find_optimal_edge_nodes_advanced(
    src_cluster_leos: List[int],
    dst_cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    load_weight: float = 0.3,
    distance_weight: float = 0.7
) -> Tuple[int, int]:
    """
    在两个cluster之间找到综合考虑地理距离和负载的最优边缘节点对。
    
    Args:
        src_cluster_leos: 源cluster中的LEO卫星ID列表
        dst_cluster_leos: 目标cluster中的LEO卫星ID列表
        leos: 所有LEO卫星的字典
        load_weight: 负载因子的权重
        distance_weight: 距离因子的权重
    
    Returns:
        (src_edge_leo_id, dst_edge_leo_id): 最优边缘节点对的ID
    """
    min_cost = float('inf')
    best_src_edge = src_cluster_leos[0]
    best_dst_edge = dst_cluster_leos[0]
    
    # 计算归一化参数
    max_distance = 0
    max_load = 0
    for src_leo_id in src_cluster_leos:
        for dst_leo_id in dst_cluster_leos:
            src_leo = get_leo_by_id(leos, src_leo_id)
            dst_leo = get_leo_by_id(leos, dst_leo_id)
            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            max_distance = max(max_distance, distance)
            max_load = max(max_load, load)
    
    # 避免除零错误
    max_distance = max(max_distance, 1)
    max_load = max(max_load, 1)
    
    for src_leo_id in src_cluster_leos:
        for dst_leo_id in dst_cluster_leos:
            src_leo = get_leo_by_id(leos, src_leo_id)
            dst_leo = get_leo_by_id(leos, dst_leo_id)
            
            # 确保边缘节点有足够的邻居连接（连通性检查）
            if len(src_leo.neighbors) < 2 or len(dst_leo.neighbors) < 2:
                continue
                
            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            
            # 归一化并计算综合成本
            normalized_distance = distance / max_distance
            normalized_load = load / max_load
            cost = distance_weight * normalized_distance + load_weight * normalized_load
            
            if cost < min_cost:
                min_cost = cost
                best_src_edge = src_leo_id
                best_dst_edge = dst_leo_id
    
    return best_src_edge, best_dst_edge


def find_optimal_edge_nodes_with_redundancy(
    src_cluster_leos: List[int],
    dst_cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    num_candidates: int = 3,
    load_weight: float = 0.25,
    distance_weight: float = 0.35,
    connectivity_weight: float = 0.25,
    reliability_weight: float = 0.15
) -> List[Tuple[int, int]]:
    """
    找到多个最优边缘节点对，支持负载均衡和冗余路径。
    
    Args:
        src_cluster_leos: 源cluster中的LEO卫星ID列表
        dst_cluster_leos: 目标cluster中的LEO卫星ID列表
        leos: 所有LEO卫星的字典
        num_candidates: 返回的候选边缘节点对数量
        load_weight: 负载因子权重
        distance_weight: 距离因子权重
        connectivity_weight: 连通性因子权重
        reliability_weight: 可靠性因子权重
    
    Returns:
        按优先级排序的边缘节点对列表
    """
    candidates = []
    
    # 计算归一化参数
    max_distance = 0
    max_load = 0
    max_connectivity = 0
    
    for src_leo_id in src_cluster_leos:
        for dst_leo_id in dst_cluster_leos:
            src_leo = get_leo_by_id(leos, src_leo_id)
            dst_leo = get_leo_by_id(leos, dst_leo_id)
            
            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            connectivity = len(src_leo.neighbors) + len(dst_leo.neighbors)
            
            max_distance = max(max_distance, distance)
            max_load = max(max_load, load)
            max_connectivity = max(max_connectivity, connectivity)
    
    # 避免除零错误
    max_distance = max(max_distance, 1)
    max_load = max(max_load, 1)
    max_connectivity = max(max_connectivity, 1)
    
    for src_leo_id in src_cluster_leos:
        for dst_leo_id in dst_cluster_leos:
            src_leo = get_leo_by_id(leos, src_leo_id)
            dst_leo = get_leo_by_id(leos, dst_leo_id)
            
            # 基本连通性检查
            if len(src_leo.neighbors) < 2 or len(dst_leo.neighbors) < 2:
                continue
            
            # 计算各项指标
            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            connectivity = len(src_leo.neighbors) + len(dst_leo.neighbors)
            
            # 可靠性评估（基于邻居的负载分布）
            src_neighbor_loads = [get_leo_by_id(leos, n).load for n in src_leo.neighbors]
            dst_neighbor_loads = [get_leo_by_id(leos, n).load for n in dst_leo.neighbors]
            reliability = 1.0 / (1.0 + max(src_neighbor_loads + dst_neighbor_loads, default=0))
            
            # 归一化指标
            normalized_distance = distance / max_distance
            normalized_load = load / max_load
            normalized_connectivity = connectivity / max_connectivity
            normalized_reliability = reliability  # 已经在[0,1]范围内
            
            # 计算综合得分（越小越好）
            score = (distance_weight * normalized_distance +
                    load_weight * normalized_load +
                    connectivity_weight * (1 - normalized_connectivity) +  # 连通性越高越好
                    reliability_weight * (1 - normalized_reliability))     # 可靠性越高越好
            
            candidates.append((score, src_leo_id, dst_leo_id))
    
    # 按得分排序并返回前N个
    candidates.sort(key=lambda x: x[0])
    return [(src_id, dst_id) for _, src_id, dst_id in candidates[:num_candidates]]

def calculate_path_score(path: List[int], leos: Dict[int, LEOSatellite], load_weight: float = 0.4, delay_weight: float = 0.6) -> float:
    """
    计算路径的综合得分（负载和延迟）
    """
    if len(path) < 2:
        return 0.0
    
    total_load = sum(get_leo_by_id(leos, leo_id).load for leo_id in path)
    path_delay = len(path) - 1  # 简单用跳数表示延迟
    
    # 归一化处理
    normalized_load = total_load / len(path) if len(path) > 0 else 0
    normalized_delay = path_delay
    
    # 计算综合得分（越小越好）
    score = load_weight * normalized_load + delay_weight * normalized_delay
    return score


def agent_generate_single_path(
    start_leo_id: int,
    end_leo_id: int,
    cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    agent: RLAgent,
    max_hops: int = 15,
    avoid_nodes: set = None
) -> List[int]:
    """
    使用agent生成单条路径，如果失败则使用BFS作为回退
    """
    if avoid_nodes is None:
        avoid_nodes = set()
    
    if start_leo_id == end_leo_id:
        return [start_leo_id]
    
    # 首先尝试使用agent
    path = _agent_path_generation(start_leo_id, end_leo_id, cluster_leos, leos, agent, max_hops, avoid_nodes)
    
    # 如果agent失败，使用BFS作为回退
    if not path:
        path = _bfs_path_generation(start_leo_id, end_leo_id, cluster_leos, leos, max_hops, avoid_nodes)
    
    return path

def _agent_path_generation(
    start_leo_id: int,
    end_leo_id: int,
    cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    agent: RLAgent,
    max_hops: int,
    avoid_nodes: set
) -> List[int]:
    """Agent路径生成"""
    path = [start_leo_id]
    current = start_leo_id
    visited = {start_leo_id}
    
    for hop in range(max_hops):
        if current == end_leo_id:
            break
            
        current_leo = get_leo_by_id(leos, current)
        
        # 获取可用的邻居节点
        available_neighbors = [
            n for n in current_leo.neighbors 
            if n in cluster_leos and n not in visited and n not in avoid_nodes
        ]
        
        if not available_neighbors:
            # 没有可用邻居，路径失败
            return []
        
        # 使用agent选择下一个节点
        state = (current, end_leo_id)
        next_node = agent.choose_action(state, available_neighbors)
        
        path.append(next_node)
        visited.add(next_node)
        current = next_node
        
        # 计算奖励（基于是否接近目标）
        if current == end_leo_id:
            reward = 10.0  # 到达目标的奖励
        else:
            # 基于距离目标的接近程度给予奖励
            current_leo = get_leo_by_id(leos, current)
            target_leo = get_leo_by_id(leos, end_leo_id)
            distance = calculate_geographic_distance(current_leo, target_leo)
            reward = -distance * 0.1  # 距离越近奖励越高
        
        # 更新agent
        next_state = (current, end_leo_id)
        next_neighbors = [
            n for n in get_leo_by_id(leos, current).neighbors 
            if n in cluster_leos and n not in visited
        ]
        agent.update(state, next_node, reward, next_state, next_neighbors)
    
    return path if current == end_leo_id else []

def _bfs_path_generation(
    start_leo_id: int,
    end_leo_id: int,
    cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    max_hops: int,
    avoid_nodes: set
) -> List[int]:
    """BFS路径生成作为回退策略"""
    from collections import deque
    
    if start_leo_id == end_leo_id:
        return [start_leo_id]
    
    queue = deque([(start_leo_id, [start_leo_id])])
    visited = {start_leo_id}
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > max_hops:
            continue
            
        if current == end_leo_id:
            return path
            
        current_leo = get_leo_by_id(leos, current)
        
        for neighbor in current_leo.neighbors:
            if (neighbor in cluster_leos and 
                neighbor not in visited and 
                neighbor not in avoid_nodes):
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    return []  # 没有找到路径


def agent_generate_k_paths(
    start_leo_id: int,
    end_leo_id: int,
    cluster_leos: List[int],
    leos: Dict[int, LEOSatellite],
    agent: RLAgent,
    k: int = 3,
    max_hops: int = 15
) -> List[List[int]]:
    """
    使用agent生成k条不同的路径
    """
    paths = []
    
    for attempt in range(k * 3):  # 多尝试几次以获得k条不同路径
        if len(paths) >= k:
            break
            
        # 只避免上一条路径中的中间节点，不避免所有节点
        avoid_nodes = set()
        if paths:
            # 只避免最近一条路径的中间节点
            last_path = paths[-1]
            if len(last_path) > 2:
                avoid_nodes = set(last_path[1:-1])  # 只避免中间节点
            
        path = agent_generate_single_path(
            start_leo_id, end_leo_id, cluster_leos, leos, agent, max_hops, 
            avoid_nodes=avoid_nodes
        )
        
        if path and path not in paths and len(path) <= max_hops + 1:
            paths.append(path)
    
    return paths


def route_request_with_intelligent_edge_selection(
    src_leo_id: int,
    dst_leo_id: int,
    leos: Dict[int, LEOSatellite],
    meos: Dict[int, MEOSatellite],
    agent: RLAgent,
    max_hops: int = 25,
    max_retries: int = 3,
    load_weight: float = 0.25,
    distance_weight: float = 0.35,
) -> Tuple[List[int], Dict[str, any]]:
    """
    原有的智能边缘节点选择路由函数（现在调用新的agent路由函数）
    """
    # 使用新的基于agent的路由函数
    return agent_based_routing_with_k_paths(
        src_leo_id, dst_leo_id, leos, meos, agent, 
        k_paths=max_retries, max_hops=max_hops, load_weight=load_weight, delay_weight=distance_weight
    )

def agent_based_routing_with_k_paths(
    src_leo_id: int,
    dst_leo_id: int,
    leos: Dict[int, LEOSatellite],
    meos: Dict[int, MEOSatellite],
    agent: RLAgent,
    k_paths: int = 3,
    max_hops: int = 25,
    load_weight: float = 0.4,
    delay_weight: float = 0.6
) -> Tuple[List[int], Dict[str, any]]:
    """
    基于agent的k路径路由函数，实现跨cluster两段式处理
    
    Args:
        src_leo_id: 源LEO卫星ID
        dst_leo_id: 目标LEO卫星ID
        leos: 所有LEO卫星的字典
        meos: 所有MEO卫星的字典
        agent: 强化学习智能体
        k_paths: 生成的候选路径数量
        max_hops: 最大跳数限制
        load_weight: 负载权重
        delay_weight: 延迟权重
    
    Returns:
        (最优路径, 路由统计信息)
    """
    src_leo = get_leo_by_id(leos, src_leo_id)
    dst_leo = get_leo_by_id(leos, dst_leo_id)
    
    # 路由统计信息
    routing_stats = {
        'total_hops': 0,
        'segment1_hops': 0,
        'inter_cluster_hops': 0,
        'segment2_hops': 0,
        'edge_nodes_used': [],
        'k_paths_generated': 0,
        'routing_strategy': 'unknown',
        'success': False,
        'path_scores': []
    }
    
    # 同cluster内路由
    if src_leo.meo_id == dst_leo.meo_id:
        routing_stats['routing_strategy'] = 'intra_cluster'
        src_meo = get_meo_by_id(meos, src_leo.meo_id)
        
        # 检查cluster内是否有直接连通性
        cluster_connected = _check_cluster_connectivity(src_leo_id, dst_leo_id, src_meo.cluster_leos, leos)
        
        if cluster_connected:
            # 使用agent生成k条路径
            k_paths_list = agent_generate_k_paths(
                src_leo_id, dst_leo_id, src_meo.cluster_leos, leos, agent, k_paths
            )
            
            if k_paths_list:
                # 选择最优路径
                best_path = min(k_paths_list, key=lambda p: calculate_path_score(p, leos, load_weight, delay_weight))
                
                routing_stats['total_hops'] = len(best_path) - 1
                routing_stats['segment1_hops'] = routing_stats['total_hops']
                routing_stats['k_paths_generated'] = len(k_paths_list)
                routing_stats['success'] = True
                routing_stats['path_scores'] = [calculate_path_score(p, leos, load_weight, delay_weight) for p in k_paths_list]
                
                return best_path, routing_stats
        
        # 如果cluster内没有直接连通性，使用全网络路由
        routing_stats['routing_strategy'] = 'intra_cluster_via_network'
        all_leo_ids = list(leos.keys())
        k_paths_list = agent_generate_k_paths(
            src_leo_id, dst_leo_id, all_leo_ids, leos, agent, k_paths
        )
        
        if k_paths_list:
            # 选择最优路径
            best_path = min(k_paths_list, key=lambda p: calculate_path_score(p, leos, load_weight, delay_weight))
            
            routing_stats['total_hops'] = len(best_path) - 1
            routing_stats['segment1_hops'] = routing_stats['total_hops']
            routing_stats['k_paths_generated'] = len(k_paths_list)
            routing_stats['success'] = True
            routing_stats['path_scores'] = [calculate_path_score(p, leos, load_weight, delay_weight) for p in k_paths_list]
            
            return best_path, routing_stats
        else:
            routing_stats['success'] = False
            return [src_leo_id], routing_stats
    
    # 跨cluster路由 - 两段式处理
    routing_stats['routing_strategy'] = 'inter_cluster_two_stage'
    src_meo = get_meo_by_id(meos, src_leo.meo_id)
    dst_meo = get_meo_by_id(meos, dst_leo.meo_id)
    
    # 首先尝试两段式路由
    edge_candidates = find_optimal_edge_nodes_with_redundancy(
        src_meo.cluster_leos,
        dst_meo.cluster_leos,
        leos,
        num_candidates=k_paths,
        load_weight=load_weight * 0.6,  # 调整权重用于边缘节点选择
        distance_weight=delay_weight * 0.6,
        connectivity_weight=0.2,
        reliability_weight=0.2
    )
    
    if edge_candidates:
        # 第二步：为每个边缘节点对生成第一段路径
        segment1_candidates = []
        for edge_src, edge_dst in edge_candidates:
            routing_stats['edge_nodes_used'].append((edge_src, edge_dst))
            
            # 使用agent生成从源到边缘节点的k条路径
            if src_leo_id == edge_src:
                segment1_paths = [[src_leo_id]]
            else:
                # 如果cluster内连通性差，使用全网络路由
                cluster_connected = _check_cluster_connectivity(src_leo_id, edge_src, src_meo.cluster_leos, leos)
                if cluster_connected:
                    segment1_paths = agent_generate_k_paths(
                        src_leo_id, edge_src, src_meo.cluster_leos, leos, agent, k_paths
                    )
                else:
                    segment1_paths = agent_generate_k_paths(
                        src_leo_id, edge_src, list(leos.keys()), leos, agent, k_paths
                    )
            
            # 为每条第一段路径计算得分
            for path in segment1_paths:
                if path:
                    score = calculate_path_score(path, leos, load_weight, delay_weight)
                    segment1_candidates.append((score, path, edge_src, edge_dst))
        
        if segment1_candidates:
            # 选择最优的第一段路径
            segment1_candidates.sort(key=lambda x: x[0])
            best_score1, best_segment1, best_edge_src, best_edge_dst = segment1_candidates[0]
            
            # 第三步：使用另一个agent实例生成第二段的k条路径
            segment2_agent = RLAgent(learning_rate=agent.lr, gamma=agent.gamma, epsilon=agent.epsilon)
            
            if best_edge_dst == dst_leo_id:
                segment2_paths = [[best_edge_dst]]
            else:
                # 如果cluster内连通性差，使用全网络路由
                cluster_connected = _check_cluster_connectivity(best_edge_dst, dst_leo_id, dst_meo.cluster_leos, leos)
                if cluster_connected:
                    segment2_paths = agent_generate_k_paths(
                        best_edge_dst, dst_leo_id, dst_meo.cluster_leos, leos, segment2_agent, k_paths
                    )
                else:
                    segment2_paths = agent_generate_k_paths(
                        best_edge_dst, dst_leo_id, list(leos.keys()), leos, segment2_agent, k_paths
                    )
            
            if segment2_paths:
                # 选择最优的第二段路径
                best_segment2 = min(segment2_paths, key=lambda p: calculate_path_score(p, leos, load_weight, delay_weight))
                
                # 组合完整路径
                if best_edge_src == best_edge_dst:
                    # 边缘节点相同，直接连接两段
                    full_path = best_segment1 + best_segment2[1:]  # 避免重复边缘节点
                else:
                    # 添加跨cluster跳跃
                    full_path = best_segment1 + [best_edge_dst] + best_segment2[1:]
                
                # 更新统计信息
                routing_stats['total_hops'] = len(full_path) - 1
                routing_stats['segment1_hops'] = len(best_segment1) - 1
                routing_stats['inter_cluster_hops'] = 1 if best_edge_src != best_edge_dst else 0
                routing_stats['segment2_hops'] = len(best_segment2) - 1
                routing_stats['k_paths_generated'] = len(segment1_candidates) + len(segment2_paths)
                routing_stats['success'] = True
                routing_stats['path_scores'] = [best_score1, calculate_path_score(best_segment2, leos, load_weight, delay_weight)]
                
                return full_path, routing_stats
    
    # 如果两段式路由失败，使用全网络直接路由作为回退
    routing_stats['routing_strategy'] = 'inter_cluster_fallback'
    all_leo_ids = list(leos.keys())
    k_paths_list = agent_generate_k_paths(
        src_leo_id, dst_leo_id, all_leo_ids, leos, agent, k_paths
    )
    
    if k_paths_list:
        # 选择最优路径
        best_path = min(k_paths_list, key=lambda p: calculate_path_score(p, leos, load_weight, delay_weight))
        
        routing_stats['total_hops'] = len(best_path) - 1
        routing_stats['segment1_hops'] = routing_stats['total_hops']
        routing_stats['k_paths_generated'] = len(k_paths_list)
        routing_stats['success'] = True
        routing_stats['path_scores'] = [calculate_path_score(p, leos, load_weight, delay_weight) for p in k_paths_list]
        
        return best_path, routing_stats
    
    # 所有尝试都失败
    routing_stats['success'] = False
    return [src_leo_id], routing_stats



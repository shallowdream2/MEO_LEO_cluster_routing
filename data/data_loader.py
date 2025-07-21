"""
基于data.json的数据加载和环境初始化模块
包含MEO卫星信息和LEO-MEO分配关系
"""

import json
from typing import Dict, Tuple, List
from src.satellites import LEOSatellite, MEOSatellite

def load_environment_from_json(json_file: str = "data.json") -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    从JSON文件加载完整的卫星环境数据
    
    Args:
        json_file: JSON数据文件路径
        
    Returns:
        (leos, meos, raw_data): LEO卫星字典, MEO卫星字典, 原始数据
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建MEO卫星
    meos = {}
    meo_positions = data.get('meo_positions', [])
    for i, (lat, lon, alt) in enumerate(meo_positions):
        meos[i] = MEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            cluster_leos=[]  # 将在后面动态分配
        )
    
    # 创建LEO卫星字典
    leos = {}
    
    return leos, meos, data

def get_leo_meo_assignment(slot_id: int, data: dict) -> List[int]:
    """
    获取指定时间槽的LEO-MEO分配关系
    
    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据
        
    Returns:
        LEO卫星对应的MEO控制节点ID列表
    """
    meo_assignments = data.get('MEO_per_slot', [])
    for slot_info in meo_assignments:
        if slot_info['slot_id'] == slot_id:
            return slot_info['leo_meo_assignments']
    
    # 如果没找到，返回默认分配
    num_leos = data.get('num_satellites', 10)
    num_meos = data.get('num_meo_satellites', 3)
    return [i % num_meos for i in range(num_leos)]

def create_leos_for_slot(slot_id: int, data: dict) -> Dict[int, LEOSatellite]:
    """
    为指定时间槽创建LEO卫星
    
    Args:
        slot_id: 时间槽ID
        data: 原始JSON数据
        
    Returns:
        LEO卫星字典
    """
    leos = {}
    
    # 获取该时间槽的卫星位置
    sat_positions = data['sat_positions_per_slot'][slot_id]
    
    # 获取该时间槽的邻居关系
    neighbors_info = None
    for neighbor_slot in data['neighbors_per_slot']:
        if neighbor_slot['slot_id'] == slot_id:
            neighbors_info = neighbor_slot['neighbors']
            break
    
    # 获取MEO分配
    meo_assignments = get_leo_meo_assignment(slot_id, data)
    
    # 创建LEO卫星
    for i, (lat, lon) in enumerate(sat_positions):
        neighbors = neighbors_info[i] if neighbors_info else []
        meo_id = meo_assignments[i] if i < len(meo_assignments) else 0
        
        leos[i] = LEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=500.0,  # 默认LEO高度
            load=0,  # 初始负载为0
            neighbors=neighbors,
            meo_id=meo_id
        )
    
    return leos

def update_meo_clusters(leos: Dict[int, LEOSatellite], meos: Dict[int, MEOSatellite]):
    """
    根据LEO的MEO分配更新MEO的cluster_leos列表
    
    Args:
        leos: LEO卫星字典
        meos: MEO卫星字典
    """
    # 清空所有MEO的cluster列表
    for meo in meos.values():
        meo.cluster_leos = []
    
    # 根据LEO的meo_id重新分配
    for leo in leos.values():
        if leo.meo_id in meos:
            meos[leo.meo_id].cluster_leos.append(leo.id)

def load_complete_environment(slot_id: int, json_file: str = "data.json") -> Tuple[Dict[int, LEOSatellite], Dict[int, MEOSatellite], dict]:
    """
    加载指定时间槽的完整环境
    
    Args:
        slot_id: 时间槽ID
        json_file: JSON数据文件路径
        
    Returns:
        (leos, meos, raw_data): 完整的环境数据
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建MEO卫星
    meos = {}
    meo_positions = data.get('meo_positions', [])
    for i, (lat, lon, alt) in enumerate(meo_positions):
        meos[i] = MEOSatellite(
            id=i,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            cluster_leos=[]
        )
    
    # 创建LEO卫星
    leos = create_leos_for_slot(slot_id, data)
    
    # 更新MEO的cluster信息
    update_meo_clusters(leos, meos)
    
    return leos, meos, data

def print_environment_summary(leos: Dict[int, LEOSatellite], meos: Dict[int, MEOSatellite], slot_id: int):
    """打印环境摘要信息"""
    print(f"\n=== 时间槽 {slot_id} 环境摘要 ===")
    print(f"LEO卫星数量: {len(leos)}")
    print(f"MEO卫星数量: {len(meos)}")
    
    # 统计各MEO的cluster大小
    for meo_id, meo in meos.items():
        print(f"MEO {meo_id}: 控制 {len(meo.cluster_leos)} 个LEO卫星 {meo.cluster_leos}")
    
    # 显示前几个LEO的信息
    print("\n前5个LEO卫星信息:")
    for i in range(min(5, len(leos))):
        leo = leos[i]
        print(f"  LEO {i}: 位置({leo.latitude:.1f}, {leo.longitude:.1f}), 控制MEO={leo.meo_id}, 邻居={leo.neighbors}")

if __name__ == "__main__":
    # 测试数据加载
    for slot_id in [0, 25, 49]:
        leos, meos, data = load_complete_environment(slot_id)
        print_environment_summary(leos, meos, slot_id)

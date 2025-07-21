#!/usr/bin/env python3
"""
生成MEO_per_slot数据并更新data.json文件
"""

import json
import random

def generate_meo_assignments():
    """生成MEO分配数据"""
    num_slots = 100  # 50 train + 50 predict
    num_leos = 10
    num_meos = 3
    
    meo_per_slot = []
    
    # 设置随机种子以保证可重复性
    random.seed(42)
    
    for slot in range(num_slots):
        slot_data = {
            "slot_id": slot,
            "leo_meo_assignments": []
        }
        
        # 为每个LEO卫星分配一个MEO控制节点
        # 策略：基于地理相似性和负载平衡
        for leo_id in range(num_leos):
            # 基础分配：根据LEO ID和时间slot进行分配
            # 这模拟了基于地理位置的cluster分配
            if leo_id < 3:
                base_meo = 0  # LEO 0,1,2 主要由MEO 0控制
            elif leo_id < 7:
                base_meo = 1  # LEO 3,4,5,6 主要由MEO 1控制  
            else:
                base_meo = 2  # LEO 7,8,9 主要由MEO 2控制
            
            # 添加时间变化：每10个slot进行一次cluster重组
            cluster_shift = (slot // 10) % num_meos
            meo_id = (base_meo + cluster_shift) % num_meos
            
            # 添加10%的随机性来模拟负载平衡切换
            if random.random() < 0.1:
                meo_id = random.randint(0, num_meos - 1)
            
            slot_data["leo_meo_assignments"].append(meo_id)
        
        meo_per_slot.append(slot_data)
    
    return meo_per_slot

def update_json_file():
    """更新data.json文件"""
    meo_assignments = generate_meo_assignments()
    
    # 生成MEO assignments的JSON字符串
    meo_json_lines = []
    for i, slot_data in enumerate(meo_assignments):
        line = "    " + json.dumps(slot_data, separators=(',', ': '))
        if i < len(meo_assignments) - 1:
            line += ","
        meo_json_lines.append(line)
    
    # 将数据写入临时文件
    with open('meo_data.json', 'w') as f:
        f.write("[\n")
        f.write("\n".join(meo_json_lines))
        f.write("\n  ]\n}")
    
    print(f"Generated {len(meo_assignments)} slots of MEO assignment data")
    print("Sample slot 0:", meo_assignments[0])
    print("Sample slot 49:", meo_assignments[49])
    print("Sample slot 99:", meo_assignments[99])

if __name__ == "__main__":
    update_json_file()

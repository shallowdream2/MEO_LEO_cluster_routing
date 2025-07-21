# 智能边缘节点选择路由系统

## 概述

本项目实现了一个**支持环路检测和智能边缘节点选择的卫星路由系统**。该系统专门针对MEO-LEO集群路由场景进行了优化，集成了多种先进的路由策略和性能监控功能。

## 核心特性

### 🧠 智能边缘节点选择
- **多维度评估**：综合考虑距离、负载、连通性、可靠性
- **动态权重调整**：根据网络状况自适应优化
- **冗余路径支持**：提供多个候选边缘节点对

### 🔄 环路检测和避免
- **广度优先搜索**：确保最短路径同时避免环路
- **智能回溯机制**：处理死锁情况
- **访问节点追踪**：防止简单环路

### ⚖️ 集成负载均衡
- **动态负载监控**：实时调整路由权重
- **智能邻居选择**：优先选择低负载邻居
- **热点避免**：自动分散网络流量

### 📊 性能监控分析
- **详细统计信息**：跳数、重试次数、边缘节点使用情况
- **多维度评估**：跳数效率、负载效率、重试惩罚
- **智能建议**：基于分析结果生成优化建议

## 项目结构

```
src/
├── routing.py              # 核心路由算法
├── satellites.py           # 卫星数据结构
├── environment.py          # 环境管理
├── rl_agent.py            # 强化学习智能体
├── main.py                # 主程序入口
└── ...

examples/
└── advanced_routing_demo.py # 功能演示

docs/
├── ADVANCED_ROUTING.md     # 详细技术文档
└── ADVANCED_ROUTING_SUMMARY.md # 技术总结
```

## 核心函数

### 主要路由函数

```python
def route_request_with_intelligent_edge_selection(
    src_leo_id: int,
    dst_leo_id: int,
    leos: Dict[int, LEOSatellite],
    meos: Dict[int, MEOSatellite],
    agent: RLAgent,
    max_hops: int = 25,
    max_retries: int = 3,
    use_redundant_paths: bool = True,
    load_weight: float = 0.25,
    distance_weight: float = 0.35,
    connectivity_weight: float = 0.25,
    reliability_weight: float = 0.15,
    load_threshold: float = 0.8
) -> Tuple[List[int], Dict[str, any]]
```

**功能特点：**
- ✅ 环路检测和避免
- ✅ 智能边缘节点选择
- ✅ 动态负载均衡
- ✅ 多路径冗余
- ✅ 性能统计

## 使用示例

### 基本使用

```python
from src.routing import route_request_with_intelligent_edge_selection
from src.rl_agent import RLAgent

# 初始化智能体
agent = RLAgent(learning_rate=0.001, gamma=0.9, epsilon=0.1)

# 执行路由请求
path, stats = route_request_with_intelligent_edge_selection(
    src_leo_id=1,
    dst_leo_id=15,
    leos=leos,
    meos=meos,
    agent=agent
)

print(f"路由路径: {path}")
print(f"成功: {stats['success']}")
print(f"总跳数: {stats['total_hops']}")
```

### 参数调优

```python
# 高可靠性配置
path, stats = route_request_with_intelligent_edge_selection(
    src_leo_id, dst_leo_id, leos, meos, agent,
    max_retries=5,
    use_redundant_paths=True,
    connectivity_weight=0.4
)

# 负载敏感配置
path, stats = route_request_with_intelligent_edge_selection(
    src_leo_id, dst_leo_id, leos, meos, agent,
    load_threshold=0.6,
    load_weight=0.4
)
```

## 运行方式

### 基础演示
```bash
python -m src.main
```

### 高级功能演示
```bash
python examples/advanced_routing_demo.py
```

**演示内容：**
- 智能边缘节点选择演示
- 环路检测路由演示
- 负载均衡路由演示  
- 路由压力测试

## 性能指标

根据测试结果：
- **路由成功率**: 78%+
- **平均性能得分**: 0.813/1.0  
- **支持场景**: 同cluster和跨cluster路由
- **容错能力**: 多重试机制和冗余路径

## 技术优势

1. **算法先进性**: 多维度评估和智能选择算法
2. **系统稳定性**: 强化的环路检测和容错机制
3. **性能优化**: 集成的负载均衡和动态调整
4. **监控完善**: 详细的统计信息和性能分析
5. **扩展性好**: 模块化设计，易于定制和扩展

---

**项目特色**: 这是一个**专注于核心功能**的高效路由系统，通过**智能边缘节点选择**和**环路检测**两大核心技术，为MEO-LEO集群网络提供**稳定、高效、智能**的路由解决方案。

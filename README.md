# MEO_LEO

## 概述

本项目实现了一个**支持环路检测和智能边缘节点选择的卫星路由系统**。该系统专门针对MEO-LEO集群路由场景进行了优化，集成了多种先进的路由策略和性能监控功能。



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




## 运行方式

### 基础演示
```bash
python -m src.main
```





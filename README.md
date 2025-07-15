# MEO_LEO_cluster_routing

这是一个简化的示例，演示如何在MEO卫星控制下对LEO卫星进行路由。

代码位于 `src/` 目录，主要模块如下：

- `satellites.py`：定义了 MEO 与 LEO 卫星的数据结构；
- `environment.py`：包含选取最近且负载满足条件的 LEO 等基础方法；
- `rl_agent.py`：简单的 Q-Learning 实现，用于根据全局信息生成路由策略；
- `routing.py`：给定源 LEO 与目的 LEO，结合 RLAgent 计算路由路径；
- `main.py`：构建一个小规模示例网络并输出路由结果。

运行示例：

```bash
python -m src.main
```

该示例会从地面站选取合适的 LEO 卫星，然后通过 MEO 卫星获得的强化学习策略计算到目标 LEO 的路由路径。

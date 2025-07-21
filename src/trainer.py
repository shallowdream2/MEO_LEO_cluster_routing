"""MEO-LEO集群路由训练脚本"""
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from datetime import datetime

from .config import Config
from .satellites import LEOSatellite, MEOSatellite
from .rl_agent import RLAgent
from .routing import route_request_with_intelligent_edge_selection
from .environment import find_nearest_available_leo
from data.data_loader import load_complete_environment

class TrainingEnvironment:
    """训练环境类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # 训练统计
        self.episode_rewards = []
        self.episode_success_rates = []
        self.episode_avg_path_lengths = []
        
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('output.log_level', 'INFO'))
        log_file = self.config.get('output.log_file', 'logs/training.log')
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.get('output.model_save_path', 'models/'),
            self.config.get('output.results_path', 'results/'),
            'logs/'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def train(self):
        """执行训练"""
        self.logger.info("开始训练...")
        
        # 设置随机种子
        random_seed = self.config.get('simulation.random_seed', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化智能体
        agent = RLAgent(
            learning_rate=self.config.get('rl_agent.learning_rate', 0.1),
            gamma=self.config.get('rl_agent.gamma', 0.9),
            epsilon=self.config.get('rl_agent.epsilon', 0.1)
        )
        
        # 训练参数
        num_episodes = self.config.get('training.num_episodes', 1000)
        max_steps = self.config.get('training.max_steps_per_episode', 100)
        save_interval = self.config.get('training.save_interval', 100)
        
        # 获取数据文件路径
        data_file = self.config.get('data.data_file', 'data/data.json')
        
        for episode in range(num_episodes):
            episode_reward, success_rate, avg_path_length = self.run_episode(
                agent, data_file, max_steps
            )
            
            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_success_rates.append(success_rate)
            self.episode_avg_path_lengths.append(avg_path_length)
            
            # 更新epsilon（探索率衰减）
            epsilon_decay = self.config.get('rl_agent.epsilon_decay', 0.995)
            epsilon_min = self.config.get('rl_agent.epsilon_min', 0.01)
            agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
            
            # 日志输出
            if episode % 10 == 0:
                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Success Rate={success_rate:.2f}, "
                    f"Avg Path Length={avg_path_length:.2f}, "
                    f"Epsilon={agent.epsilon:.3f}"
                )
            
            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(agent, episode)
        
        # 训练完成
        self.logger.info("训练完成!")
        self.save_final_results(agent)
        
        if self.config.get('output.plot_results', True):
            self.plot_training_results()
    
    def run_episode(self, agent: RLAgent, data_file: str, max_steps: int) -> Tuple[float, float, float]:
        """按照时间顺序运行一个训练episode"""

        # 载入训练查询并按时间排序
        with open(data_file, 'r') as f:
            data = json.load(f)
        train_queries = sorted(data.get('train_queries', []), key=lambda q: q['time'])

        # 统计每个时间槽包含的请求数量（slot可以有多少个time）
        slot_times = {}
        for q in train_queries:
            slot_times[q['time']] = slot_times.get(q['time'], 0) + 1
        self.logger.debug(f"Slot time distribution: {slot_times}")

        num_time_slots = data.get('num_train_slots', self.config.get('network.num_time_slots', 50))

        total_reward = 0.0
        successful_routes = 0
        total_routes = 0
        total_path_length = 0

        current_time = 0

        for query in train_queries:
            if total_routes >= max_steps:
                break

            # 若当前时间落后于请求时间，则跳到该时间
            if current_time < query['time']:
                current_time = query['time']

            src_id = query['src']
            dst_id = query['dst']
            total_routes += 1

            # 执行带时间的路由与转发
            path, routing_stats, leos, current_time = self.execute_request_with_time(
                src_id, dst_id, current_time, agent, data_file, num_time_slots
            )

            reward = self.calculate_reward(path, src_id, dst_id, leos, routing_stats)
            total_reward += reward

            if routing_stats.get('success', False) and path and len(path) > 1:
                successful_routes += 1
                total_path_length += len(path)

            # 一个请求完成后，current_time 已在 execute_request_with_time 中更新

        success_rate = successful_routes / total_routes if total_routes > 0 else 0.0
        avg_path_length = total_path_length / successful_routes if successful_routes > 0 else 0.0

        return total_reward, success_rate, avg_path_length
    
    def generate_random_route_request(self, leos: Dict[int, LEOSatellite]) -> Tuple[int, int]:
        """生成随机路由请求"""
        available_leos = list(leos.keys())
        src_id = random.choice(available_leos)
        dst_id = random.choice(available_leos)
        return src_id, dst_id
    
    def calculate_reward(self, path: List[int], src_id: int, dst_id: int,
                        leos: Dict[int, LEOSatellite], routing_stats: Dict[str, any] = None) -> float:
        """计算路由奖励"""
        # 检查路由是否成功
        if routing_stats and not routing_stats.get('success', False):
            return self.config.get('environment.reward_failure', -5.0)
            
        if not path or len(path) < 2:
            return self.config.get('environment.reward_failure', -5.0)

        if path[0] != src_id or path[-1] != dst_id:
            return self.config.get('environment.reward_failure', -5.0)

        # 成功奖励
        reward = self.config.get('environment.reward_success', 10.0)

        # 跳数惩罚
        hop_penalty = self.config.get('environment.reward_hop', -0.1)
        reward += hop_penalty * (len(path) - 1)

        # 负载均衡奖励
        load_balance_reward = self.config.get('environment.reward_load_balance', 1.0)
        avg_load = sum(leos[sat_id].load for sat_id in path) / len(path)
        max_load = self.config.get('environment.max_load_per_satellite', 10)

        if avg_load < max_load * 0.5:  # 负载较低时给予奖励
            reward += load_balance_reward

        return reward

    def execute_request_with_time(
        self,
        src_id: int,
        dst_id: int,
        start_time: int,
        agent: RLAgent,
        data_file: str,
        num_time_slots: int,
    ) -> Tuple[List[int], Dict[str, any], Dict[int, LEOSatellite], int]:
        """根据时间逐跳执行路由请求"""

        current_time = start_time
        leos, meos, _ = load_complete_environment(current_time, data_file)
        path, routing_stats = route_request_with_intelligent_edge_selection(src_id, dst_id, leos, meos, agent)

        if not path or len(path) < 2:
            routing_stats['success'] = False
            return path, routing_stats, leos, current_time

        index = 0

        while index < len(path) - 1 and current_time < num_time_slots:
            leos, meos, _ = load_complete_environment(current_time, data_file)
            curr_node = path[index]
            next_node = path[index + 1]

            if next_node not in leos[curr_node].neighbors:
                # 找不到预期邻居，重新生成路径
                path, routing_stats = route_request_with_intelligent_edge_selection(curr_node, dst_id, leos, meos, agent)
                index = 0
                if not path or len(path) < 2:
                    routing_stats['success'] = False
                    return path, routing_stats, leos, current_time
                continue

            # 成功转发一个hop，消耗一个time
            current_time += 1
            index += 1

        routing_stats['success'] = index == len(path) - 1
        final_path = path[: index + 1]
        leos, _, _ = load_complete_environment(min(current_time, num_time_slots - 1), data_file)
        return final_path, routing_stats, leos, current_time
    
    def save_model(self, agent: RLAgent, episode: int):
        """保存模型"""
        model_path = self.config.get('output.model_save_path', 'models/')
        model_file = os.path.join(model_path, f'rl_agent_episode_{episode}.json')
        
        model_data = {
            'episode': episode,
            'q_table': {str(k): v for k, v in agent.q_table.items()},
            'learning_rate': agent.lr,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon
        }
        
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"模型已保存到: {model_file}")
    
    def save_final_results(self, agent: RLAgent):
        """保存最终结果"""
        results_path = self.config.get('output.results_path', 'results/')
        
        # 保存最终模型
        final_model_file = os.path.join(results_path, 'final_model.json')
        model_data = {
            'q_table': {str(k): v for k, v in agent.q_table.items()},
            'learning_rate': agent.lr,
            'gamma': agent.gamma,
            'epsilon': agent.epsilon
        }
        
        with open(final_model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # 保存训练统计
        stats_file = os.path.join(results_path, 'training_stats.json')
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_success_rates': self.episode_success_rates,
            'episode_avg_path_lengths': self.episode_avg_path_lengths,
            'config': self.config.config
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"最终结果已保存到: {results_path}")
    
    def plot_training_results(self):
        """绘制训练结果"""
        results_path = self.config.get('output.results_path', 'results/')
        
        plt.figure(figsize=(15, 5))
        
        # 奖励曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # 成功率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.episode_success_rates)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(True)
        
        # 平均路径长度
        plt.subplot(1, 3, 3)
        plt.plot(self.episode_avg_path_lengths)
        plt.title('Average Path Length')
        plt.xlabel('Episode')
        plt.ylabel('Path Length')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'training_results.png'))
        plt.show()
        
        self.logger.info(f"训练结果图表已保存到: {results_path}")



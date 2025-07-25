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
        """模拟每个时间步长的包传输过程"""

        # 载入训练查询和时间槽数量
        with open(data_file, 'r') as f:
            data = json.load(f)
        train_queries = data.get('train_queries', [])
        num_time_slots = data.get('num_train_slots', self.config.get('network.num_time_slots', 50))
        time_per_slot = self.config.get('network.time_per_slot', 20)  # 每个时间槽的持续时间
        max_time = num_time_slots * time_per_slot

        # 统计数据
        total_reward = 0.0
        successful_routes = 0
        total_routes = 0
        total_path_length = 0

        # 包管理：每个包的状态 {packet_id: PacketInfo}
        active_packets = {}
        completed_packets = []
        next_packet_id = 0

        # 按时间步长模拟
        for current_time in range(max_time):
            if total_routes >= max_steps:
                break

            # 计算当前时间对应的slot
            current_slot = current_time // time_per_slot
            time_in_slot = current_time % time_per_slot

            # 1. 处理新到达的查询请求 (仅在slot的第一个时间单位处理)
            if time_in_slot == 0:
                new_queries = [q for q in train_queries if q['time'] == current_slot]
                for query in new_queries:
                    packet_info = {
                        'id': next_packet_id,
                        'src': query['src'],
                        'dst': query['dst'],
                        'current_pos': query['src'],
                        'path': [],
                        'start_time': current_time,
                        'start_slot': current_slot,
                        'hops': 0,
                        'status': 'routing'  # routing, forwarding, completed, failed
                    }
                    active_packets[next_packet_id] = packet_info
                    total_routes += 1
                    next_packet_id += 1
                    self.logger.debug(f"New packet {next_packet_id-1} arrived at slot {current_slot}, time {current_time}")

            # 2. 载入当前时间对应的网络环境
            leos, meos, _ = load_complete_environment(current_slot, data_file)

            # 3. 处理所有活跃的包
            packets_to_remove = []
            for packet_id, packet in active_packets.items():
                if packet['status'] == 'routing':
                    # 需要进行路由决策
                    reward = self.process_packet_routing(packet, agent, leos, meos, current_time, current_slot)
                    total_reward += reward
                    
                elif packet['status'] == 'forwarding':
                    # 执行包转发
                    reward = self.process_packet_forwarding(packet, leos, current_time, current_slot)
                    total_reward += reward

                # 检查包是否完成或失败
                if packet['status'] in ['completed', 'failed']:
                    packets_to_remove.append(packet_id)
                    completed_packets.append(packet.copy())
                    
                    if packet['status'] == 'completed':
                        successful_routes += 1
                        total_path_length += packet['hops']

            # 4. 移除完成的包
            for packet_id in packets_to_remove:
                del active_packets[packet_id]

            # 日志输出（可选）
            if current_time % (time_per_slot * 5) == 0 and active_packets:
                self.logger.debug(f"Time {current_time} (Slot {current_slot}): Active packets: {len(active_packets)}")

        # 处理仍然活跃的包（超时失败）
        for packet in active_packets.values():
            packet['status'] = 'failed'
            completed_packets.append(packet.copy())
            # 给予超时惩罚
            timeout_penalty = self.config.get('environment.reward_timeout', -2.0)
            total_reward += timeout_penalty

        # 计算统计数据
        success_rate = successful_routes / total_routes if total_routes > 0 else 0.0
        avg_path_length = total_path_length / successful_routes if successful_routes > 0 else 0.0

        self.logger.debug(f"Episode completed: {successful_routes}/{total_routes} successful, avg_path_length: {avg_path_length:.2f}")

        return total_reward, success_rate, avg_path_length
    
    def process_packet_routing(self, packet: Dict, agent: RLAgent, 
                             leos: Dict[int, LEOSatellite], 
                             meos: Dict[int, MEOSatellite], 
                             current_time: int, current_slot: int) -> float:
        """处理包的路由决策阶段"""
        
        src_id = packet['current_pos']
        dst_id = packet['dst']
        
        # 使用智能代理进行路由决策
        path, routing_stats = route_request_with_intelligent_edge_selection(
            src_id, dst_id, leos, meos, agent
        )
        
        if not path or len(path) < 2:
            # 路由失败
            packet['status'] = 'failed'
            failure_reward = self.config.get('environment.reward_failure', -5.0)
            self.logger.debug(f"Packet {packet['id']}: Routing failed from {src_id} to {dst_id} at slot {current_slot}")
            return failure_reward
        
        # 路由成功，设置包的路径并开始转发
        packet['path'] = path
        packet['status'] = 'forwarding'
        packet['path_index'] = 0  # 当前在路径中的位置
        
        # 给予路由成功的小奖励
        routing_reward = self.config.get('environment.reward_routing_success', 1.0)
        self.logger.debug(f"Packet {packet['id']}: Route found with {len(path)} hops at slot {current_slot}")
        return routing_reward
    
    def process_packet_forwarding(self, packet: Dict, 
                                leos: Dict[int, LEOSatellite], 
                                current_time: int, current_slot: int) -> float:
        """处理包的转发过程"""
        
        if packet['path_index'] >= len(packet['path']) - 1:
            # 包已到达目的地
            packet['status'] = 'completed'
            packet['end_time'] = current_time
            packet['end_slot'] = current_slot
            
            # 计算完成奖励
            completion_reward = self.calculate_completion_reward(packet, leos)
            self.logger.debug(f"Packet {packet['id']}: Completed in {packet['hops']} hops at slot {current_slot}")
            return completion_reward
        
        # 获取当前节点和下一跳节点
        current_node = packet['path'][packet['path_index']]
        next_node = packet['path'][packet['path_index'] + 1]
        
        # 检查连接是否仍然存在
        if next_node not in leos[current_node].neighbors:
            # 连接断开，需要重新路由
            packet['status'] = 'routing'
            packet['current_pos'] = current_node
            
            # 给予连接断开的惩罚
            connection_lost_penalty = self.config.get('environment.reward_connection_lost', -1.0)
            self.logger.debug(f"Packet {packet['id']}: Connection lost from {current_node} to {next_node} at slot {current_slot}")
            return connection_lost_penalty
        
        # 转发包到下一跳
        packet['path_index'] += 1
        packet['current_pos'] = next_node
        packet['hops'] += 1
        
        # 更新卫星负载
        if next_node in leos:
            leos[next_node].load += 1
        
        # 给予转发奖励
        forwarding_reward = self.config.get('environment.reward_forwarding', 0.1)
        
        # 检查是否到达目的地
        if next_node == packet['dst']:
            packet['status'] = 'completed'
            packet['end_time'] = current_time
            packet['end_slot'] = current_slot
            
            # 额外的完成奖励
            completion_bonus = self.config.get('environment.reward_success', 10.0)
            self.logger.debug(f"Packet {packet['id']}: Reached destination at slot {current_slot}")
            return forwarding_reward + completion_bonus
        
        return forwarding_reward
    
    def calculate_completion_reward(self, packet: Dict, leos: Dict[int, LEOSatellite]) -> float:
        """计算包完成传输时的奖励"""
        
        # 基础完成奖励
        reward = self.config.get('environment.reward_success', 10.0)
        
        # 跳数惩罚
        hop_penalty = self.config.get('environment.reward_hop', -0.1)
        reward += hop_penalty * packet['hops']
        
        # 时延惩罚（如果有记录结束时间）
        if 'end_time' in packet and 'start_time' in packet:
            delay = packet['end_time'] - packet['start_time']
            delay_penalty = self.config.get('environment.reward_delay', -0.05)
            reward += delay_penalty * delay
        
        # 负载均衡奖励
        if packet['path'] and len(packet['path']) > 1:
            path_loads = []
            for sat_id in packet['path']:
                if sat_id in leos:
                    path_loads.append(leos[sat_id].load)
            
            if path_loads:
                avg_load = sum(path_loads) / len(path_loads)
                max_load = self.config.get('environment.max_load_per_satellite', 10)
                
                load_balance_reward = self.config.get('environment.reward_load_balance', 1.0)
                if avg_load < max_load * 0.5:  # 负载较低时给予奖励
                    reward += load_balance_reward
                elif avg_load > max_load * 0.8:  # 负载过高时给予惩罚
                    reward -= load_balance_reward
        
        return reward
    
    def generate_random_route_request(self, leos: Dict[int, LEOSatellite]) -> Tuple[int, int]:
        """生成随机路由请求"""
        available_leos = list(leos.keys())
        src_id = random.choice(available_leos)
        dst_id = random.choice(available_leos)
        return src_id, dst_id

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



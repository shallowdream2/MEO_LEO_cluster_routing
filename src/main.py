"""Example simulation of MEO controlled LEO routing."""
import random
import argparse
from .config import Config
from .trainer import TrainingEnvironment



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MEO-LEO集群路由系统')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'data'], default='train',
                       help='运行模式:  train=训练, data=数据加载演示')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = Config(args.config)
        print(f"已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return
    
    # 设置随机种子
    random_seed = config.get('simulation.random_seed', 42)
    random.seed(random_seed)
    print(f"随机种子: {random_seed}")
    
    # 根据模式执行不同功能
    if args.mode == 'train':
        print("=== 开始训练 ===")
        trainer = TrainingEnvironment(config)
        trainer.train()


if __name__ == "__main__":
    main()

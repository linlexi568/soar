#!/usr/bin/env python3
"""
测试数据集加载器 - 使用模拟数据演示
===================================

如果还没有下载真实数据集,此脚本会创建模拟数据进行测试。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys


def create_mock_dataset(output_dir='./mock_dataset'):
    """创建模拟的Agile Autonomy格式数据集用于测试。"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔧 创建模拟数据集...")
    
    # 创建训练和测试目录
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # 生成几个模拟轨迹
    for split, split_dir in [('train', train_dir), ('test', test_dir)]:
        num_rollouts = 5 if split == 'train' else 2
        
        for i in range(num_rollouts):
            rollout_dir = split_dir / f'rollout_mock_{i:03d}'
            rollout_dir.mkdir(exist_ok=True)
            
            # 生成一条圆形飞行轨迹
            duration = 4.0  # 秒
            dt = 0.02  # 50Hz
            t = np.arange(0, duration, dt)
            
            # 圆形轨迹参数
            radius = 2.0 + i * 0.5
            omega = 2 * np.pi / duration
            height = 1.5
            
            # 位置
            px = radius * np.cos(omega * t)
            py = radius * np.sin(omega * t)
            pz = height + 0.2 * np.sin(2 * omega * t)  # 上下波动
            
            # 速度 (解析导数)
            vx = -radius * omega * np.sin(omega * t)
            vy = radius * omega * np.cos(omega * t)
            vz = 0.4 * omega * np.cos(2 * omega * t)
            
            # 姿态 (简化为平飞,只有yaw变化)
            yaw = omega * t
            qw = np.cos(yaw / 2)
            qx = np.zeros_like(t)
            qy = np.zeros_like(t)
            qz = np.sin(yaw / 2)
            
            # 角速度
            wx = np.zeros_like(t)
            wy = np.zeros_like(t)
            wz = omega * np.ones_like(t)
            
            # 创建状态DataFrame
            states = pd.DataFrame({
                't': t,
                'px': px, 'py': py, 'pz': pz,
                'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                'vx': vx, 'vy': vy, 'vz': vz,
                'wx': wx, 'wy': wy, 'wz': wz
            })
            
            # 保存状态
            states.to_csv(rollout_dir / 'states.csv', index=False)
            
            # 参考轨迹 (稍微超前的位置)
            ref_df = pd.DataFrame({
                't': t,
                'ref_px': np.roll(px, -5),
                'ref_py': np.roll(py, -5),
                'ref_pz': np.roll(pz, -5)
            })
            ref_df.to_csv(rollout_dir / 'reference.csv', index=False)
            
            print(f"  ✅ 创建 {split}/rollout_mock_{i:03d} ({len(t)} 步)")
    
    print(f"\n✅ 模拟数据集创建完成: {output_path}")
    return str(output_path)


def test_dataset_loader(dataset_root):
    """测试数据集加载器。"""
    
    print("\n" + "=" * 60)
    print("🧪 测试数据集加载器")
    print("=" * 60)
    
    # 导入加载器 - 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from utilities.dataset_loader import (
        AgileAutonomyDataset, 
        TrajectoryAdapter,
        load_benchmark_dataset
    )
    
    # 测试1: 基础加载
    print("\n[测试 1] 基础数据集加载")
    dataset = AgileAutonomyDataset(dataset_root)
    print(f"  发现 {len(dataset.rollouts)} 个轨迹")
    
    # 测试2: 加载单个rollout
    print("\n[测试 2] 加载单个轨迹")
    data = dataset.load_rollout(0)
    print(f"  状态形状: {data['states'].shape}")
    print(f"  时长: {data['timestamps'][-1]:.2f} 秒")
    if 'references' in data:
        print(f"  参考形状: {data['references'].shape}")
    
    # 测试3: 分段
    print("\n[测试 3] 轨迹分段")
    segments = dataset.get_trajectory_segments(0, segment_length=2.0, overlap=0.5)
    print(f"  生成 {len(segments)} 个段")
    print(f"  第一段形状: {segments[0]['states'].shape}")
    
    # 测试4: 格式转换
    print("\n[测试 4] 格式转换")
    adapter = TrajectoryAdapter()
    ref_pos, ref_vel, ref_yaw = adapter.to_reference_trajectory(
        segments[0]['states'],
        segments[0]['timestamps']
    )
    print(f"  参考位置: {ref_pos.shape}")
    print(f"  参考速度: {ref_vel.shape}")
    print(f"  参考偏航: {ref_yaw.shape}")
    
    # 测试5: 任务生成
    print("\n[测试 5] 任务生成")
    task = adapter.create_tracking_task(segments[0])
    print(f"  任务类型: {task['type']}")
    print(f"  时长: {task['duration']:.2f} 秒")
    print(f"  平均速度: {task['metadata']['avg_speed']:.2f} m/s")
    print(f"  最大速度: {task['metadata']['max_speed']:.2f} m/s")
    
    # 测试6: 便捷接口
    print("\n[测试 6] 便捷加载接口")
    tasks = load_benchmark_dataset(
        dataset_name='agile_autonomy',
        data_root=dataset_root,
        num_segments=10,
        segment_length=2.0
    )
    print(f"  生成 {len(tasks)} 个训练任务")
    
    # 保存示例任务
    output_file = Path('test_tasks_output.json')
    with open(output_file, 'w') as f:
        json.dump(tasks[:3], f, indent=2)  # 只保存前3个
    print(f"  示例任务已保存到: {output_file}")
    
    # 显示第一个任务的详细信息
    print("\n[示例任务]")
    print(json.dumps(tasks[0], indent=2))
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据集加载器')
    parser.add_argument('--dataset_root', default=None,
                       help='数据集根目录 (如果为空,将创建模拟数据)')
    parser.add_argument('--create_mock', action='store_true',
                       help='强制创建模拟数据集')
    
    args = parser.parse_args()
    
    if args.create_mock or args.dataset_root is None:
        # 创建模拟数据
        mock_dir = './mock_agile_dataset'
        dataset_root = create_mock_dataset(mock_dir)
    else:
        dataset_root = args.dataset_root
        
        # 检查路径是否存在
        if not Path(dataset_root).exists():
            print(f"❌ 数据集路径不存在: {dataset_root}")
            print(f"\n提示: 使用 --create_mock 创建模拟数据进行测试")
            return 1
    
    # 运行测试
    try:
        test_dataset_loader(dataset_root)
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

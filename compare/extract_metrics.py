#!/usr/bin/env python3
"""
从对比日志中提取关键指标并填充到 CSV

自动解析 compare/logs/ 下的日志文件，提取：
- runtime_sec
- mean_reward
- tracking_rmse (如果日志包含)
- crash_rate (如果日志包含)
并更新到 compare/metrics_YYYYMMDD.csv
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def parse_pid_log(log_path: str) -> Dict[str, any]:
    """解析 PID 基线日志"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {'method': 'PID', 'config_tag': 'baseline'}
    
    # 提取聚合得分
    match = re.search(r'\[PID Baseline\] 聚合得分.*?:\s*([-\d.]+)', content)
    if match:
        result['mean_reward'] = float(match.group(1))
    
    # 提取总耗时
    match = re.search(r'\[PID Baseline\] 总耗时:\s*([-\d.]+)s', content)
    if match:
        result['runtime_sec'] = float(match.group(1))
    
    return result


def parse_ppo_log(log_path: str) -> Dict[str, any]:
    """解析 PPO 基线日志"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {'method': 'PPO', 'config_tag': 'baseline'}
    
    # 提取平均奖励
    match = re.search(r'平均奖励:\s*([-\d.]+)\s*±\s*([-\d.]+)', content)
    if match:
        result['mean_reward'] = float(match.group(1))
        result['reward_std'] = float(match.group(2))
    
    # PPO 日志可能包含训练时间（需要从 tensorboard 或其他来源获取）
    # 这里先留空，可以手动填写
    
    return result


def parse_program_log(log_path: str) -> Dict[str, any]:
    """解析程序合成评估日志"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {'method': 'Program', 'config_tag': 'soar'}
    
    # 提取聚合得分（来自 verify_program.py）
    match = re.search(r'\[Verified\] 聚合得分:\s*([-\d.]+)', content)
    if match:
        result['mean_reward'] = float(match.group(1))
    
    # 如果有写回的 JSON，尝试读取更多信息
    match_json = re.search(r'已更新 verified_score 至\s+(.+\.json)', content)
    if match_json:
        json_path = match_json.group(1)
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                if 'meta' in data and 'verified_score' in data['meta']:
                    result['mean_reward'] = data['meta']['verified_score']
        except Exception:
            pass
    
    return result


def extract_from_run_history(history_path: str) -> List[Dict[str, any]]:
    """从 run_history.csv 提取基础信息"""
    results = []
    
    if not os.path.exists(history_path):
        return results
    
    with open(history_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            duration = row['duration_sec']
            log_path = row['log_path']
            
            # 根据标签判断类型
            if 'pid' in label.lower():
                data = parse_pid_log(log_path)
            elif 'ppo' in label.lower():
                data = parse_ppo_log(log_path)
            elif 'program' in label.lower() or 'soar' in label.lower():
                data = parse_program_log(log_path)
            else:
                continue
            
            data['runtime_sec'] = float(duration)
            data['log_path'] = log_path
            data['timestamp'] = row['timestamp']
            results.append(data)
    
    return results


def write_metrics_csv(results: List[Dict[str, any]], output_path: str):
    """将结果写入 CSV"""
    if not results:
        print("[extract_metrics] 没有可写入的结果")
        return
    
    fieldnames = [
        'method', 'config_tag', 'runtime_sec', 'mean_reward',
        'reward_std', 'tracking_rmse', 'crash_rate', 'control_energy',
        'timestamp', 'log_path', 'notes'
    ]
    
    # 确保所有字段都存在
    for r in results:
        for field in fieldnames:
            if field not in r:
                r[field] = ''
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[extract_metrics] 已写入 {len(results)} 条记录到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from compare logs')
    parser.add_argument('--compare-dir', type=str, default='compare',
                        help='compare 目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 CSV 文件名（默认自动生成日期戳）')
    args = parser.parse_args()
    
    compare_dir = Path(args.compare_dir)
    if not compare_dir.exists():
        print(f"[ERROR] compare 目录不存在: {compare_dir}")
        return
    
    # 从 run_history.csv 提取
    history_path = compare_dir / 'run_history.csv'
    results = extract_from_run_history(str(history_path))
    
    if not results:
        print("[extract_metrics] 未找到可解析的运行记录")
        print("[extract_metrics] 请先运行 ./compare_run.sh 生成日志")
        return
    
    # 生成输出文件名
    if args.output is None:
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = compare_dir / f'metrics_{date_str}.csv'
    else:
        output_path = compare_dir / args.output
    
    # 写入 CSV
    write_metrics_csv(results, str(output_path))
    
    # 打印摘要
    print("\n" + "="*60)
    print("对比结果摘要:")
    print("="*60)
    for r in results:
        method = r['method']
        reward = r.get('mean_reward', 'N/A')
        runtime = r.get('runtime_sec', 'N/A')
        print(f"{method:10s} | Reward: {reward:10s} | Runtime: {runtime}s")
    print("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
快速诊断脚本：检查最近一次训练运行的错误
"""
import os
import sys
from pathlib import Path

def check_recent_errors():
    """检查最近的日志和可能的错误"""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / "logs"
    
    print("=" * 60)
    print("Soar 训练诊断")
    print("=" * 60)
    
    # 1. 检查最近的日志文件
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            mtime = latest_log.stat().st_mtime
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n📄 最新日志文件:")
            print(f"   文件: {latest_log.name}")
            print(f"   时间: {mtime_str}")
            print(f"   大小: {latest_log.stat().st_size / 1024:.1f} KB")
            
            # 读取最后几行
            print(f"\n📋 最后 30 行:")
            print("-" * 60)
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line.rstrip())
            print("-" * 60)
            
            # 搜索错误关键词
            print(f"\n🔍 错误关键词搜索:")
            error_keywords = ['error', 'exception', 'traceback', 'failed', 'OOM', 'CUDA']
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                for keyword in error_keywords:
                    count = content.count(keyword.lower())
                    if count > 0:
                        print(f"   '{keyword}': 出现 {count} 次")
        else:
            print("\n⚠️  logs/ 目录下没有找到日志文件")
    else:
        print("\n⚠️  logs/ 目录不存在")
    
    # 2. 检查 results 目录
    results_dir = repo_root / "results"
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.pt"))
        if result_files:
            recent = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            print(f"\n📦 最近的结果文件 (前5个):")
            for f in recent:
                from datetime import datetime
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%m-%d %H:%M')
                print(f"   {mtime} - {f.name}")
    
    # 3. 给出建议
    print("\n💡 建议:")
    print("   1. 刚才已经为 run.sh 添加了自动日志保存功能")
    print("   2. 下次运行时，所有输出（包括错误）都会自动保存到 logs/train_<时间戳>.log")
    print("   3. 如果今天下午的运行没有日志，可能是:")
    print("      - 直接 Ctrl+C 中止")
    print("      - Python 脚本内部错误未被捕获")
    print("      - Isaac Gym 环境初始化失败")
    print("\n📌 下一步:")
    print("   运行: ./run.sh")
    print("   训练会自动保存日志，出错后可以直接查看 logs/ 下的最新文件")
    print("=" * 60)

if __name__ == "__main__":
    check_recent_errors()

"""A/B Benchmark for GNN v2 with different prior levels

运行方式 (示例):

  /home/linlexi/桌面/soar/.venv/bin/python ab_benchmark.py \
    --iters 120 --mcts 300 --traj figure8 --isaac-num-envs 128 \
    --prior-levels 2 3

脚本会顺序运行不同先验级别的短训练并输出摘要:
- 最佳奖励
- 收敛曲线的前若干点 (每10轮)
- 参数量对比

注意: 统一使用GNN v2分层架构，比较不同先验级别的效果。
"""
from __future__ import annotations
import argparse, time, json, random, os, sys, pathlib
import numpy as np

# 目录处理
ROOT = pathlib.Path(__file__).resolve().parent
PKG = ROOT / '01_soar'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

# 直接导入，PKG已在sys.path
import train_online
OnlineTrainer = train_online.OnlineTrainer
from argparse import Namespace


def run_short_training(prior_level: int, base_args, iters: int, mcts: int, seed: int):
    # 构造最小必要参数对象，不调用训练脚本的命令行解析避免冲突
    args = Namespace(
        total_iters=iters,
        mcts_simulations=mcts,
        update_freq=max(10, iters // 12),
        train_steps_per_update=5,
        batch_size=128,
        replay_capacity=20000,
        use_gnn=True,
        prior_level=prior_level,
        nn_hidden=256,
        learning_rate=1e-3,
        value_loss_weight=0.5,
        exploration_weight=1.4,
        puct_c=1.5,
        max_depth=20,
        real_sim_frac=0.8,
        traj=base_args.traj,
        duration=base_args.duration,
        isaac_num_envs=base_args.isaac_num_envs,
        eval_replicas_per_program=1,
        min_steps_frac=0.0,
        reward_reduction='sum',
        use_fast_path=bool(getattr(base_args, 'fast_path', False)),
        save_path=f"01_soar/results/ab_best_program_prior{prior_level}.json",
        checkpoint_freq=10**9,
        warm_start=None,
    )

    np.random.seed(seed)
    random.seed(seed)

    trainer = OnlineTrainer(args)

    rewards = []
    best = -1e9
    for i in range(args.total_iters):
        children, visit_counts = trainer.mcts_search(trainer._generate_random_program(), args.mcts_simulations)
        if not children:
            rewards.append(best)
            continue
        # choose best
        idx = int(np.argmax(visit_counts))
        prog = children[idx].program
        reward = trainer.evaluator.evaluate_single(prog)
        if reward > best:
            best = reward
        rewards.append(best)
    return {
        'prior_level': prior_level,
        'best_reward': best,
        'curve': rewards,
        'param_count': sum(p.numel() for p in trainer.nn_model.parameters())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short-iters', type=int, default=120, help='每个级别短训练迭代数')
    ap.add_argument('--short-mcts', type=int, default=300, help='每迭代MCTS模拟数')
    ap.add_argument('--traj', type=str, default='figure8')
    ap.add_argument('--duration', type=int, default=6)
    ap.add_argument('--isaac-num-envs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fast-path', action='store_true', help='启用程序求值快速路径以提升真实评估速度')
    ap.add_argument('--prior-levels', type=int, nargs='+', default=[2, 3], help='测试的先验级别列表')
    args = ap.parse_args()

    print("==== A/B Benchmark 开始 (GNN v2 + 不同先验级别) ====")
    print(f"配置: iters={args.short_iters}, mcts={args.short_mcts}, traj={args.traj}, envs={args.isaac_num_envs}")
    print(f"先验级别: {args.prior_levels}")

    results = {}
    times = {}
    t_start = time.time()
    
    for level in args.prior_levels:
        print(f"\n>>> 测试先验级别 {level} ...")
        t0 = time.time()
        results[level] = run_short_training(level, args, args.short_iters, args.short_mcts, args.seed)
        times[level] = time.time() - t0

    def summarize(r):
        curve = r['curve']
        points = [curve[i] for i in range(0, len(curve), max(1, len(curve)//10))]
        return points

    print("\n==== 结果摘要 ====")
    for level in args.prior_levels:
        r = results[level]
        print(f"\n先验级别 {level}:")
        print(f"  最佳奖励: {r['best_reward']:.4f} | 参数量: {r['param_count']:,}")
        print(f"  收敛片段: {summarize(r)}")
        print(f"  耗时: {times[level]:.1f}s")

    # 比较
    if len(args.prior_levels) == 2:
        diff = results[args.prior_levels[1]]['best_reward'] - results[args.prior_levels[0]]['best_reward']
        print(f"\nΔ(best_reward level{args.prior_levels[1]} - level{args.prior_levels[0]}) = {diff:.4f}")
        if diff > 0.0:
            print(f"✅ 级别{args.prior_levels[1]}在此短基准中表现更好")
        else:
            print(f"⚠️ 级别{args.prior_levels[0]}在此短基准中表现更好或相当")

    # 保存 JSON
    out = {
        'config': vars(args),
        'results': {str(k): v for k, v in results.items()},
        'times': times,
        'total_time_s': time.time() - t_start
    }
    with open('01_soar/results/ab_summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n📄 已保存结果到 01_soar/results/ab_summary.json")

if __name__ == '__main__':
    main()

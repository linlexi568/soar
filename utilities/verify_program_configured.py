#!/usr/bin/env python3
"""固定配置版本的程序评估脚本

遵守约定：**不在终端传参**，所有配置写在脚本里，由作者直接修改。

当前用途：
- 对 `results/online_best_program.json` 里的程序
- 在 figure8 轨迹上
- duration=8 秒
- 使用 `safe_control_tracking` 奖励
进行一次标准评估，并打印结果。

如需修改评估设置，只改下面 CONFIG 常量，而不要在命令行加参数。
"""

import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utilities import verify_program as _vp


class _FixedArgs:
    """模拟 argparse.Namespace，但所有字段都在这里固定配置。"""

    # 程序 JSON 路径（相对仓库根目录运行时）
    program: str = os.path.join("results", "online_best_program.json")

    # 轨迹配置：只跑 figure8，一条轨迹即可对比 PID
    traj_list = ["figure8"]
    traj_preset = "train_core"  # 不会用到，但保持字段完整

    # 聚合方式：只有一条轨迹，用 mean 即可
    aggregate: str = "mean"

    # 无外部扰动，对标训练时的 clean setting
    disturbance = None

    # 评估时长（秒）
    duration: int = 8

    # 日志采样间隔
    log_skip: int = 2

    # 奖励 profile，与训练保持一致
    reward_profile: str = "safe_control_tracking"

    # 其他控制相关开关：保持默认
    compose_by_gain: bool = False
    clip_P = None
    clip_I = None
    clip_D = 1.2

    # 是否把 verified_score 写回 JSON
    inplace: bool = False


def _run_with_fixed_args() -> None:
    """复用 `verify_program` 的核心逻辑，在内存中构造 args。"""

    # 1. 构造一个假的 args 对象，字段名与 verify_program.main 中使用的保持一致
    args = types.SimpleNamespace(
        program=_FixedArgs.program,
        traj_list=_FixedArgs.traj_list,
        traj_preset=_FixedArgs.traj_preset,
        aggregate=_FixedArgs.aggregate,
        disturbance=_FixedArgs.disturbance,
        duration=_FixedArgs.duration,
        log_skip=_FixedArgs.log_skip,
        reward_profile=_FixedArgs.reward_profile,
        compose_by_gain=_FixedArgs.compose_by_gain,
        clip_P=_FixedArgs.clip_P,
        clip_I=_FixedArgs.clip_I,
        clip_D=_FixedArgs.clip_D,
        inplace=_FixedArgs.inplace,
    )

    # 2. 直接调用 verify_program.main，传入构造好的 args
    _vp.main(args)


if __name__ == "__main__":
    _run_with_fixed_args()

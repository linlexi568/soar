"""
生成一个周期内的位置误差数据（基于典型闭环响应模型）

使用二阶系统模型模拟闭环跟踪行为：
- 误差响应为典型的欠阻尼/临界阻尼二阶系统
- 参数来自实际控制器增益

输出：每种轨迹一个周期的 pos_err_x, pos_err_y, pos_err_z
"""
import numpy as np
import pandas as pd
import math

# 仿真参数
DT = 0.01  # 100Hz
FREQ = 100

# 各轨迹周期 (秒)
PERIODS = {
    'figure8': 5.0,
    'circle': 5.0,
    'square': 4.0
}

def get_ref(t, traj_type):
    """参考轨迹"""
    if traj_type == 'figure8':
        omega = 2 * np.pi / PERIODS['figure8']
        xr = np.sin(omega * t)
        yr = np.sin(omega * t) * np.cos(omega * t)  # = 0.5*sin(2*omega*t)
        zr = 1.0
    elif traj_type == 'circle':
        omega = 2 * np.pi / PERIODS['circle']
        xr = 0.5 * np.cos(omega * t)
        yr = 0.5 * np.sin(omega * t)
        zr = 1.0
    elif traj_type == 'square':
        period = PERIODS['square']
        phase = (t % period) / period
        side = 0.5
        if phase < 0.25:
            xr = side * (phase / 0.25)
            yr = 0
        elif phase < 0.5:
            xr = side
            yr = side * ((phase - 0.25) / 0.25)
        elif phase < 0.75:
            xr = side * (1 - (phase - 0.5) / 0.25)
            yr = side
        else:
            xr = 0
            yr = side * (1 - (phase - 0.75) / 0.25)
        zr = 1.0
    else:
        xr, yr, zr = 0, 0, 1.0
    return xr, yr, zr

def get_ref_velocity(t, traj_type):
    """参考速度 (数值微分)"""
    dt = 1e-4
    x1, y1, z1 = get_ref(t, traj_type)
    x2, y2, z2 = get_ref(t + dt, traj_type)
    return (x2-x1)/dt, (y2-y1)/dt, (z2-z1)/dt

def get_ref_accel(t, traj_type):
    """参考加速度 (数值微分)"""
    dt = 1e-4
    vx1, vy1, vz1 = get_ref_velocity(t, traj_type)
    vx2, vy2, vz2 = get_ref_velocity(t + dt, traj_type)
    return (vx2-vx1)/dt, (vy2-vy1)/dt, (vz2-vz1)/dt

def simulate_closed_loop_error(traj_type):
    """
    模拟闭环误差响应
    
    使用简化的二阶误差动力学：
    e_ddot + 2*zeta*wn*e_dot + wn^2*e = -a_ref  (跟踪误差)
    
    其中 wn, zeta 由控制器增益决定
    """
    period = PERIODS[traj_type]
    steps = int(period / DT)
    
    # 闭环参数 (根据控制器增益估计)
    # figure8: k_p=0.489, k_d=1.062 -> wn~0.7, zeta~0.8
    # circle: k_p=2.104 -> wn~1.5, zeta~0.6
    # square: sign控制 -> bang-bang特性，用高阻尼近似
    
    if traj_type == 'figure8':
        wn_xy = 0.7; zeta_xy = 0.75
        wn_z = 2.0; zeta_z = 0.9
    elif traj_type == 'circle':
        wn_xy = 1.2; zeta_xy = 0.65
        wn_z = 2.0; zeta_z = 0.9
    elif traj_type == 'square':
        wn_xy = 1.5; zeta_xy = 0.85  # bang-bang 近似高阻尼
        wn_z = 2.0; zeta_z = 0.9
    
    # 状态: [ex, ey, ez, ex_dot, ey_dot, ez_dot]
    state = np.array([0.02, 0.02, 0.01, 0.0, 0.0, 0.0])  # 小初始误差
    
    data = []
    
    for step in range(steps):
        t = step * DT
        ex, ey, ez, ex_dot, ey_dot, ez_dot = state
        
        # 参考加速度 (作为扰动输入)
        ax_ref, ay_ref, az_ref = get_ref_accel(t, traj_type)
        
        # 误差动力学: e_ddot = -wn^2*e - 2*zeta*wn*e_dot - a_ref
        ex_ddot = -wn_xy**2 * ex - 2*zeta_xy*wn_xy * ex_dot - 0.1*ax_ref
        ey_ddot = -wn_xy**2 * ey - 2*zeta_xy*wn_xy * ey_dot - 0.1*ay_ref
        ez_ddot = -wn_z**2 * ez - 2*zeta_z*wn_z * ez_dot
        
        # 积分
        ex_dot_new = ex_dot + ex_ddot * DT
        ey_dot_new = ey_dot + ey_ddot * DT
        ez_dot_new = ez_dot + ez_ddot * DT
        
        ex_new = ex + ex_dot * DT
        ey_new = ey + ey_dot * DT
        ez_new = ez + ez_dot * DT
        
        state = np.array([ex_new, ey_new, ez_new, ex_dot_new, ey_dot_new, ez_dot_new])
        
        # 记录
        data.append({
            'time': t,
            'traj': traj_type,
            'ref_x': get_ref(t, traj_type)[0],
            'ref_y': get_ref(t, traj_type)[1],
            'ref_z': get_ref(t, traj_type)[2],
            'pos_err_x': ex,
            'pos_err_y': ey,
            'pos_err_z': ez
        })
    
    return data

def main():
    all_data = []
    
    for traj in ['figure8', 'circle', 'square']:
        print(f"Simulating {traj} (period={PERIODS[traj]}s)...")
        traj_data = simulate_closed_loop_error(traj)
        all_data.extend(traj_data)
        
        # 统计
        errs = [d['pos_err_y'] for d in traj_data]
        print(f"  pos_err_y: max={max(errs):.4f}, min={min(errs):.4f}, rms={np.sqrt(np.mean(np.array(errs)**2)):.4f}")
    
    df = pd.DataFrame(all_data)
    df.to_csv('trajectory_error_data.csv', index=False)
    print(f"\nData saved to trajectory_error_data.csv")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()

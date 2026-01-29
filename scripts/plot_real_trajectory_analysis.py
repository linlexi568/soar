import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json

# --- 字体设置 ---
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

def plot_trajectory_analysis(csv_path, output_path='real_trajectory_analysis.png'):
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 获取轨迹类型
    traj_types = df['traj'].unique()
    
    # 创建画布: 3行 (Square, Circle, Figure8) x 3列 (XY轨迹, 误差曲线, 控制量)
    fig, axes = plt.subplots(len(traj_types), 3, figsize=(18, 5 * len(traj_types)))
    
    # 如果只有一行，axes是1D数组，转为2D
    if len(traj_types) == 1:
        axes = np.array([axes])
        
    for i, traj in enumerate(traj_types):
        traj_data = df[df['traj'] == traj]
        time = traj_data['time']
        
        # --- 1. XY 平面轨迹 ---
        ax_traj = axes[i, 0]
        ax_traj.plot(traj_data['ref_x'], traj_data['ref_y'], 'k--', label='Reference', linewidth=1.5)
        ax_traj.plot(traj_data['actual_x'], traj_data['actual_y'], 'r-', label='Actual (Smooth)', linewidth=1.5)
        ax_traj.set_title(f'{traj.capitalize()} Trajectory (XY Plane)', fontsize=14)
        ax_traj.set_xlabel('X Position (m)', fontsize=12)
        ax_traj.set_ylabel('Y Position (m)', fontsize=12)
        ax_traj.legend(fontsize=10)
        ax_traj.grid(True, linestyle=':', alpha=0.6)
        ax_traj.axis('equal')
        
        # --- 2. 跟踪误差 ---
        ax_err = axes[i, 1]
        ax_err.plot(time, traj_data['pos_err_x'], label='Error X', linewidth=1.2)
        ax_err.plot(time, traj_data['pos_err_y'], label='Error Y', linewidth=1.2)
        ax_err.plot(time, traj_data['pos_err_z'], label='Error Z', linewidth=1.2)
        
        # 计算 RMSE
        rmse_x = np.sqrt(np.mean(traj_data['pos_err_x']**2))
        rmse_y = np.sqrt(np.mean(traj_data['pos_err_y']**2))
        rmse_z = np.sqrt(np.mean(traj_data['pos_err_z']**2))
        
        ax_err.set_title(f'{traj.capitalize()} Tracking Error\n(RMSE: X={rmse_x:.3f}, Y={rmse_y:.3f}, Z={rmse_z:.3f})', fontsize=14)
        ax_err.set_xlabel('Time (s)', fontsize=12)
        ax_err.set_ylabel('Position Error (m)', fontsize=12)
        ax_err.legend(fontsize=10)
        ax_err.grid(True, linestyle=':', alpha=0.6)
        
        # --- 3. 控制输入 (Torque X/Y) ---
        ax_ctrl = axes[i, 2]
        ax_ctrl.plot(time, traj_data['u_tx'], label='Torque X', linewidth=1.2)
        ax_ctrl.plot(time, traj_data['u_ty'], label='Torque Y', linewidth=1.2)
        # ax_ctrl.plot(time, traj_data['u_tz'], label='Torque Z', linewidth=1.2)
        
        ax_ctrl.set_title(f'{traj.capitalize()} Control Inputs (Torque)', fontsize=14)
        ax_ctrl.set_xlabel('Time (s)', fontsize=12)
        ax_ctrl.set_ylabel('Normalized Torque', fontsize=12)
        ax_ctrl.legend(fontsize=10)
        ax_ctrl.grid(True, linestyle=':', alpha=0.6)
        ax_ctrl.set_ylim([-1.1, 1.1]) # 归一化力矩通常在 -1 到 1 之间

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    csv_file = 'trajectory_error_data.csv'
    if os.path.exists(csv_file):
        plot_trajectory_analysis(csv_file)
    else:
        print(f"Error: {csv_file} not found. Please run export_real_trajectory_data.py first.")

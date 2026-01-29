#!/usr/bin/env python3
"""
Soar DSL Controller for Benchmark Environment

Implements the optimal control rules from manual.md:
- Figure8: u_tx = ((-0.489 * smooth(pos_err_y, s=1.285)) + (1.062 * vel_y)) - (0.731 * ang_vel_x)
- Square:  u_tx = ((-0.600 * smooth(pos_err_y, s=0.1)) + (1.766 * vel_y)) - (0.773 * ang_vel_x)
- Circle:  u_tx = ((-2.104 * smooth(pos_err_y, s=0.296)) + (1.111 * vel_y)) - (0.727 * ang_vel_x)

Note: Square uses small s=0.1 to approximate sign (bang-bang) while maintaining Lipschitz continuity.
      As s→0, smooth(e,s) → sign(e), but smooth is differentiable everywhere.
"""
from __future__ import annotations
import math
from typing import Dict, Any
import numpy as np


class SoarController:
    """Soar nonlinear controller based on manual.md optimal parameters.
    
    All trajectories use smooth(e, s) = s * tanh(e/s) with different scale parameters:
    - figure8: s=1.285 (gentle saturation for smooth curves)
    - square:  s=0.1   (sharp saturation approximating bang-bang for corners)
    - circle:  s=0.296 (medium saturation)
    
    The smooth operator is Lipschitz continuous and differentiable everywhere,
    unlike sign which has a discontinuity at e=0.
    """
    
    # Control parameters from manual.md
    PARAMS = {
        'figure8': {
            'k_p': 0.489,
            'k_s': 1.285,  # smooth scale parameter
            'k_d': 1.062,
            'k_w': 0.731,
            'cost': 81.78,
            'nonlinear': 'smooth'
        },
        'square': {
            'k_p': 1.6434643124552815,
            'k_s': 0.7861879055222127,
            'k_d': 1.632741597362691,
            'k_w': 0.6294447587302711,
            'cost': 44.38,  
            'nonlinear': 'smooth'  
        },
        'circle': {
            'k_p': 2.104,
            'k_s': 0.296,
            'k_d': 1.111,
            'k_w': 0.727,
            'cost': 144.21,
            'nonlinear': 'smooth'
        }
    }
    
    def __init__(self,
                 trajectory: str = 'circle',
                 mass: float = 0.027,
                 g: float = 9.81,
                 thrust_clip: float = 2.0,
                 torque_clip: float = 0.5):
        """Initialize Soar controller.
        
        Args:
            trajectory: Trajectory type ('figure8', 'square', 'circle')
            mass: Quadrotor mass (kg)
            g: Gravitational acceleration (m/s^2)
            thrust_clip: Max thrust multiplier
            torque_clip: Max torque (Nm)
        """
        if trajectory not in self.PARAMS:
            raise ValueError(f"Trajectory '{trajectory}' not supported. Choose from {list(self.PARAMS.keys())}")
        
        self.trajectory = trajectory
        self.params = self.PARAMS[trajectory]
        self.mass = float(mass)
        self.g = float(g)
        # For Isaac Gym: u_fz = 0.65 is hover (not mass*g)
        # Environment applies scaling: fz * (mass*g/0.65)
        self.hover_thrust = 0.65
        self.thrust_clip = float(thrust_clip)
        self.torque_clip = float(torque_clip)
        self.dt = 1.0 / 48.0
        
        # Extract control gains
        self.k_p = self.params['k_p']
        self.k_d = self.params['k_d']
        self.k_w = self.params['k_w']
        self.k_s = self.params.get('k_s', 1.0)  # smooth scale (only for smooth trajectories)
        self.nonlinear = self.params['nonlinear']
        
        # Z-axis and yaw control (fixed for all trajectories)
        self.k_p_z = 1.0
        self.k_d_z = 0.5
        self.k_p_yaw = 4.0
        self.k_d_yaw = 0.8
        
    def set_dt(self, dt: float):
        """Set control timestep (for compatibility with evaluation framework)."""
        self.dt = float(dt)
    
    def reset(self):
        """Reset internal state (Soar is stateless, but included for API compatibility)."""
        pass
    
    def smooth(self, val: float, s: float) -> float:
        """Smooth operator: s * tanh(val/s)
        
        Properties:
        - Small errors: approximately linear (≈ val)
        - Large errors: saturates to ±s
        - Differentiable everywhere
        """
        return s * math.tanh(val / s)
    
    def sign(self, val: float) -> float:
        """Sign operator (bang-bang control).
        
        Returns:
            1.0 if val > 0
            -1.0 if val < 0
            0.0 if val == 0
        """
        if val > 0:
            return 1.0
        elif val < 0:
            return -1.0
        return 0.0
    def compute_nonlinear_term(self, error: float) -> float:
        """Compute the nonlinear proportional term based on trajectory type.
        
        All trajectories now use smooth operator with different s parameters:
        - Small s (e.g., 0.1 for square): approximates sign (bang-bang)
        - Medium s (e.g., 0.3 for circle): balanced saturation
        - Large s (e.g., 1.3 for figure8): gentle saturation
        
        Args:
            error: Position error (target - current)
        
        Returns:
            Nonlinear control output
        """
        # All trajectories use smooth (Lipschitz continuous)
        return self.smooth(error, self.k_s)
    
    def compute(self, pos, vel, quat, omega, target_pos, target_vel=None, target_acc=None):
        """Compute control action using Soar DSL control law.
        
        Args:
            pos: Current position [x, y, z] (numpy array or list)
            vel: Current velocity [vx, vy, vz] (numpy array or list)
            quat: Current quaternion [qx, qy, qz, qw] (numpy array or list)
            omega: Current angular velocity [wx, wy, wz] (numpy array or list)
            target_pos: Target position [x, y, z] (numpy array or list)
            target_vel: Target velocity (optional, unused by Soar)
            target_acc: Target acceleration (optional, unused by Soar)
        
        Returns:
            action: [fz, torque_x, torque_y, torque_z] (numpy array)
        """
        # Convert to numpy arrays
        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)
        omega = np.asarray(omega, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        
        # Position error
        pos_err = target_pos - pos
        
        # ============================================================================
        # Lateral Control (X-Y plane): Soar DSL control law
        # ============================================================================
        # u_tx = ((-k_p * nonlinear(pos_err_y)) + (k_d * vel_y)) - (k_w * ang_vel_x)
        # u_ty = ((k_p * nonlinear(pos_err_x)) - (k_d * vel_x)) - (k_w * ang_vel_y)
        
        u_tx = (
            (-self.k_p * self.compute_nonlinear_term(pos_err[1])) +  # Y error -> X torque
            (self.k_d * vel[1])                                       # Y velocity damping
        ) - (self.k_w * omega[0])                                     # X angular velocity damping
        
        u_ty = (
            (self.k_p * self.compute_nonlinear_term(pos_err[0])) -    # X error -> Y torque
            (self.k_d * vel[0])                                       # X velocity damping
        ) - (self.k_w * omega[1])                                     # Y angular velocity damping
        
        # ============================================================================
        # Vertical Control (Z-axis): Simple PD
        # ============================================================================
        # u_fz = k_p_z * pos_err_z - k_d_z * vel_z + hover_thrust
        u_fz = (self.k_p_z * pos_err[2]) - (self.k_d_z * vel[2]) + self.hover_thrust
        
        # ============================================================================
        # Yaw Control: PD on yaw angle
        # ============================================================================
        # Convert quaternion to yaw
        qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Target yaw is 0
        yaw_err = 0.0 - yaw
        # Wrap to [-pi, pi]
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi
        
        u_tz = self.k_p_yaw * yaw_err - self.k_d_yaw * omega[2]
        
        # ============================================================================
        # Clipping (safety limits)
        # ============================================================================
        u_fz = float(np.clip(u_fz, 0.0, 1.5))  # Isaac Gym normalized range
        u_tx = float(np.clip(u_tx, -1.0, 1.0))  # Normalized torque range
        u_ty = float(np.clip(u_ty, -1.0, 1.0))
        u_tz = float(np.clip(u_tz, -1.0, 1.0))
        
        return np.array([u_fz, u_tx, u_ty, u_tz], dtype=np.float32)


def get_soar_controller(trajectory: str, **kwargs) -> SoarController:
    """Factory function to create a Soar controller for a specific trajectory.
    
    Args:
        trajectory: Trajectory type ('figure8', 'square', 'circle')
        **kwargs: Additional parameters passed to SoarController.__init__
    
    Returns:
        SoarController instance
    
    Example:
        >>> ctrl = get_soar_controller('figure8')
        >>> action = ctrl.compute(pos, vel, quat, omega, target_pos)
    """
    return SoarController(trajectory=trajectory, **kwargs)


if __name__ == '__main__':
    # Demo: create controllers for each trajectory
    import sys
    
    print("Soar Controller Parameters from manual.md\n" + "="*60)
    
    for traj in ['figure8', 'square', 'circle']:
        ctrl = get_soar_controller(traj)
        params = ctrl.params
        print(f"\n{traj.upper()} Trajectory:")
        print(f"  Nonlinear type: {params['nonlinear']}")
        print(f"  k_p = {params['k_p']:.3f}")
        if 'k_s' in params:
            print(f"  k_s = {params['k_s']:.3f}")
        print(f"  k_d = {params['k_d']:.3f}")
        print(f"  k_w = {params['k_w']:.3f}")
        print(f"  Cost = {params['cost']:.2f}")
    
    print("\n" + "="*60)
    print("✓ Controller classes ready for benchmark evaluation")

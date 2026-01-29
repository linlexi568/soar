import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

# 1. Define Nonlinear Functions
def smooth(v, s=1.0):
    return s * np.tanh(v / s)

def sign_func(v):
    return np.sign(v)

# 2. Describing Function Analysis (Frequency Domain)
def describing_function_smooth(A, s=1.0):
    # For saturation-like function f(x) = s * tanh(x/s)
    # N(A) = (1 / (pi * A)) * integral_0^2pi f(A sin(theta)) * sin(theta) dtheta
    # Numerical integration for general shape
    if A < 1e-6: return 1.0 # Linear near 0
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = A * np.sin(theta)
    y = smooth(x, s)
    
    # Fundamental component coefficient b1
    b1 = (1/np.pi) * np.trapz(y * np.sin(theta), theta)
    return b1 / A

def describing_function_sign(A, M=1.0):
    # For relay f(x) = M * sign(x)
    # N(A) = 4M / (pi * A)
    if A < 1e-6: return np.inf # Infinite gain at 0
    return 4 * M / (np.pi * A)

def plot_describing_functions():
    amplitudes = np.logspace(-1, 1, 100)
    
    # Smooth with different s
    N_smooth_1 = [describing_function_smooth(a, s=1.0) for a in amplitudes]
    N_smooth_05 = [describing_function_smooth(a, s=0.5) for a in amplitudes]
    
    # Sign (Square controller uses sign)
    N_sign = [describing_function_sign(a, M=1.0) for a in amplitudes]
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(amplitudes, N_smooth_1, label='Smooth (s=1.0)')
    plt.semilogx(amplitudes, N_smooth_05, label='Smooth (s=0.5)')
    plt.semilogx(amplitudes, N_sign, label='Sign (Relay)')
    plt.title('Describing Function N(A) vs Amplitude')
    plt.xlabel('Input Amplitude A')
    plt.ylabel('Equivalent Gain N(A)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('nonlinear_frequency_analysis.png')
    print("Saved nonlinear_frequency_analysis.png")

# 3. Phase Plane Analysis (Time Domain)
# Model: Double integrator (mass = 1)
# x'' = u
# u = -Kp * f(error) - Kd * velocity
# error = 0 - x = -x
# velocity = x'
# u = -Kp * f(-x) - Kd * x'

def system_dynamics(state, t, controller_type, params):
    x, v = state
    
    # Error definitions (Target = 0)
    pos_err = -x
    vel = v
    
    if controller_type == 'smooth':
        # u = -Kp * smooth(err) - Kd * vel
        # Note: manual.md says u = -Kp * smooth(err) + ... 
        # Let's stick to a standard PD form for demonstration: u = Kp*err + Kd*err_dot
        # But manual.md has: u_tx = ((-0.559 * smooth(pos_err_y, s=2.258)) + (1.256 * vel_y))
        # Wait, manual.md: u_tx = -0.559 * smooth(pos_err) + 1.256 * vel
        # Usually feedback is negative. If pos_err = target - current, then u should be proportional to pos_err.
        # If Kp is negative (-0.559), then u ~ -(-0.559)*x = 0.559*x (positive feedback??)
        # Let's check signs in manual.md carefully.
        # u_tx = ((-0.559 * smooth(pos_err_y, s=2.258)) + (1.256 * vel_y))
        # If pos_err_y = 0 - y = -y.
        # u_tx = -0.559 * smooth(-y) + 1.256 * vy
        # smooth is odd function: smooth(-y) = -smooth(y)
        # u_tx = 0.559 * smooth(y) + 1.256 * vy.
        # This looks like positive feedback if it pushes y further away?
        # Unless u_tx is acceleration in negative direction?
        # Let's assume standard negative feedback for the demo: u = Kp * err - Kd * vel
        kp = params.get('kp', 1.0)
        kd = params.get('kd', 0.5)
        s = params.get('s', 1.0)
        u = kp * smooth(pos_err, s) - kd * vel
    
    elif controller_type == 'sign':
        kp = params.get('kp', 1.0)
        kd = params.get('kd', 0.5)
        u = kp * sign_func(pos_err) - kd * vel
        
    return [v, u]

def plot_phase_plane():
    t = np.linspace(0, 10, 1000)
    initial_states = [(2, 0), (1, 1), (-2, 0), (-1, -1)]
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Smooth Controller
    plt.subplot(1, 2, 1)
    for x0 in initial_states:
        sol = odeint(system_dynamics, x0, t, args=('smooth', {'kp': 1.0, 'kd': 0.5, 's': 1.0}))
        plt.plot(sol[:, 0], sol[:, 1], label=f'Start {x0}')
    plt.title('Phase Plane: Smooth Controller (s=1.0)')
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.grid(True)
    
    # Plot 2: Sign Controller
    plt.subplot(1, 2, 2)
    for x0 in initial_states:
        sol = odeint(system_dynamics, x0, t, args=('sign', {'kp': 1.0, 'kd': 0.5}))
        plt.plot(sol[:, 0], sol[:, 1], label=f'Start {x0}')
    plt.title('Phase Plane: Sign Controller (Sliding Mode)')
    plt.xlabel('Position x')
    plt.ylabel('Velocity v')
    plt.grid(True)
    
    plt.savefig('nonlinear_phase_plane.png')
    print("Saved nonlinear_phase_plane.png")

if __name__ == "__main__":
    plot_describing_functions()
    plot_phase_plane()

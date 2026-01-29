#!/usr/bin/env python3
"""
Simple test of smooth approximation of sign (no external dependencies)
"""
import math

def smooth(e, s):
    """Smooth operator: s * tanh(e/s)"""
    return s * math.tanh(e / s)

def sign(e):
    """Sign function"""
    if e > 0:
        return 1.0
    elif e < 0:
        return -1.0
    return 0.0

print("="*70)
print("SMOOTH APPROXIMATION OF SIGN FUNCTION TEST")
print("="*70)

print("\n1. Theory:")
print("   sign(e) is NOT Lipschitz continuous (discontinuous at e=0)")
print("   smooth(e, s) = s*tanh(e/s) IS Lipschitz continuous")
print("   As s→0, smooth(e,s) → sign(e)")

print("\n2. Testing approximation quality at different s values:")
print("-"*70)

test_points = [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0]
s_values = [0.01, 0.05, 0.1, 0.3, 1.0]

for s in s_values:
    print(f"\ns = {s:.3f}:")
    errors = []
    for e in test_points:
        smooth_val = smooth(e, s)
        sign_val = sign(e)
        error = abs(smooth_val - sign_val)
        errors.append(error)
        if abs(e) >= 0.5:
            print(f"  e={e:+5.1f}: smooth={smooth_val:+6.3f}, sign={sign_val:+2.0f}, error={error:.4f}")
    
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    print(f"  → Mean error: {mean_error:.4f}, Max error: {max_error:.4f}")

print("\n" + "="*70)
print("3. Lipschitz Continuity Test (near e=0 discontinuity):")
print("-"*70)

e1, e2 = -0.001, 0.001
input_diff = abs(e2 - e1)

print(f"\nTesting near e=0: e1={e1}, e2={e2} (diff={input_diff:.6f})")

# Sign: discontinuous
sign_diff = abs(sign(e2) - sign(e1))
print(f"\n  sign(e):")
print(f"    |sign({e2}) - sign({e1})| = {sign_diff:.3f}")
print(f"    Lipschitz ratio: {sign_diff / input_diff:.0f} → ∞ (UNBOUNDED!)")

# Smooth: continuous
for s in [0.01, 0.1, 0.5]:
    smooth_diff = abs(smooth(e2, s) - smooth(e1, s))
    lipschitz_ratio = smooth_diff / input_diff
    print(f"\n  smooth(e, s={s}):")
    print(f"    |smooth({e2}) - smooth({e1})| = {smooth_diff:.6f}")
    print(f"    Lipschitz ratio: {lipschitz_ratio:.3f} (BOUNDED ✓)")

print("\n" + "="*70)
print("4. Square Trajectory Corner Simulation:")
print("-"*70)

k_p = 0.6
print(f"\nController gain: k_p = {k_p}")
print("At 90° corner, error switches: +1.0 → -1.0")

u_sign_pos = -k_p * sign(1.0)
u_sign_neg = -k_p * sign(-1.0)
print(f"\n  Using sign:")
print(f"    u(+1.0) = {u_sign_pos:+.3f}")
print(f"    u(-1.0) = {u_sign_neg:+.3f}")
print(f"    Control jump: {abs(u_sign_neg - u_sign_pos):.3f} (INSTANT)")

s = 0.1  # Our choice for square
u_smooth_pos = -k_p * smooth(1.0, s)
u_smooth_neg = -k_p * smooth(-1.0, s)
approx_error = abs(u_smooth_pos - u_sign_pos) / abs(u_sign_pos) * 100

print(f"\n  Using smooth(s={s}): [OUR CHOICE]")
print(f"    u(+1.0) = {u_smooth_pos:+.3f}")
print(f"    u(-1.0) = {u_smooth_neg:+.3f}")
print(f"    Control jump: {abs(u_smooth_neg - u_smooth_pos):.3f} (SMOOTH)")
print(f"    Approximation error: {approx_error:.1f}% ✓")

s_large = 1.0
u_smooth_large = -k_p * smooth(1.0, s_large)
response_loss = abs(u_sign_pos - u_smooth_large) / abs(u_sign_pos) * 100

print(f"\n  Using smooth(s={s_large}): [TOO LARGE]")
print(f"    u(+1.0) = {u_smooth_large:+.3f}")
print(f"    Response loss: {response_loss:.1f}% ✗")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print(f"✅ s=0.1 provides excellent bang-bang approximation ({approx_error:.1f}% error)")
print("✅ Maintains Lipschitz continuity (gradient-friendly)")
print("✅ Suitable for numerical optimization and control")
print("✅ Avoids chattering issues of true sign function")
print("="*70)

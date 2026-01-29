"""
Academic benchmark presets and builders for Isaac Gym-based evaluation.

This module maps academically recognized quadrotor benchmarks (ETH RPG aggressive
trajectories, minimum-snap style classics, etc.) into the trajectory/disturbance
config format already used by SimulationTester and our Isaac Gym env.

Design goals:
- Pure Python, no external simulator dependency (remain in Isaac Gym stack)
- Use existing trajectory primitives (figure_8, circle, helix, square, ...)
- Provide standardized presets and disturbances with citations

References (for paper/README citations):
- Faessler et al., RA-L 2017/2018: Aggressive quadrotor flight and differential flatness
  http://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf
- Mellinger & Kumar, ICRA 2011: Minimum snap trajectory generation
- Safe-Control-Gym (Yuan et al., RA-L 2022) for standardized disturbance motifs

Note: We do not redistribute external datasets; we reproduce recognized benchmark
profiles through parameterized trajectories in our Isaac Gym environment, which
is a common practice for control benchmarking as long as references are provided.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional


def build_trajectory(name: str) -> Dict[str, Any]:
    """Build a trajectory dict by academic alias.

    We map academic aliases to existing internal trajectory types to keep full
    compatibility with SimulationTester.
    
    起点规范 (t=0):
    - Square:  [0, 0, 1]    (中心，先向 +y 移动)
    - Circle:  [R, 0, 1]    (圆周右侧)
    - Figure8: [0, 0, 1]    (中心)
    """
    # Core primitives (existing internal names)
    if name == 'figure8':
        # Figure8 起点在中心 [0, 0, 1]
        return { 'type': 'figure_8','initial_xyz': [0.0, 0.0, 1.0], 'params': {'A': 0.8,'B': 0.5,'period': 12}}
    if name == 'circle':
        # Circle 起点在圆周右侧 [R, 0, 1]，R=0.9
        return { 'type': 'circle','initial_xyz': [0.9, 0.0, 1.0], 'params': {'R': 0.9,'period': 10}}
    if name == 'helix':
        # Helix 起点在圆周右侧 [R, 0, z0]，R=0.8
        return { 'type': 'helix','initial_xyz': [0.8, 0.0, 0.6], 'params': {'R': 0.8,'period': 12,'v_z': 0.12}}
    if name == 'square':
        # Square 起点在中心 [0, 0, 1]
        return { 'type': 'square','initial_xyz': [0.0, 0.0, 1.0], 'params': {'side_len': 1.4,'period': 12,'corner_hold': 0.4}}
    if name == 'lemniscate3d':
        return { 'type': 'lemniscate3d','initial_xyz':[0,0,0.8], 'params': {'a':0.9,'period':14.0,'z_amp':0.22}}
    if name == 'spiral_in_out':
        return { 'type': 'spiral_in_out','initial_xyz':[0,0,0.7], 'params': {'R_in':0.9,'R_out':0.2,'period':14,'z_wave':0.15}}
    if name == 'zigzag3d':
        return { 'type': 'zigzag3d','initial_xyz':[0,0,0.7], 'params': {'amplitude':0.8,'segments':6,'z_inc':0.08,'period':14.0}}
    if name == 'stairs':
        return { 'type': 'stairs','initial_xyz':[0,0,0.6], 'params': {'levels':[0.6,0.9,1.2], 'segment_time':3.0}}
    if name == 'coupled_surface':
        return { 'type': 'coupled_surface','initial_xyz':[0,0,0.8], 'params': {'ax':0.9,'ay':0.7,'f1':1.0,'f2':2.0,'phase':1.0472,'z_amp':0.25,'surf_amp':0.15}}

    # Academic aliases mapped to primitives with more aggressive params
    if name == 'eth_min_snap_8':  # Minimum-snap style figure-8 (faster period)
        return { 'type': 'figure_8','initial_xyz': [0, 0, 1.0], 'params': {'A': 1.0,'B': 0.7,'period': 8.0}}
    if name == 'eth_aggressive_figure8':
        return { 'type': 'figure_8','initial_xyz': [0, 0, 0.9], 'params': {'A': 1.2,'B': 0.8,'period': 7.5}}
    if name == 'eth_fast_circle':
        return { 'type': 'circle','initial_xyz': [0, 0, 0.9], 'params': {'R': 1.1,'period': 7.0}}
    if name == 'eth_spiral_gate_like':  # proxy for racing corridor centerline
        return { 'type': 'spiral_in_out','initial_xyz':[0,0,0.7], 'params': {'R_in':1.0,'R_out':0.25,'period':10,'z_wave':0.20}}
    if name == 'eth_lemniscate':
        return { 'type': 'lemniscate3d','initial_xyz':[0,0,0.75], 'params': {'a':1.0,'period':12.0,'z_amp':0.30}}

    # Extreme variants
    if name == 'eth_extreme_zigzag':
        return { 'type': 'zigzag3d','initial_xyz': [0, 0, 0.6], 'params': {'amplitude': 1.1,'segments': 8,'z_inc': 0.12,'period': 10.0}}
    if name == 'eth_extreme_stairs':
        return { 'type': 'stairs','initial_xyz': [0, 0, 0.5], 'params': {'levels': [0.5, 0.8, 1.1, 1.4],'segment_time': 2.2}}
    if name == 'eth_extreme_coupled':
        return { 'type': 'coupled_surface','initial_xyz': [0, 0, 0.9], 'params': {'ax': 1.1,'ay': 0.9,'f1': 1.5,'f2': 3.0,'phase': 0.7,'z_amp': 0.35,'surf_amp': 0.22}}

    raise ValueError(f"Unknown academic trajectory alias: {name}")


def build_disturbances(preset: Optional[str]) -> List[Dict[str, Any]]:
    """Standardized disturbance sets inspired by Safe-Control-Gym and control literature."""
    if not preset:
        return []
    if preset == 'none':
        return []
    if preset == 'mild_wind':  # reproducible mild wind + single pulse (softer)
        return [
            {'type': 'SUSTAINED_WIND','info':'mild','start_time':3.0,'end_time':6.0,'force':[0.004,0.0,0.0]},
            {'type': 'PULSE','time':8.0,'force':[0.008,-0.004,0.0],'info':'pulse'}
        ]
    if preset == 'stress':  # multi-disturbance stress test (softer)
        return [
            {'type': 'SUSTAINED_WIND','info':'stress:steady_wind','start_time':2.5,'end_time':6.5,'force':[0.007,0.0,0.0]},
            {'type': 'GUSTY_WIND','info':'stress:gusty_wind','start_time':7.5,'end_time':11.5,'base_force':[0.0,-0.004,0.0],'gust_frequency':6.0,'gust_amplitude':0.006},
            {'type': 'MASS_CHANGE','info':'stress:mass_up','time':12.0,'mass_multiplier':1.08},
            {'type': 'PULSE','info':'stress:pulse','time':14.0,'force':[-0.008,0.008,0.0]}
        ]
    raise ValueError(f"Unknown disturbance preset: {preset}")


def build_preset(preset: str) -> List[str]:
    """Return academic preset trajectory aliases.

    Names are mapped to actual primitives in build_trajectory.
    """
    if preset == 'eth_rpg_core':
        return ['eth_min_snap_8', 'eth_fast_circle', 'eth_lemniscate', 'spiral_in_out']
    if preset == 'eth_rpg_challenge':
        return ['eth_aggressive_figure8', 'zigzag3d', 'coupled_surface', 'helix']
    if preset == 'eth_rpg_extreme':
        return ['eth_extreme_zigzag', 'eth_extreme_coupled', 'eth_extreme_stairs']
    if preset == 'academic_full':
        return (build_preset('eth_rpg_core') + build_preset('eth_rpg_challenge'))
    raise ValueError(f"Unknown academic preset: {preset}")


ACADEMIC_PRESET_NAMES = {
    'eth_rpg_core', 'eth_rpg_challenge', 'eth_rpg_extreme', 'academic_full'
}

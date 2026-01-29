"""
Dataset Loader for Standard Quadrotor Control Benchmarks
=========================================================
Loads and adapts verified trajectory datasets for MCTS training.

Currently supports:
- UZH-RPG Agile Autonomy Dataset (Science Robotics 2021)
- Custom format extensibility

Dataset can be downloaded from:
https://zenodo.org/record/5517791/files/agile_autonomy_dataset.tar.xz?download=1
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle


class AgileAutonomyDataset:
    """
    Loader for UZH-RPG Agile Autonomy Dataset.
    
    Paper: "Learning High-Speed Flight in the Wild"
    Dataset includes expert trajectories at ~7 m/s average speed
    with state observations, reference trajectories, and labels.
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to extracted agile_autonomy_dataset directory
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.data_root}\n"
                f"Please download from: "
                f"https://zenodo.org/record/5517791/files/agile_autonomy_dataset.tar.xz?download=1"
            )
        
        self.rollouts = self._discover_rollouts()
        print(f"Discovered {len(self.rollouts)} rollouts in dataset")
    
    def _discover_rollouts(self) -> List[Path]:
        """Find all rollout directories in dataset."""
        rollouts = []
        for split in ['train', 'test']:
            split_dir = self.data_root / split
            if split_dir.exists():
                rollouts.extend(sorted(split_dir.glob('rollout_*')))
        return rollouts
    
    def load_rollout(self, rollout_idx: int) -> Dict[str, np.ndarray]:
        """
        Load a single rollout's data.
        
        Returns:
            Dictionary with keys:
                - 'states': [T, 13] - position(3), orientation_quat(4), velocity(3), angular_vel(3)
                - 'references': [T, 3] - reference position at each timestep
                - 'actions': [T, 4] - expert actions (if available)
                - 'timestamps': [T] - time in seconds
        """
        rollout_path = self.rollouts[rollout_idx]
        
        data = {}
        
        # Load state trajectory (typical format: states.csv or states.npy)
        states_file = rollout_path / 'state.csv'
        if not states_file.exists():
            states_file = rollout_path / 'states.csv'
        if not states_file.exists():
            # Try numpy format
            states_file = rollout_path / 'states.npy'
            if states_file.exists():
                data['states'] = np.load(states_file)
        else:
            df = pd.read_csv(states_file)
            # Assume columns: t, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz
            state_cols = [c for c in df.columns if c != 't']
            data['states'] = df[state_cols].values
            data['timestamps'] = df['t'].values
        
        # Load reference trajectory
        ref_file = rollout_path / 'reference.csv'
        if not ref_file.exists():
            ref_file = rollout_path / 'ref.csv'
        if ref_file.exists():
            ref_df = pd.read_csv(ref_file)
            ref_cols = [c for c in ref_df.columns if c != 't']
            data['references'] = ref_df[ref_cols].values
        
        # Load expert actions if available
        action_file = rollout_path / 'expert_traj.csv'
        if action_file.exists():
            act_df = pd.read_csv(action_file)
            act_cols = [c for c in act_df.columns if c not in ['t', 'time']]
            data['actions'] = act_df[act_cols].values
        
        return data
    
    def get_trajectory_segments(
        self, 
        rollout_idx: int,
        segment_length: float = 2.0,
        overlap: float = 0.5
    ) -> List[Dict[str, np.ndarray]]:
        """
        Split a rollout into overlapping segments for training.
        
        Args:
            rollout_idx: Index of rollout to segment
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments (0-1)
            
        Returns:
            List of segment dictionaries, each with same keys as load_rollout
        """
        data = self.load_rollout(rollout_idx)
        
        if 'timestamps' not in data:
            # Assume 50Hz if no timestamps
            dt = 0.02
            T = len(data['states'])
            data['timestamps'] = np.arange(T) * dt
        
        timestamps = data['timestamps']
        total_time = timestamps[-1] - timestamps[0]
        
        segments = []
        stride = segment_length * (1.0 - overlap)
        
        start_time = timestamps[0]
        while start_time + segment_length <= timestamps[-1]:
            end_time = start_time + segment_length
            
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            
            segment = {
                'states': data['states'][mask],
                'timestamps': data['timestamps'][mask] - start_time,
            }
            
            if 'references' in data:
                segment['references'] = data['references'][mask]
            if 'actions' in data:
                segment['actions'] = data['actions'][mask]
            
            segments.append(segment)
            start_time += stride
        
        return segments


class TrajectoryAdapter:
    """
    Adapts external datasets to soar training format.
    Converts state trajectories into reference tracking problems.
    """
    
    def __init__(self, dt: float = 0.02):
        """
        Args:
            dt: Simulation timestep (default 50Hz)
        """
        self.dt = dt
    
    def to_reference_trajectory(
        self,
        states: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert state trajectory to reference format for controller training.
        
        Args:
            states: [T, 13] - full state trajectory
            timestamps: [T] - optional timestamps
            
        Returns:
            - ref_positions: [T, 3] - xyz reference
            - ref_velocities: [T, 3] - velocity reference  
            - ref_yaws: [T] - yaw reference
        """
        # Extract position (first 3 dims)
        ref_positions = states[:, :3]
        
        # Extract velocity (dims 7-10 typically)
        if states.shape[1] >= 10:
            ref_velocities = states[:, 7:10]
        else:
            # Compute from position derivative
            ref_velocities = np.gradient(ref_positions, axis=0) / self.dt
        
        # Extract or compute yaw from quaternion (dims 3-7)
        if states.shape[1] >= 7:
            quats = states[:, 3:7]  # [qw, qx, qy, qz] or [qx, qy, qz, qw]
            ref_yaws = self._quat_to_yaw(quats)
        else:
            ref_yaws = np.zeros(len(states))
        
        return ref_positions, ref_velocities, ref_yaws
    
    def _quat_to_yaw(self, quats: np.ndarray) -> np.ndarray:
        """Extract yaw angle from quaternion array."""
        # Assuming [qw, qx, qy, qz] format
        qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 
                         1.0 - 2.0 * (qy**2 + qz**2))
        return yaw
    
    def create_tracking_task(
        self,
        segment: Dict[str, np.ndarray],
        task_type: str = 'position_tracking'
    ) -> Dict:
        """
        Create a tracking task suitable for MCTS training.
        
        Args:
            segment: Data segment from dataset
            task_type: Type of tracking task
            
        Returns:
            Task dictionary compatible with train_soar.py
        """
        states = segment['states']
        ref_pos, ref_vel, ref_yaw = self.to_reference_trajectory(
            states, 
            segment.get('timestamps')
        )
        
        task = {
            'type': task_type,
            'duration': segment['timestamps'][-1] if 'timestamps' in segment else len(states) * self.dt,
            'initial_state': {
                'position': states[0, :3].tolist(),
                'velocity': states[0, 7:10].tolist() if states.shape[1] >= 10 else [0, 0, 0],
                'yaw': float(ref_yaw[0])
            },
            'reference_trajectory': {
                'positions': ref_pos.tolist(),
                'velocities': ref_vel.tolist(),
                'yaws': ref_yaw.tolist(),
                'dt': self.dt
            },
            'metadata': {
                'source': 'agile_autonomy',
                'avg_speed': float(np.linalg.norm(ref_vel, axis=1).mean()),
                'max_speed': float(np.linalg.norm(ref_vel, axis=1).max()),
            }
        }
        
        return task


def load_benchmark_dataset(
    dataset_name: str = 'agile_autonomy',
    data_root: Optional[str] = None,
    num_segments: int = 100,
    segment_length: float = 2.0
) -> List[Dict]:
    """
    Convenience function to load a benchmark dataset and convert to tasks.
    
    Args:
        dataset_name: Name of dataset ('agile_autonomy', etc.)
        data_root: Path to dataset root (if None, looks in standard locations)
        num_segments: Number of trajectory segments to extract
        segment_length: Length of each segment in seconds
        
    Returns:
        List of task dictionaries ready for MCTS training
    """
    if data_root is None:
        # Try standard locations
        possible_roots = [
            Path.home() / 'datasets' / 'agile_autonomy_dataset',
            Path('/data') / 'agile_autonomy_dataset',
            Path.cwd() / 'data' / 'agile_autonomy_dataset'
        ]
        for root in possible_roots:
            if root.exists():
                data_root = str(root)
                break
        
        if data_root is None:
            raise FileNotFoundError(
                f"Dataset not found in standard locations: {possible_roots}\n"
                f"Please download and extract to one of these locations, or specify data_root."
            )
    
    print(f"Loading {dataset_name} dataset from {data_root}")
    
    if dataset_name == 'agile_autonomy':
        dataset = AgileAutonomyDataset(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    adapter = TrajectoryAdapter()
    
    tasks = []
    segments_per_rollout = max(1, num_segments // len(dataset.rollouts))
    
    for rollout_idx in range(len(dataset.rollouts)):
        segments = dataset.get_trajectory_segments(
            rollout_idx, 
            segment_length=segment_length,
            overlap=0.5
        )
        
        # Sample evenly from this rollout's segments
        step = max(1, len(segments) // segments_per_rollout)
        sampled = segments[::step][:segments_per_rollout]
        
        for seg in sampled:
            task = adapter.create_tracking_task(seg)
            tasks.append(task)
            
            if len(tasks) >= num_segments:
                break
        
        if len(tasks) >= num_segments:
            break
    
    print(f"Created {len(tasks)} tracking tasks from dataset")
    print(f"Average speed: {np.mean([t['metadata']['avg_speed'] for t in tasks]):.2f} m/s")
    print(f"Max speed: {np.max([t['metadata']['max_speed'] for t in tasks]):.2f} m/s")
    
    return tasks


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and inspect benchmark dataset')
    parser.add_argument('--dataset', default='agile_autonomy', help='Dataset name')
    parser.add_argument('--data_root', default=None, help='Path to dataset root')
    parser.add_argument('--num_segments', type=int, default=10, help='Number of segments to load')
    parser.add_argument('--output', default=None, help='Output file to save tasks (JSON)')
    
    args = parser.parse_args()
    
    tasks = load_benchmark_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        num_segments=args.num_segments
    )
    
    print(f"\n=== Loaded {len(tasks)} tasks ===")
    print(f"First task sample:")
    print(json.dumps(tasks[0], indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"\nSaved tasks to {args.output}")

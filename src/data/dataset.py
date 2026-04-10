"""
dataset.py - Generate training data from heat equation simulations

This module converts simulator trajectories into (input, target) pairs
for training the neural network surrogate model.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
from .simulator import HeatEquationSimulator


def generate_trajectories(
    num_trajectories: int,
    num_steps: int = 100,
    grid_size: int = 64,
    alpha: float = 0.01,
    dt: float = 0.1,
    num_blobs: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Generate multiple heat equation trajectories with different initial conditions.
    
    Args:
        num_trajectories: Number of different simulations to run
        num_steps: Time steps per trajectory
        grid_size: Spatial grid size (grid_size × grid_size)
        alpha: Thermal diffusivity
        dt: Time step size
        num_blobs: Number of Gaussian blobs in each initial condition
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (num_trajectories, num_steps, grid_size, grid_size)
    """
    # Initialize storage for all trajectories
    trajectories = np.zeros((num_trajectories, num_steps, grid_size, grid_size))
    
    # Create simulator instance
    simulator = HeatEquationSimulator(grid_size=grid_size, alpha=alpha, dt=dt)
    
    # Generate each trajectory with a different random seed
    for i in range(num_trajectories):
        # Use seed + i to get different initial conditions for each trajectory
        trajectory_seed = seed + i
        trajectory = simulator.simulate(
            num_steps=num_steps,
            num_blobs=num_blobs,
            seed=trajectory_seed
        )
        trajectories[i] = trajectory
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_trajectories} trajectories")
    
    return trajectories


def create_training_data(
    trajectories: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: str = "data/processed"
) -> None:
    """
    Convert trajectories into (input, target) pairs and save as train/val/test splits.
    
    Args:
        trajectories: Array of shape (num_trajectories, num_steps, grid_size, grid_size)
        train_ratio: Fraction of data for training (default 0.8)
        val_ratio: Fraction of data for validation (default 0.1)
        output_dir: Directory to save processed data files
    """
    num_trajectories, num_steps, height, width = trajectories.shape
    
    # Extract (input, target) pairs
    # For each trajectory, we get (num_steps - 1) pairs
    # because the last state has no "next state" to predict
    inputs = []
    targets = []
    
    for i in range(num_trajectories):
        for t in range(num_steps - 1):
            # Input: state at time t
            inputs.append(trajectories[i, t])
            # Target: state at time t+1
            targets.append(trajectories[i, t + 1])
    
    # Convert lists to arrays
    inputs = np.array(inputs)   # Shape: (num_samples, height, width)
    targets = np.array(targets)  # Shape: (num_samples, height, width)
    
    # Add channel dimension for CNN (PyTorch expects: batch, channels, height, width)
    inputs = inputs[:, np.newaxis, :, :]    # Shape: (num_samples, 1, height, width)
    targets = targets[:, np.newaxis, :, :]  # Shape: (num_samples, 1, height, width)
    
    print(f"Total samples: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Shuffle data
    num_samples = len(inputs)
    indices = np.random.permutation(num_samples)
    inputs = inputs[indices]
    targets = targets[indices]
    
    # Split into train/val/test
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    
    val_inputs = inputs[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    test_inputs = inputs[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all splits
    np.save(output_path / "train_inputs.npy", train_inputs)
    np.save(output_path / "train_targets.npy", train_targets)
    np.save(output_path / "val_inputs.npy", val_inputs)
    np.save(output_path / "val_targets.npy", val_targets)
    np.save(output_path / "test_inputs.npy", test_inputs)
    np.save(output_path / "test_targets.npy", test_targets)
    
    print(f"\nData splits saved to {output_dir}/")
    print(f"Train: {len(train_inputs)} samples")
    print(f"Val: {len(val_inputs)} samples")
    print(f"Test: {len(test_inputs)} samples")


if __name__ == "__main__":
    """
    Generate dataset when run as a script.
    Start with 1000 trajectories for testing, scale up to 10000+ for real training.
    """
    print("Generating training data...")
    
    # Generate trajectories
    trajectories = generate_trajectories(
        num_trajectories=1000,  # Start small for testing
        num_steps=100,
        grid_size=64,
        alpha=0.01,
        dt=0.1,
        num_blobs=3,
        seed=42
    )
    
    # Create and save training splits
    create_training_data(
        trajectories=trajectories,
        train_ratio=0.8,
        val_ratio=0.1,
        output_dir="data/processed"
    )
    
    print("\nDataset generation complete!")
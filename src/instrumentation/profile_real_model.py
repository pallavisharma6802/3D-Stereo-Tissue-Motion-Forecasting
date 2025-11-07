"""
Real Model Profiler - Measures actual inference latency on Hamlyn data

This script profiles a real stereo depth model on sample Hamlyn frames
to measure realistic system delays for your hardware.

Usage:
    python profile_real_model.py --model raft_stereo --num_frames 50
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from instrumentation.latency_profiler import SystemLatencyProfiler


class DummyStereoModel:
    """
    Placeholder stereo model for profiling.
    Replace with actual RAFT-Stereo or other baseline once implemented.
    """
    
    def __init__(self, resolution=(256, 320)):
        """
        Args:
            resolution: (height, width) for processing
        """
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a dummy model - just convolutions to simulate real work
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, 3, padding=1),  # 6 channels for stereo pair
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3, padding=1),  # 1 channel for disparity
        ).to(self.device)
        
        print(f"Initialized dummy stereo model on {self.device}")
    
    def __call__(self, left_img, right_img):
        """
        Run inference on stereo pair.
        
        Args:
            left_img: Left image (H, W, 3) numpy array
            right_img: Right image (H, W, 3) numpy array
            
        Returns:
            Disparity map (H, W) numpy array
        """
        # Resize to model resolution
        left = cv2.resize(left_img, (self.resolution[1], self.resolution[0]))
        right = cv2.resize(right_img, (self.resolution[1], self.resolution[0]))
        
        # Convert to torch tensors (normalize to [0, 1])
        left_t = torch.from_numpy(left).permute(2, 0, 1).float() / 255.0
        right_t = torch.from_numpy(right).permute(2, 0, 1).float() / 255.0
        
        # Concatenate stereo pair
        stereo_input = torch.cat([left_t, right_t], dim=0).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            disparity = self.model(stereo_input)
        
        # Convert back to numpy
        disp_np = disparity.squeeze().cpu().numpy()
        
        # Resize to original resolution
        disp_resized = cv2.resize(disp_np, (left_img.shape[1], left_img.shape[0]))
        
        return disp_resized


def load_hamlyn_frames(hamlyn_dir: Path, 
                       sequence: str = 'rectified01',
                       num_frames: int = 50) -> tuple:
    """
    Load sample frames from Hamlyn dataset.
    
    Args:
        hamlyn_dir: Path to hamlyn_data directory
        sequence: Sequence name (e.g., 'rectified01')
        num_frames: Number of frames to load
        
    Returns:
        Tuple of (left_frames, right_frames) as lists of numpy arrays
    """
    seq_path = hamlyn_dir / sequence
    left_path = seq_path / 'image01'
    right_path = seq_path / 'image02'
    
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(
            f"Sequence {sequence} not found in {hamlyn_dir}. "
            f"Make sure Hamlyn data is extracted."
        )
    
    # Get sorted list of image files (try both jpg and png)
    left_files = sorted(left_path.glob('*.jpg'))[:num_frames]
    if len(left_files) == 0:
        left_files = sorted(left_path.glob('*.png'))[:num_frames]
    
    right_files = sorted(right_path.glob('*.jpg'))[:num_frames]
    if len(right_files) == 0:
        right_files = sorted(right_path.glob('*.png'))[:num_frames]
    
    if len(left_files) == 0:
        raise FileNotFoundError(f"No JPG or PNG images found in {left_path}")
    
    print(f"Loading {len(left_files)} frame pairs from {sequence}...")
    
    left_frames = []
    right_frames = []
    
    for left_f, right_f in zip(left_files, right_files):
        left_img = cv2.imread(str(left_f))
        right_img = cv2.imread(str(right_f))
        
        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_f} or {right_f}")
            continue
        
        left_frames.append(left_img)
        right_frames.append(right_img)
    
    print(f"Loaded {len(left_frames)} frame pairs")
    print(f"Image resolution: {left_frames[0].shape[:2]}")
    
    return left_frames, right_frames


def profile_stereo_model(model,
                        left_frames,
                        right_frames,
                        fps: float = 30.0) -> SystemLatencyProfiler:
    """
    Profile stereo model on real frames.
    
    Args:
        model: Stereo model with __call__ interface
        left_frames: List of left images
        right_frames: List of right images
        fps: Target frame rate for vsync simulation
        
    Returns:
        SystemLatencyProfiler with measurements
    """
    profiler = SystemLatencyProfiler(fps=fps, simulate_vsync=True)
    
    num_frames = len(left_frames)
    print(f"\nProfiling stereo model on {num_frames} frames...")
    
    for i, (left, right) in enumerate(zip(left_frames, right_frames)):
        # Start timing
        profiler.start_capture()
        
        # Run inference (this is the key measurement)
        disparity = model(left, right)
        profiler.mark_prediction()
        
        # Simulate rendering (in real system, this would be actual rendering)
        # For now, just normalize disparity for visualization
        disp_normalized = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
        disp_vis = (disp_normalized * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
        profiler.mark_render()
        
        # Mark display (vsync)
        profiler.mark_display()
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")
    
    return profiler


def main():
    parser = argparse.ArgumentParser(description='Profile stereo model latency')
    parser.add_argument('--hamlyn_dir', type=str, 
                       default='hamlyn_data',
                       help='Path to hamlyn_data directory')
    parser.add_argument('--sequence', type=str, 
                       default='rectified01',
                       help='Hamlyn sequence to use')
    parser.add_argument('--num_frames', type=int, 
                       default=50,
                       help='Number of frames to profile')
    parser.add_argument('--fps', type=float, 
                       default=30.0,
                       help='Target frame rate')
    parser.add_argument('--resolution', type=str,
                       default='256x320',
                       help='Processing resolution (HxW)')
    parser.add_argument('--output', type=str,
                       default='results/system_delay_profile.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Parse resolution
    h, w = map(int, args.resolution.split('x'))
    
    # Setup paths
    hamlyn_dir = Path(args.hamlyn_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        left_frames, right_frames = load_hamlyn_frames(
            hamlyn_dir, args.sequence, args.num_frames
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure hamlyn_data is extracted in the current directory.")
        return 1
    
    # Initialize model
    print(f"\nInitializing stereo model (resolution: {h}x{w})...")
    model = DummyStereoModel(resolution=(h, w))
    
    # Profile
    profiler = profile_stereo_model(
        model, left_frames, right_frames, fps=args.fps
    )
    
    # Print results
    print("\n")
    profiler.print_summary()
    
    # Save results
    profiler.save_results(str(output_path))
    print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

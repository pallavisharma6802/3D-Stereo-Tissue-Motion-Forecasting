"""
RAFT-Stereo Baseline Implementation

This module implements Baseline A: Optimized per-frame stereo depth estimation
using RAFT-Stereo-small for real-time performance.

RAFT-Stereo paper: https://arxiv.org/abs/2109.07547
Official implementation: https://github.com/princeton-vl/RAFT-Stereo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import sys

# Add RAFT-Stereo to path if available
RAFT_STEREO_PATH = Path(__file__).parent.parent.parent / 'third_party' / 'RAFT-Stereo'
if RAFT_STEREO_PATH.exists():
    sys.path.insert(0, str(RAFT_STEREO_PATH))
    sys.path.insert(0, str(RAFT_STEREO_PATH / 'core'))


class RAFTStereoBaseline(nn.Module):
    """
    RAFT-Stereo baseline for per-frame stereo depth estimation.
    
    This serves as Baseline A in our evaluation:
    - Single-frame stereo depth (no temporal context)
    - Optimized for real-time performance
    - Does NOT predict future frames
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 image_size: Tuple[int, int] = (256, 320),
                 mixed_precision: bool = True,
                 num_iters: int = 12):
        """
        Args:
            model_path: Path to pretrained RAFT-Stereo checkpoint
            image_size: (H, W) for processing
            mixed_precision: Use FP16 for speedup
            num_iters: Number of refinement iterations (trade-off speed/accuracy)
        """
        super().__init__()
        
        self.image_size = image_size
        self.mixed_precision = mixed_precision
        self.num_iters = num_iters
        
        # Try to load RAFT-Stereo
        try:
            from raft_stereo import RAFTStereo
            self.model = RAFTStereo(args=self._get_args())
            
            if model_path and Path(model_path).exists():
                self._load_checkpoint(model_path)
                print(f"Loaded RAFT-Stereo checkpoint from {model_path}")
            else:
                print("Warning: No checkpoint loaded, using random initialization")
                print("Download pretrained model from: https://github.com/princeton-vl/RAFT-Stereo")
        
        except ImportError:
            print("Warning: RAFT-Stereo not found, using dummy model")
            print("To use real RAFT-Stereo:")
            print("  1. Clone: git clone https://github.com/princeton-vl/RAFT-Stereo third_party/RAFT-Stereo")
            print("  2. Download weights to checkpoints/raft-stereo.pth")
            self.model = self._create_dummy_model()
        
        self.model.eval()
    
    def _get_args(self):
        """Create args namespace for RAFT-Stereo"""
        from argparse import Namespace
        args = Namespace()
        args.mixed_precision = self.mixed_precision
        args.valid_iters = self.num_iters
        args.hidden_dims = [128, 128, 128]
        args.corr_implementation = 'reg'  # Regular correlation
        args.corr_levels = 4
        args.corr_radius = 4
        args.n_downsample = 2
        args.slow_fast_gru = False
        args.n_gru_layers = 3
        args.shared_backbone = False
        args.context_norm = 'batch'  # batch normalization for context network
        return args
    
    def _load_checkpoint(self, path: str):
        """Load pretrained weights"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def _create_dummy_model(self) -> nn.Module:
        """Create a dummy model for testing without RAFT-Stereo"""
        class DummyRAFTStereo(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple CNN for demonstration
                self.conv1 = nn.Conv2d(6, 32, 7, padding=3)
                self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
                self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
                self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
            
            def forward(self, left, right, iters=12, test_mode=True):
                # Concatenate stereo pair
                x = torch.cat([left, right], dim=1)
                
                # Simple processing
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                disparity = self.conv4(x)
                
                # Return list of predictions (RAFT-Stereo returns multiple)
                return [disparity] * iters
        
        return DummyRAFTStereo()
    
    @torch.no_grad()
    def forward(self, 
                left: torch.Tensor, 
                right: torch.Tensor,
                return_all: bool = False) -> torch.Tensor:
        """
        Predict disparity for a stereo pair.
        
        Args:
            left: Left image (B, 3, H, W) in [0, 1]
            right: Right image (B, 3, H, W) in [0, 1]
            return_all: Return all refinement iterations
            
        Returns:
            disparity: Predicted disparity (B, H, W) or list if return_all=True
        """
        # Ensure correct input size
        orig_size = left.shape[-2:]
        if orig_size != self.image_size:
            left = F.interpolate(left, size=self.image_size, mode='bilinear', align_corners=False)
            right = F.interpolate(right, size=self.image_size, mode='bilinear', align_corners=False)
        
        # Run RAFT-Stereo
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            disparity_predictions = self.model(left, right, iters=self.num_iters, test_mode=True)
        
        # Get final prediction
        if return_all:
            results = disparity_predictions
        else:
            results = disparity_predictions[-1]
        
        # Resize back if needed
        if orig_size != self.image_size:
            if return_all:
                results = [F.interpolate(d, size=orig_size, mode='bilinear', align_corners=False) 
                          for d in results]
            else:
                results = F.interpolate(results, size=orig_size, mode='bilinear', align_corners=False)
        
        # Remove channel dimension
        if not return_all:
            results = results.squeeze(1)
        else:
            results = [d.squeeze(1) for d in results]
        
        return results
    
    def predict_depth(self,
                     left: torch.Tensor,
                     right: torch.Tensor,
                     focal_length: float,
                     baseline: float) -> torch.Tensor:
        """
        Predict depth from stereo pair.
        
        Args:
            left: Left image (B, 3, H, W)
            right: Right image (B, 3, H, W)
            focal_length: Camera focal length in pixels
            baseline: Stereo baseline in mm
            
        Returns:
            depth: Predicted depth in mm (B, H, W)
        """
        # Predict disparity
        disparity = self.forward(left, right)
        
        # Convert to depth: depth = (focal * baseline) / disparity
        depth = (focal_length * baseline) / (disparity + 1e-6)
        
        return depth
    
    def profile_latency(self,
                       left: torch.Tensor,
                       right: torch.Tensor,
                       num_warmup: int = 10,
                       num_runs: int = 100) -> Dict[str, float]:
        """
        Profile inference latency.
        
        Args:
            left: Left image (B, 3, H, W)
            right: Right image (B, 3, H, W)
            num_warmup: Warmup iterations
            num_runs: Measurement iterations
            
        Returns:
            Dictionary with timing statistics
        """
        device = left.device
        
        # Warmup
        print(f"Warming up for {num_warmup} iterations...")
        for _ in range(num_warmup):
            _ = self.forward(left, right)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Measure
        print(f"Profiling for {num_runs} iterations...")
        timings = []
        
        for i in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t_start = time.perf_counter()
            _ = self.forward(left, right)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t_end = time.perf_counter()
            timings.append((t_end - t_start) * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_runs}")
        
        timings = np.array(timings)
        
        stats = {
            'mean_ms': float(np.mean(timings)),
            'median_ms': float(np.median(timings)),
            'std_ms': float(np.std(timings)),
            'min_ms': float(np.min(timings)),
            'max_ms': float(np.max(timings)),
            'p95_ms': float(np.percentile(timings, 95)),
            'p99_ms': float(np.percentile(timings, 99)),
            'fps': float(1000.0 / np.median(timings)),
        }
        
        return stats


def test_baseline():
    """Test RAFT-Stereo baseline"""
    print("Testing RAFT-Stereo Baseline...")
    print("=" * 70)
    
    # Create model
    model = RAFTStereoBaseline(
        model_path=None,  # Will use dummy model
        image_size=(256, 320),
        mixed_precision=True,
        num_iters=12
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 1
    left = torch.rand(batch_size, 3, 256, 320).to(device)
    right = torch.rand(batch_size, 3, 256, 320).to(device)
    
    print(f"\nInput shape: {left.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    disparity = model(left, right)
    print(f"Output disparity shape: {disparity.shape}")
    print(f"Disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")
    
    # Test depth prediction
    print("\nTesting depth prediction...")
    focal_length = 383.19  # Hamlyn calibration
    baseline = 5.38  # mm
    depth = model.predict_depth(left, right, focal_length, baseline)
    print(f"Output depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.1f}, {depth.max():.1f}] mm")
    
    # Profile latency
    print("\nProfiling latency...")
    stats = model.profile_latency(left, right, num_warmup=5, num_runs=50)
    
    print(f"\nLatency Statistics:")
    print(f"  Mean:   {stats['mean_ms']:.2f} ms")
    print(f"  Median: {stats['median_ms']:.2f} ms")
    print(f"  Std:    {stats['std_ms']:.2f} ms")
    print(f"  Min:    {stats['min_ms']:.2f} ms")
    print(f"  Max:    {stats['max_ms']:.2f} ms")
    print(f"  P95:    {stats['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['p99_ms']:.2f} ms")
    print(f"  FPS:    {stats['fps']:.1f}")
    
    print("\n" + "=" * 70)
    print("Baseline test complete!")


if __name__ == '__main__':
    test_baseline()

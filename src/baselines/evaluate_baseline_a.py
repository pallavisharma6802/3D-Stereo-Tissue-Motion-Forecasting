"""
Evaluate RAFT-Stereo Baseline (Baseline A) on Hamlyn Dataset

This script evaluates per-frame stereo depth estimation without temporal context.
Measures accuracy and latency on the test set.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
from typing import Dict, List
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baselines import RAFTStereoBaseline
from src.data import create_dataloaders


class DepthMetrics:
    """Compute standard depth estimation metrics"""
    
    @staticmethod
    def compute(pred: torch.Tensor, 
                gt: torch.Tensor, 
                mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute depth metrics.
        
        Args:
            pred: Predicted depth (B, H, W) or (H, W)
            gt: Ground truth depth (B, H, W) or (H, W)
            mask: Valid mask (B, H, W) or (H, W)
            
        Returns:
            Dictionary with metrics:
                - epe: End-point error (mean absolute error)
                - rmse: Root mean squared error
                - abs_rel: Absolute relative error
                - sq_rel: Squared relative error
                - mae: Mean absolute error
                - mse: Mean squared error
        """
        # Apply mask
        pred_valid = pred[mask > 0]
        gt_valid = gt[mask > 0]
        
        if len(pred_valid) == 0:
            return {
                'epe': float('nan'),
                'rmse': float('nan'),
                'abs_rel': float('nan'),
                'sq_rel': float('nan'),
                'mae': float('nan'),
                'mse': float('nan'),
            }
        
        # Compute errors
        abs_error = torch.abs(pred_valid - gt_valid)
        sq_error = (pred_valid - gt_valid) ** 2
        
        # Metrics
        metrics = {
            'epe': float(abs_error.mean()),
            'mae': float(abs_error.mean()),
            'mse': float(sq_error.mean()),
            'rmse': float(torch.sqrt(sq_error.mean())),
            'abs_rel': float((abs_error / (gt_valid + 1e-6)).mean()),
            'sq_rel': float((sq_error / (gt_valid + 1e-6)).mean()),
        }
        
        return metrics


def evaluate_baseline_a(model: RAFTStereoBaseline,
                       test_loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       focal_length: float = 383.19,
                       baseline_mm: float = 5.38,
                       max_samples: int = None) -> Dict:
    """
    Evaluate RAFT-Stereo baseline on test set.
    
    Args:
        model: RAFTStereoBaseline model
        test_loader: DataLoader for test set
        device: Device to run on
        focal_length: Camera focal length (pixels)
        baseline_mm: Stereo baseline (mm)
        max_samples: Max samples to evaluate (None = all)
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    all_metrics = []
    latencies = []
    
    print("Evaluating Baseline A on test set...")
    print("=" * 70)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        
        for batch_idx, batch in enumerate(pbar):
            if max_samples and batch_idx >= max_samples:
                break
            
            # Unpack batch
            # Context: (B, 3, 3, H, W) - we only use current frame (index 2)
            context_left = batch['context_left'][:, 2, :, :, :].to(device)   # (B, 3, H, W)
            context_right = batch['context_right'][:, 2, :, :, :].to(device) # (B, 3, H, W)
            
            target_depth = batch['target_depth'].to(device)      # (B, H, W)
            valid_mask = batch['valid_mask'].to(device)          # (B, H, W)
            
            # Measure latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t_start = time.perf_counter()
            
            # Predict depth (at current time t, not future)
            pred_depth = model.predict_depth(
                context_left, 
                context_right,
                focal_length,
                baseline_mm
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            latencies.append(latency_ms)
            
            # Compute metrics for each sample in batch
            batch_size = pred_depth.shape[0]
            for i in range(batch_size):
                metrics = DepthMetrics.compute(
                    pred_depth[i],
                    target_depth[i],
                    valid_mask[i]
                )
                
                if not np.isnan(metrics['epe']):
                    all_metrics.append(metrics)
            
            # Update progress bar
            if len(all_metrics) > 0:
                avg_epe = np.mean([m['epe'] for m in all_metrics])
                avg_rmse = np.mean([m['rmse'] for m in all_metrics])
                avg_latency = np.mean(latencies)
                pbar.set_postfix({
                    'EPE': f'{avg_epe:.2f}mm',
                    'RMSE': f'{avg_rmse:.2f}mm',
                    'Latency': f'{avg_latency:.1f}ms'
                })
    
    # Aggregate results
    results = {
        'num_samples': len(all_metrics),
        'metrics': {
            'epe': float(np.mean([m['epe'] for m in all_metrics])),
            'rmse': float(np.mean([m['rmse'] for m in all_metrics])),
            'abs_rel': float(np.mean([m['abs_rel'] for m in all_metrics])),
            'sq_rel': float(np.mean([m['sq_rel'] for m in all_metrics])),
            'mae': float(np.mean([m['mae'] for m in all_metrics])),
            'mse': float(np.mean([m['mse'] for m in all_metrics])),
        },
        'latency': {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'fps': float(1000.0 / np.median(latencies)),
        }
    }
    
    return results


def print_results(results: Dict, horizon_ms: float):
    """Pretty print evaluation results"""
    print("\n" + "=" * 70)
    print(f"BASELINE A RESULTS (δ = {horizon_ms:.2f} ms)")
    print("=" * 70)
    
    print(f"\nSamples evaluated: {results['num_samples']}")
    
    print("\nDepth Accuracy Metrics:")
    print(f"  EPE (End-Point Error):    {results['metrics']['epe']:.2f} mm")
    print(f"  RMSE:                      {results['metrics']['rmse']:.2f} mm")
    print(f"  MAE (Mean Absolute):       {results['metrics']['mae']:.2f} mm")
    print(f"  AbsRel:                    {results['metrics']['abs_rel']:.4f}")
    print(f"  SqRel:                     {results['metrics']['sq_rel']:.4f}")
    
    print("\nInference Latency:")
    print(f"  Mean:   {results['latency']['mean_ms']:.2f} ms")
    print(f"  Median: {results['latency']['median_ms']:.2f} ms")
    print(f"  Std:    {results['latency']['std_ms']:.2f} ms")
    print(f"  P95:    {results['latency']['p95_ms']:.2f} ms")
    print(f"  P99:    {results['latency']['p99_ms']:.2f} ms")
    print(f"  FPS:    {results['latency']['fps']:.1f}")
    
    print("\n" + "=" * 70)


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAFT-Stereo Baseline A')
    parser.add_argument('--data_dir', type=str, default='hamlyn_data',
                       help='Path to Hamlyn dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to RAFT-Stereo checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 320],
                       help='Image size (H W)')
    parser.add_argument('--delta_ms', type=float, default=66.67,
                       help='Prediction horizon (for data loading)')
    parser.add_argument('--num_iters', type=int, default=12,
                       help='RAFT-Stereo refinement iterations')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to evaluate (for quick testing)')
    parser.add_argument('--output', type=str, default='results/baseline_a_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading test data...")
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        delta_ms=args.delta_ms,
        image_size=tuple(args.image_size)
    )
    test_loader = loaders['test']
    
    # Create model
    print("\nInitializing RAFT-Stereo baseline...")
    model = RAFTStereoBaseline(
        model_path=args.checkpoint,
        image_size=tuple(args.image_size),
        mixed_precision=True,
        num_iters=args.num_iters
    )
    model = model.to(device)
    
    # Evaluate
    results = evaluate_baseline_a(
        model=model,
        test_loader=test_loader,
        device=device,
        max_samples=args.max_samples
    )
    
    # Print results
    print_results(results, args.delta_ms)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

"""
System Delay Measurement (η) - Step 1 of Pipeline

This module measures the end-to-end latency of the surgical vision system:
    T_cap → T_pred → T_render → T_vsync
    
Where:
    - T_cap: Frame capture timestamp
    - T_pred: Prediction/inference completion time
    - T_render: Rendering completion time  
    - T_vsync: Display vsync (when result actually appears on screen)
    
The system delay η = T_vsync - T_cap represents how "outdated" our 
predictions are by the time they're displayed.

Usage:
    profiler = SystemLatencyProfiler()
    
    # Measure over multiple frames
    for frame in frames:
        profiler.start_capture()
        
        # Your stereo depth inference here
        depth = model(frame)
        profiler.mark_prediction()
        
        # Render the output
        render_output(depth)
        profiler.mark_render()
        
        # Simulate or measure actual display timing
        profiler.mark_display()
    
    # Get results
    eta = profiler.compute_system_delay()
    deltas = profiler.get_ablation_horizons()
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import json


@dataclass
class TimingMeasurement:
    """Single frame timing measurement"""
    t_cap: float      # Capture timestamp
    t_pred: float     # Prediction complete
    t_render: float   # Rendering complete
    t_vsync: float    # Display vsync
    
    @property
    def total_latency(self) -> float:
        """Total system delay (η) for this frame"""
        return self.t_vsync - self.t_cap
    
    @property
    def inference_time(self) -> float:
        """Pure inference time"""
        return self.t_pred - self.t_cap
    
    @property
    def render_time(self) -> float:
        """Rendering time"""
        return self.t_render - self.t_pred
    
    @property
    def display_time(self) -> float:
        """Display pipeline time"""
        return self.t_vsync - self.t_render


class SystemLatencyProfiler:
    """
    Profiles system latency by measuring timing at each stage.
    
    For real hardware: Hook into actual vsync callbacks
    For simulation: Estimates display timing based on refresh rate
    """
    
    def __init__(self, 
                 fps: float = 30.0,
                 simulate_vsync: bool = True,
                 vsync_callback: Optional[Callable[[], float]] = None):
        """
        Args:
            fps: Expected frame rate (for vsync simulation)
            simulate_vsync: If True, simulates vsync timing
            vsync_callback: Optional callback to get real vsync timestamps
        """
        self.fps = fps
        self.frame_period = 1.0 / fps  # Period between frames in seconds
        self.simulate_vsync = simulate_vsync
        self.vsync_callback = vsync_callback
        
        # Measurement storage
        self.measurements: List[TimingMeasurement] = []
        self.current_measurement: Optional[Dict[str, float]] = None
        
    def start_capture(self) -> None:
        """Mark the start of frame capture"""
        self.current_measurement = {
            't_cap': time.perf_counter()
        }
    
    def mark_prediction(self) -> None:
        """Mark completion of model prediction/inference"""
        if self.current_measurement is None:
            raise RuntimeError("Must call start_capture() first")
        self.current_measurement['t_pred'] = time.perf_counter()
    
    def mark_render(self) -> None:
        """Mark completion of rendering"""
        if self.current_measurement is None:
            raise RuntimeError("Must call start_capture() first")
        self.current_measurement['t_render'] = time.perf_counter()
    
    def mark_display(self) -> None:
        """
        Mark when frame is displayed (vsync).
        
        If simulate_vsync=True, estimates next vsync based on refresh rate.
        Otherwise uses vsync_callback if provided.
        """
        if self.current_measurement is None:
            raise RuntimeError("Must call start_capture() first")
        
        current_time = time.perf_counter()
        
        if self.simulate_vsync:
            # Simulate vsync: round up to next frame boundary
            t_cap = self.current_measurement['t_cap']
            elapsed = current_time - t_cap
            
            # How many frame periods have passed?
            frames_passed = np.ceil(elapsed / self.frame_period)
            
            # Next vsync occurs at this boundary
            t_vsync = t_cap + frames_passed * self.frame_period
            
        elif self.vsync_callback is not None:
            # Use real hardware vsync callback
            t_vsync = self.vsync_callback()
            
        else:
            # No simulation, no callback - just use current time
            # This underestimates latency but useful for debugging
            t_vsync = current_time
        
        self.current_measurement['t_vsync'] = t_vsync
        
        # Store completed measurement
        measurement = TimingMeasurement(**self.current_measurement)
        self.measurements.append(measurement)
        
        # Reset for next frame
        self.current_measurement = None
    
    def compute_system_delay(self) -> float:
        """
        Compute system delay η as median of all measurements.
        
        We use median instead of mean because it's more robust to outliers
        (e.g., occasional GC pauses, OS scheduling hiccups).
        
        Returns:
            η (eta) in milliseconds
        """
        if not self.measurements:
            raise RuntimeError("No measurements recorded yet")
        
        latencies = [m.total_latency for m in self.measurements]
        eta_seconds = np.median(latencies)
        eta_ms = float(eta_seconds * 1000)  # Convert to milliseconds
        
        return eta_ms
    
    def get_ablation_horizons(self, delta_offset: float = 33.0) -> Dict[str, float]:
        """
        Compute ablation test horizons: δ ∈ {η - 33ms, η, η + 33ms}
        
        Args:
            delta_offset: Offset in milliseconds (default 33ms ≈ 1 frame at 30fps)
            
        Returns:
            Dictionary with 'short', 'medium', 'long' horizon values in ms
        """
        eta = self.compute_system_delay()
        
        return {
            'short': eta - delta_offset,   # η - 33ms
            'medium': eta,                  # η
            'long': eta + delta_offset      # η + 33ms
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get detailed timing statistics across all measurements.
        
        Returns:
            Dictionary with mean, median, std, min, max for each timing stage
        """
        if not self.measurements:
            raise RuntimeError("No measurements recorded yet")
        
        total_latencies = [m.total_latency * 1000 for m in self.measurements]
        inference_times = [m.inference_time * 1000 for m in self.measurements]
        render_times = [m.render_time * 1000 for m in self.measurements]
        display_times = [m.display_time * 1000 for m in self.measurements]
        
        def compute_stats(values: List[float], name: str) -> Dict[str, float]:
            return {
                f'{name}_mean': float(np.mean(values)),
                f'{name}_median': float(np.median(values)),
                f'{name}_std': float(np.std(values)),
                f'{name}_min': float(np.min(values)),
                f'{name}_max': float(np.max(values)),
            }
        
        stats = {}
        stats.update(compute_stats(total_latencies, 'total_latency'))
        stats.update(compute_stats(inference_times, 'inference'))
        stats.update(compute_stats(render_times, 'render'))
        stats.update(compute_stats(display_times, 'display'))
        
        return stats
    
    def save_results(self, filepath: str) -> None:
        """Save profiling results to JSON file"""
        results = {
            'system_delay_ms': self.compute_system_delay(),
            'ablation_horizons': self.get_ablation_horizons(),
            'statistics': self.get_statistics(),
            'num_measurements': len(self.measurements),
            'fps': self.fps,
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def print_summary(self) -> None:
        """Print human-readable summary of measurements"""
        eta = self.compute_system_delay()
        horizons = self.get_ablation_horizons()
        stats = self.get_statistics()
        
        print("=" * 70)
        print("System Latency Profiling Results")
        print("=" * 70)
        print(f"\nMeasurements collected: {len(self.measurements)}")
        print(f"Frame rate: {self.fps} FPS ({self.frame_period*1000:.2f} ms/frame)")
        print(f"\nSystem Delay (η): {eta:.2f} ms")
        print(f"  - Inference:     {stats['inference_median']:.2f} ms")
        print(f"  - Rendering:     {stats['render_median']:.2f} ms")
        print(f"  - Display:       {stats['display_median']:.2f} ms")
        print(f"\nAblation Horizons (δ):")
        print(f"  - Short  (η-33): {horizons['short']:.2f} ms")
        print(f"  - Medium (η):    {horizons['medium']:.2f} ms")
        print(f"  - Long  (η+33):  {horizons['long']:.2f} ms")
        print("\nLatency Statistics:")
        print(f"  - Mean:   {stats['total_latency_mean']:.2f} ms")
        print(f"  - Median: {stats['total_latency_median']:.2f} ms")
        print(f"  - Std:    {stats['total_latency_std']:.2f} ms")
        print(f"  - Range:  [{stats['total_latency_min']:.2f}, {stats['total_latency_max']:.2f}] ms")
        print("=" * 70)


def profile_dummy_model(num_frames: int = 100, 
                       inference_delay: float = 0.050,
                       render_delay: float = 0.005,
                       fps: float = 30.0) -> SystemLatencyProfiler:
    """
    Profile a dummy model to demonstrate usage.
    
    This simulates a typical lightweight stereo model that takes ~50ms 
    for inference and ~5ms for rendering, running at 30 FPS.
    
    Args:
        num_frames: Number of frames to profile
        inference_delay: Simulated inference time in seconds
        render_delay: Simulated rendering time in seconds
        fps: Target frame rate
        
    Returns:
        SystemLatencyProfiler with measurements
    """
    profiler = SystemLatencyProfiler(fps=fps, simulate_vsync=True)
    
    print(f"Profiling dummy model for {num_frames} frames...")
    print(f"  Simulated inference: {inference_delay*1000:.1f} ms")
    print(f"  Simulated rendering: {render_delay*1000:.1f} ms")
    print(f"  Target FPS: {fps}")
    
    for i in range(num_frames):
        profiler.start_capture()
        
        # Simulate inference
        time.sleep(inference_delay)
        profiler.mark_prediction()
        
        # Simulate rendering
        time.sleep(render_delay)
        profiler.mark_render()
        
        # Mark display (vsync simulation)
        profiler.mark_display()
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_frames} frames...")
    
    return profiler


if __name__ == "__main__":
    # Example usage: Profile a dummy model
    profiler = profile_dummy_model(num_frames=100)
    
    # Print results
    profiler.print_summary()
    
    # Save to file
    profiler.save_results('system_delay_profile.json')
    print("\nResults saved to 'system_delay_profile.json'")

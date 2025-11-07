# Step 1: System Delay Measurement (η)

## Overview

This directory contains instrumentation code to measure end-to-end system latency in the surgical vision pipeline. The system delay η represents how "outdated" our predictions are by the time they're displayed.

## What We Measure

The timing chain through the system:

```
T_cap → T_pred → T_render → T_vsync
```

Where:
- **T_cap**: Frame capture timestamp
- **T_pred**: Prediction/inference completion time
- **T_render**: Rendering completion time
- **T_vsync**: Display vsync (when result actually appears on screen)

**System Delay**: η = T_vsync - T_cap

## Why This Matters

If η = 100ms, our depth prediction shows where tissue was 100ms ago, not where it is now. By measuring η, we can:

1. Understand how much our system lags behind reality
2. Determine optimal prediction horizons for future depth forecasting
3. Set ablation test points at η±33ms (roughly ±1 frame at 30 FPS)

## Results from Our System

Running on Hamlyn dataset (rectified01, 30 frames):

```
System Delay (η): 66.67 ms
  - Inference:     47.22 ms
  - Rendering:     0.54 ms
  - Display:       19.10 ms

Ablation Horizons (δ):
  - Short  (η-33): 33.67 ms
  - Medium (η):    66.67 ms
  - Long  (η+33):  99.67 ms
```

This means we need to predict approximately 67ms into the future to compensate for system lag.

## Usage

### Basic Profiling with Dummy Model

```python
from src.instrumentation import SystemLatencyProfiler

profiler = SystemLatencyProfiler(fps=30.0)

for frame in frames:
    profiler.start_capture()
    
    # Your stereo inference
    depth = model(frame)
    profiler.mark_prediction()
    
    # Render output
    render(depth)
    profiler.mark_render()
    
    # Mark display
    profiler.mark_display()

# Get results
eta = profiler.compute_system_delay()
horizons = profiler.get_ablation_horizons()
profiler.print_summary()
```

### Profile Real Stereo Model on Hamlyn Data

```bash
# Activate environment
source venv/bin/activate

# Profile with default settings (30 frames, rectified01)
python src/instrumentation/profile_real_model.py

# Custom settings
python src/instrumentation/profile_real_model.py \
    --sequence rectified04 \
    --num_frames 100 \
    --resolution 512x640 \
    --fps 30 \
    --output results/latency_profile_r04.json
```

## Files

- **`latency_profiler.py`**: Core SystemLatencyProfiler class
  - Measures timing at each pipeline stage
  - Computes η as median of measurements
  - Calculates ablation horizons (η±33ms)
  - Provides detailed statistics

- **`profile_real_model.py`**: Real model profiling script
  - Loads Hamlyn stereo frames
  - Runs dummy stereo model (replace with RAFT-Stereo later)
  - Measures actual inference latency on your hardware
  - Saves results to JSON

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "system_delay_ms": 66.67,
  "ablation_horizons": {
    "short": 33.67,
    "medium": 66.67,
    "long": 99.67
  },
  "statistics": {
    "total_latency_mean": 68.89,
    "total_latency_median": 66.67,
    "total_latency_std": 8.31,
    "inference_mean": 47.22,
    ...
  },
  "num_measurements": 30,
  "fps": 30.0
}
```

## Next Steps

Once we have η:

1. **Step 2**: Use these ablation horizons for data preparation
2. **Step 3-5**: Build models that predict depth at t+δ where δ ∈ {η-33, η, η+33}
3. **Step 6**: Evaluate with latency-matched protocol using measured η

## Notes

- Currently uses simulated vsync based on FPS
- For real hardware: Replace `simulate_vsync=True` with actual vsync callback
- Dummy stereo model is placeholder - will be replaced with RAFT-Stereo baseline
- Median is used instead of mean to be robust to outliers (GC pauses, OS scheduling)

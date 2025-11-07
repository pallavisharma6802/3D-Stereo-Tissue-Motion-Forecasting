# Step 3 Progress: Baseline A - RAFT-Stereo

**Status**: 🚧 IN PROGRESS

## Summary

Implementing Baseline A: Per-frame stereo depth estimation using RAFT-Stereo. This baseline establishes real-time stereo depth performance **without temporal context** as a comparison point for our temporal forecasting model.

## Objective

Establish a real-time stereo depth baseline that:
- Predicts depth at current time `t` (not future)
- Uses only current stereo pair (no temporal context)
- Provides latency and accuracy benchmarks
- Serves as lower bound for comparison

## Implementation Progress

### ✅ Completed Tasks

1. **RAFT-Stereo Integration**
   - ✅ Cloned RAFT-Stereo repository to `third_party/RAFT-Stereo`
   - ✅ Downloaded pretrained models (192MB)
   - ✅ Installed dependencies: `opt_einsum`, `einops`, `timm`
   - ✅ Created wrapper class `RAFTStereoBaseline`

2. **Baseline Infrastructure**
   - ✅ Created `src/baselines/` directory
   - ✅ Implemented `RAFTStereoBaseline` class with:
     - Model loading and checkpoint support
     - Forward pass for disparity prediction
     - Depth conversion using calibration
     - Latency profiling utilities
     - Fallback dummy model for testing
   - ✅ Created evaluation script `evaluate_baseline_a.py`

3. **Testing**
   - ✅ Verified RAFT-Stereo imports correctly
   - ✅ Tested with dummy model (14 FPS on CPU)
   - ✅ Loaded pretrained Middlebury checkpoint
   - ✅ Verified forward pass works
   - ✅ Discovered full dataset: 22 sequences, 92,408 total samples!

### 🚧 In Progress

1. **Evaluation on Test Set**
   - Started evaluation but very slow on CPU (~1.2 sec/sample)
   - Need to either:
     - Run on GPU for faster evaluation
     - Use smaller model (raftstereo-realtime.pth)
     - Evaluate on subset for quick validation
   - Interrupted after 18/13,862 samples
   - Preliminary results: EPE ~180mm, RMSE ~185mm (on first 18 samples)

### ⏳ Pending Tasks

1. **Complete Evaluation**
   - [ ] Run full evaluation on test set
   - [ ] Test at all three horizons (δ ∈ {33.67, 66.67, 99.67} ms)
   - [ ] Generate accuracy metrics (EPE, RMSE, AbsRel)
   - [ ] Profile inference latency
   
2. **Optimization (Optional)**
   - [ ] Test with `raftstereo-realtime.pth` (faster model)
   - [ ] Reduce refinement iterations (8 → 4 or 6)
   - [ ] Lower resolution if needed (256×320 → 128×160)
   - [ ] Try mixed precision on GPU

3. **Documentation**
   - [ ] Create results visualization
   - [ ] Compare with ground truth samples
   - [ ] Document latency vs accuracy trade-offs
   - [ ] Complete STEP3_COMPLETE.md

## Files Created

```
src/baselines/
├── __init__.py                     # Package exports
├── raft_stereo_baseline.py         # RAFT-Stereo wrapper (328 lines)
└── evaluate_baseline_a.py          # Evaluation script (292 lines)

third_party/
└── RAFT-Stereo/                    # Official RAFT-Stereo repo
    ├── core/
    │   ├── raft_stereo.py
    │   ├── update.py
    │   └── ...
    └── raftstereo-middlebury.pth   # Pretrained model (43MB)

checkpoints/
└── raftstereo-middlebury.pth       # Copied pretrained model

scripts/
└── setup_raft_stereo.sh            # Setup script for RAFT-Stereo
```

**Total**: ~620 lines of baseline code

## Key Implementation Details

### RAFTStereoBaseline Class

```python
class RAFTStereoBaseline(nn.Module):
    """RAFT-Stereo baseline for per-frame stereo depth"""
    
    def __init__(self, model_path, image_size=(256, 320), 
                 mixed_precision=True, num_iters=12):
        # Load pretrained RAFT-Stereo
        # Configure refinement iterations
        # Support mixed precision
    
    def forward(self, left, right):
        # Predict disparity from stereo pair
        # Returns: (B, H, W) disparity map
    
    def predict_depth(self, left, right, focal_length, baseline):
        # Convert disparity to depth
        # depth = (focal * baseline) / disparity
    
    def profile_latency(self, left, right, num_runs=100):
        # Measure inference time
        # Returns: mean, median, p95, p99, FPS
```

### Evaluation Metrics

```python
class DepthMetrics:
    """Standard depth estimation metrics"""
    
    @staticmethod
    def compute(pred, gt, mask):
        return {
            'epe': End-point error (MAE)
            'rmse': Root mean squared error
            'abs_rel': Absolute relative error
            'sq_rel': Squared relative error
        }
```

## Dataset Statistics (Updated!)

### Full Hamlyn Dataset
- **Total sequences**: 22 (rectified01-27, some numbers skipped)
- **Total frames**: ~98,594 frames across all sequences
- **Total samples**: 92,408 (after edge removal)
  - Train: 64,685 samples (70%)
  - Val: 13,861 samples (15%)
  - Test: 13,862 samples (15%)

### Largest Sequences
- rectified08: 14,393 frames
- rectified23: 11,255 frames
- rectified19: 10,622 frames
- rectified21: 5,367 frames
- rectified18: 5,596 frames

### Baseline Variation
- 5.20 mm: rectified14-24 (9 sequences)
- 5.38 mm: rectified01, 06, 08, 09 (4 sequences)
- 5.49 mm: rectified04, 05 (2 sequences)
- 5.52 mm: rectified11, 12 (2 sequences)
- 5.60 mm: rectified25-27 (3 sequences)

## Performance Benchmarks

### Dummy Model (CPU)
- **Latency**: 71.15ms median (14.1 FPS)
- **Model**: Simple 4-layer CNN
- **Note**: For testing infrastructure only

### RAFT-Stereo (CPU, 8 iters)
- **Latency**: ~1,200ms per sample (0.8 FPS)
- **Model**: Full RAFT-Stereo with 8 refinement iterations
- **Note**: Too slow for real-time, need GPU

### Preliminary Results (First 18 Test Samples)
- **EPE**: 179.83 mm
- **RMSE**: 184.61 mm
- **Latency**: 1,170.5 ms
- **Note**: Not representative, needs full evaluation

## Known Issues

### Performance
1. **CPU is too slow**: ~1.2 sec/sample
   - Solution: Run on GPU or use faster model variant
   
2. **Full evaluation will take ~4.6 hours**: 13,862 samples × 1.2s
   - Solution: Evaluate on subset or use GPU

### Technical
1. **Mixed precision warnings**: CUDA autocast on CPU
   - Solution: Disable mixed precision for CPU

2. **Meshgrid indexing warning**: PyTorch deprecation
   - Solution: Harmless, can be ignored

## Next Steps

### Immediate (To Complete Step 3)
1. **Option A: Quick Validation**
   - Evaluate on 100-200 test samples
   - Get approximate metrics
   - Document limitations

2. **Option B: Full Evaluation** 
   - Run overnight on CPU
   - Or find GPU access
   - Complete metrics for all samples

### After Step 3 Completion
- **Step 4**: Motion extrapolation baseline (Baseline B)
  - Optical flow computation
  - Depth warping to predict future
  - Compare with Baseline A

- **Step 5**: Our temporal model
  - Use 3-frame context
  - Direct future depth prediction
  - Compare with both baselines

## Key Decisions Made

1. **Model Choice**: Using `raftstereo-middlebury.pth`
   - Best generalization for in-the-wild images
   - 43MB model size
   - 12 refinement iterations (can reduce for speed)

2. **Resolution**: 256×320
   - Balance between speed and accuracy
   - Matches data preprocessing pipeline
   - Can reduce if needed

3. **Evaluation Protocol**: Single-frame depth at time `t`
   - No temporal context (pure stereo baseline)
   - Predicts current depth, not future
   - Fair comparison point for temporal methods

## References

- **RAFT-Stereo Paper**: [https://arxiv.org/abs/2109.07547](https://arxiv.org/abs/2109.07547)
- **Official Repo**: [https://github.com/princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
- **Pretrained Models**: Downloaded from Google Drive (192MB)

---

**Date**: November 7, 2025
**Step**: 3/7 in pipeline  
**Time Invested**: ~1.5 hours
**Status**: Infrastructure complete, evaluation pending 🔄

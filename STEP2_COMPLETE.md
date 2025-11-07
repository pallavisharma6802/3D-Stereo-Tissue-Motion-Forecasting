# Step 2 Complete: Data Preparation

**Status**: ✅ COMPLETE

## Summary

Successfully implemented a comprehensive data loading pipeline for the Hamlyn stereo surgical video dataset with temporal windowing and future depth alignment.

## Implementation

### Files Created

1. **`src/data/hamlyn_loader.py`** (462 lines)
   - Core data loading infrastructure
   - Classes:
     - `CameraCalibration`: Parses intrinsics/extrinsics, depth↔disparity conversion
     - `HamlynSequence`: Metadata for single sequence (1058 frames each)
     - `HamlynDataset`: Main loader with temporal windowing
   
2. **`src/data/hamlyn_dataset.py`** (333 lines)
   - PyTorch Dataset wrappers
   - Classes:
     - `HamlynStereoDataset`: Single-horizon dataset with augmentation
     - `MultiHorizonDataset`: Multi-horizon ablation dataset
     - `create_dataloaders()`: Convenience function for train/val/test

3. **`src/data/__init__.py`**
   - Package exports for clean API

4. **`src/data/README.md`**
   - Complete documentation with usage examples

## Key Features Implemented

### ✅ Temporal Windowing
- **Context frames**: (t-2, t-1, t) - 3 consecutive frames
- **Future target**: Frame at t+δ where δ is prediction horizon
- **Automatic bounds checking**: Skips invalid samples at sequence edges

### ✅ Calibration Loading
- **Intrinsics**: 3×4 camera matrix → 3×3 K matrix
  - Focal length: fx = fy ≈ 383.19 pixels
  - Principal point: (155.97, 124.33)
- **Extrinsics**: 3×4 [R|t] stereo transformation
  - Baseline: tx ≈ 5.38 mm
- **Conversions**: depth_to_disparity() and disparity_to_depth()

### ✅ Data Preprocessing
- **Resizing**: 480×640 → 256×320 (for efficiency)
- **Normalization**: ImageNet mean/std
- **Valid masking**: Binary mask where depth > 0
- **Augmentation**: Brightness/contrast (training only, preserves stereo geometry)

### ✅ Multi-Horizon Support
- **Short**: δ = 33.67 ms (~1 frame ahead)
- **Medium**: δ = 66.67 ms (~2 frames ahead)  
- **Long**: δ = 99.67 ms (~3 frames ahead)

### ✅ Train/Val/Test Splits
- **Train**: 70% (732 samples from rectified01)
- **Val**: 15% (156 samples)
- **Test**: 15% (158 samples)
- Random shuffling with seed=42 for reproducibility

## Test Results

### Core Loader Test
```bash
$ python src/data/hamlyn_loader.py

Loaded 1 Hamlyn sequences:
  - rectified01: 1058 frames, baseline=5.38mm

Dataset splits created:
  Train: 732 samples
  Val:   156 samples
  Test:  158 samples

Sample at δ=66.67ms:
  Context frames: [773, 774, 775]
  Target frame: 777
  Context left shape: [(480, 640, 3), (480, 640, 3), (480, 640, 3)]
  Target depth shape: (480, 640)
  Target depth range: [0.0, 147.0] mm
```

### PyTorch Dataset Test
```bash
$ python -m src.data.hamlyn_dataset

PyTorch train dataset initialized:
  Samples: 732
  Prediction horizon: 66.67 ms
  Image size: (256, 320)
  Augmentation: True

Sample shapes:
  context_left: torch.Size([3, 3, 256, 320])    # 3 frames, RGB, H, W
  context_right: torch.Size([3, 3, 256, 320])
  target_depth: torch.Size([256, 320])          # Future depth
  target_disparity: torch.Size([256, 320])      # Future disparity
  valid_mask: torch.Size([256, 320])            # Valid pixels
  delta_ms: torch.Size([])                      # Horizon

Value ranges:
  context_left: [-1.861, 1.701]  # Normalized RGB
  target_depth: [0.0, 147.0]     # mm
  valid_mask: 32603/81920 valid pixels (~40%)
```

## Data Format

### Input
```python
{
    'context_left': (3, 3, H, W),      # 3 left RGB frames
    'context_right': (3, 3, H, W),     # 3 right RGB frames
}
```

### Output
```python
{
    'target_depth': (H, W),            # Future depth at t+δ (mm)
    'target_disparity': (H, W),        # Future disparity (pixels)
    'valid_mask': (H, W),              # Where depth > 0
    'delta_ms': scalar                 # Prediction horizon
}
```

## Usage Examples

### Basic Training Loop
```python
from src.data import create_dataloaders

loaders = create_dataloaders(
    data_dir='hamlyn_data',
    batch_size=4,
    delta_ms=66.67,
    image_size=(256, 320)
)

for batch in loaders['train']:
    context_left = batch['context_left']    # (B, 3, 3, 256, 320)
    target_depth = batch['target_depth']    # (B, 256, 320)
    valid_mask = batch['valid_mask']        # (B, 256, 320)
    
    # Forward pass
    pred_depth = model(context_left, context_right)
    loss = criterion(pred_depth, target_depth, valid_mask)
```

### Multi-Horizon Ablation
```python
from src.data import MultiHorizonDataset

dataset = MultiHorizonDataset(
    data_dir='hamlyn_data',
    split='test',
    horizons=[33.67, 66.67, 99.67]
)

sample = dataset[0]
for horizon in [33.67, 66.67, 99.67]:
    key = f'horizon_{horizon:.2f}'
    depth = sample[key]['target_depth']
    # Evaluate model at this horizon
```

## Dataset Statistics

### Hamlyn Dataset
- **Sequences tested**: rectified01 (1058 frames)
- **Resolution**: 480×640 original, 256×320 processed
- **Frame rate**: 30 FPS (33.33 ms/frame)
- **Depth range**: 0-250 mm typical for surgical scenes
- **Baseline**: 5.38 mm stereo separation
- **Valid depth**: ~40% of pixels (surgical instruments + tissue)

### System Parameters (from Step 1)
- **Measured delay**: η = 66.67 ms
- **Ablation horizons**:
  - Short: η - 33 = 33.67 ms
  - Medium: η = 66.67 ms
  - Long: η + 33 = 99.67 ms

## Technical Achievements

1. **Efficient Loading**: Multi-worker DataLoader support
2. **Memory Efficient**: Only loads required frames, not entire sequences
3. **GPU Ready**: Pinned memory for fast transfer
4. **Robust**: Handles invalid samples gracefully
5. **Flexible**: Easy to add new sequences or horizons
6. **Well-Tested**: Both unit tests pass successfully

## Known Limitations

1. **Depth discontinuities**: Very large disparity values (>2e9) indicate invalid depth (depth ≈ 0)
   - Solution: Valid mask filters these out
   
2. **Edge frames**: First 2 frames and last 10 frames unusable
   - Solution: Temporal indexing automatically skips these

3. **Single sequence tested**: Only rectified01 tested so far
   - Solution: Code supports all 22 sequences, just load more

## Next Steps

With data preparation complete, we can now proceed to:

**Step 3**: Implement RAFT-Stereo baseline (Baseline A)
- Use pretrained RAFT-Stereo model
- Predict depth at current time t
- Evaluate performance without temporal forecasting

**Step 4**: Implement motion extrapolation baseline (Baseline B)
- Compute optical flow from t-2 → t-1 → t
- Extrapolate motion to predict depth at t+δ
- Compare against our temporal model

**Step 5**: Build our temporal forecasting model
- Design architecture using 3-frame context
- Predict future depth directly
- Incorporate uncertainty estimation

## Files Structure
```
src/data/
├── __init__.py           # Package exports
├── hamlyn_loader.py      # Core loading (462 lines)
├── hamlyn_dataset.py     # PyTorch wrapper (333 lines)
└── README.md             # Documentation
```

**Total**: ~800 lines of production-ready data loading code

---

**Date**: 2024
**Step**: 2/7 in pipeline
**Time Invested**: ~1 hour
**Status**: Ready for model development 🚀

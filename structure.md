# Project Pipeline: Probabilistic 3D Tissue Motion Forecasting from Stereo Surgical Video

## Pipeline Overview

```
Setup → Measure Delay → Data Prep → Baseline A → Baseline B → Our Model → Evaluation → Ablation & Reporting
```

---

## 0. Setup

**Objective:** Initialize project configuration and dataset selection

### Tasks:
- Choose dataset: **Hamlyn rectified stereo**
- Fix frame rate assumption: **30 FPS** (unless measured)

---

## 1. Measure System Delay η (Instrumentation)

**Objective:** Determine end-to-end system latency

### Process:
For each frame, measure timing chain:
```
T_cap → T_pred → T_render → T_vsync
```

### Output:
- **η** = median(T_vsync - T_cap)
- Set ablation horizons: **δ ∈ {η - 33ms, η, η + 33ms}**

---

## 2. Data Preparation (Hamlyn)

**Objective:** Prepare training, validation, and test datasets

### Tasks:
- Load rectified left/right images + calibration files
- Implement train/val/test split
- Build 3-frame context windows: **(t-2, t-1, t)**
- Align ground truth depth/disparity at **t+δ** for each δ

### Data Structure:
```
hamlyn_data/
├── rectified01/
│   ├── image01/  (left camera)
│   ├── image02/  (right camera)
│   ├── depth01/
│   └── depth02/
├── rectified04/
├── ...
└── calibration/
    ├── 01/
    │   ├── intrinsics.txt
    │   └── extrinsics.txt
    └── ...
```

---

## 3. Baseline A: Optimized Per-Frame Stereo Depth

**Objective:** Establish real-time stereo depth baseline

### Implementation:
- Use **RAFT-Stereo-small** (or similar lightweight model)
- Optimize for latency:
  - Resolution tuning
  - Quantization
  - TensorRT optimization
- Profile end-to-end runtime

### Metrics:
- Inference time
- Depth accuracy (EPE, RMSE, AbsRel)

---

## 4. Baseline B: Motion-Only Extrapolation

**Objective:** Predict future depth using motion warping

### Implementation:
- Take last per-frame depth at time **t**
- Compute optical flow: **t → t+Δ**
  - Options: Constant-velocity model, Kalman filter, or flow network
- Warp depth map to predict **t+δ**

### Constraint:
- Runtime must be **≤ Baseline A**

---

## 5. Our Model: Future-Depth Head (Deterministic)

**Objective:** Direct regression of future depth with temporal context

### Architecture:
- **Input:** Last 3 rectified L/R pairs + calibration parameters
- **Backbone:** RAFT-Stereo-small
  - Freeze weights OR light fine-tuning
- **Temporal Head:** Regress left-view disparity at **t+δ**

### Training Strategy:
- Primary training horizon: **δ = η**
- Optional: Condition on δ for generalization to **±33ms**

---

## 6. Latency-Matched Evaluation (Streaming)

**Objective:** Fair comparison under realistic display latency constraints

### Evaluation Protocol:
- Buffer all method outputs so **display latency equals η**
- At each display instant **τ**:
  - Compare latest shown depth map to ground truth at **τ**

### Metrics:
- **Future-EPE@δ:** End-point error at prediction horizon δ
- **AbsRel:** Absolute relative depth error
- **RMSE:** Root mean squared error
- **Overlay drift:** Pixel-wise drift in overlays (px)

---

## 7. Light Ablation & Reporting

**Objective:** Validate across multiple prediction horizons and synthesize results

### Ablation Study:
- Test all methods at **δ ∈ {η - 33ms, η, η + 33ms}**

### Results Table:
```
| Method            | δ (ms)   | EPE    | RMSE   | AbsRel | Drift (px) |
|-------------------|----------|--------|--------|--------|------------|
| Baseline A        | η - 33   | ...    | ...    | ...    | ...        |
| Baseline A        | η        | ...    | ...    | ...    | ...        |
| Baseline A        | η + 33   | ...    | ...    | ...    | ...        |
| Baseline B        | η - 33   | ...    | ...    | ...    | ...        |
| Baseline B        | η        | ...    | ...    | ...    | ...        |
| Baseline B        | η + 33   | ...    | ...    | ...    | ...        |
| Our Model         | η - 33   | ...    | ...    | ...    | ...        |
| Our Model         | η        | ...    | ...    | ...    | ...        |
| Our Model         | η + 33   | ...    | ...    | ...    | ...        |
```

### Decision Criteria:
- **If** per-frame stereo @ η eliminates drift:
  - Report **negative result** (future prediction not needed)
- **Else:**
  - Quantify **% improvement** of our temporal head vs both baselines
  - Highlight where temporal context provides benefit

---

## Summary

This pipeline establishes a rigorous framework for evaluating whether temporal context improves depth prediction under real-time constraints in surgical video. The three-way comparison (real-time stereo, motion warping, temporal prediction) provides comprehensive evidence for the utility (or lack thereof) of future depth forecasting.

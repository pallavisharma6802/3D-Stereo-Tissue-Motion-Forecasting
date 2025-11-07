# Project Understanding: Probabilistic 3D Tissue Motion Forecasting from Stereo Surgical Video

## Overview
This project aims to predict future depth/disparity maps in surgical endoscopy by leveraging temporal context from stereo video streams, enabling more stable surgical guidance systems.

---

## Key Changes from Original Proposal

### 1. Dataset Change
- **Original Proposal**: SCARED dataset
- **Current Implementation**: **Hamlyn dataset** (rectified stereo sequences)
  - 22 sequences (rectified01-27, some numbers skipped)
  - Each contains: `image01/`, `image02/` (stereo pairs), `depth01/`, `depth02/` (depth maps)
  - Calibration data available for all sequences

### 2. Prediction Horizon Change
- **Original Proposal**: δ = 200ms (fixed)
- **Current Implementation**: **Light ablations** at δ ∈ {η - 33ms, η, η + 33ms}
  - η = system delay (measured via instrumentation)
  - More flexible evaluation across multiple horizons

### 3. Architecture Simplification
- **Original Proposal**: Conditional diffusion forecast head with probabilistic outputs
- **Current Implementation**: **Deterministic future-depth head**
  - Simpler, more interpretable approach
  - Direct regression of left-view disparity at t+δ

---

## Core Problem Definition

**Task**: Given a short window of rectified stereo frames + calibration → predict left-view depth/disparity at time t+δ

**Why This Matters**:
- Current surgical vision estimates "depth now"
- Tissue moves (breathing, heartbeat, tool motion) → overlays/trackers drift
- Need "depth next" to keep guidance systems stable
- Critical for catheter/stent alignment, AR/VR surgical navigation

---

## Research Background (Key Papers)

### 1. **Streaming Perception** (Li et al., CMU)
**Core Insight**: By the time an algorithm finishes processing a frame (after latency η), the world has changed!

**Key Concepts**:
- **Streaming accuracy**: Joint metric integrating latency + accuracy
- **Zero-order hold**: If computation isn't done, use last available prediction
- **Optimal sweet spot**: There exists an optimal latency-accuracy trade-off point
- **Future forecasting emerges naturally** as a solution to streaming tasks

**Relevance to Our Project**:
- Validates need for predictive forecasting in real-time systems
- Provides evaluation framework (latency-matched comparison)
- Shows that asynchronous tracking + forecasting improve streaming performance

### 2. **LightEndoStereo** (Ding et al., 2025)
**Contribution**: Real-time lightweight stereo matching for endoscopy

**Key Innovations**:
- **3D Mamba Coordinate Attention (MCA)**: Captures long-range dependencies across spatial dimensions
- **High-Frequency Disparity Optimization (HFDO)**: Refines tissue boundaries using wavelet transform
- **Performance**: 42 FPS at 1024×1280, MAE of 2.592mm on SCARED
- Uses MobileNetV4 backbone for efficiency

**Relevance to Our Project**:
- Provides state-of-the-art baseline for real-time stereo depth
- Shows importance of boundary refinement in surgical scenes
- Demonstrates feasibility of HD real-time processing

### 3. **Long-term Reprojection Loss** (Shi et al., UCL)
**Contribution**: Self-supervised monocular depth with temporal context

**Key Insights**:
- **Small camera pose changes** in narrow surgical environment require **longer temporal windows**
- Vanilla reprojection (2 adjacent frames) insufficient for occlusion artifacts
- **LT-RL**: Uses 4 adjacent frames (t-2, t-1, t+1, t+2) to handle occlusions
- Significant improvement: AbsRel from 0.177 → 0.147

**Relevance to Our Project**:
- Validates use of 3-frame context windows (t-2, t-1, t)
- Shows temporal context crucial for surgical scenes
- Provides monocular depth baseline for comparison

---

## Proposed Pipeline (From structure.md)

### Stage 0: Setup
- Dataset: **Hamlyn rectified stereo**
- Frame rate: **30 FPS** (unless measured)

### Stage 1: Measure System Delay η
**Goal**: Determine end-to-end latency

**Process**:
```
T_cap → T_pred → T_render → T_vsync
η = median(T_vsync - T_cap)
```

**Output**: Set ablation horizons δ ∈ {η - 33ms, η, η + 33ms}

### Stage 2: Data Preparation
**Tasks**:
1. Load rectified L/R images + calibration from Hamlyn sequences
2. Implement train/val/test split
3. Build **3-frame context windows**: (t-2, t-1, t)
4. Align ground truth depth/disparity at t+δ for each δ

### Stage 3: Baseline A - Optimized Per-Frame Stereo
**Method**: RAFT-Stereo-small (or similar)

**Optimizations**:
- Resolution tuning
- Quantization
- TensorRT optimization

**Metrics**: Inference time, EPE, RMSE, AbsRel

### Stage 4: Baseline B - Motion-Only Extrapolation
**Approach**:
1. Take last per-frame depth at time t
2. Compute optical flow: t → t+Δ (constant-velocity/Kalman/flow network)
3. Warp depth to predict t+δ

**Constraint**: Runtime ≤ Baseline A

### Stage 5: Our Model - Future-Depth Head
**Architecture**:
- **Input**: Last 3 rectified L/R pairs + calibration
- **Backbone**: RAFT-Stereo-small (freeze or light fine-tuning)
- **Temporal Head**: Regress left-view disparity at t+δ

**Training**:
- Primary: δ = η
- Optional: Condition on δ for ±33ms generalization

### Stage 6: Latency-Matched Evaluation
**Protocol**:
- Buffer outputs so display latency equals η for ALL methods
- At each display instant τ: compare latest shown map to GT at τ

**Metrics**:
- **Future-EPE@δ**: End-point error at prediction horizon
- **AbsRel**: Absolute relative depth error
- **RMSE**: Root mean squared error
- **Overlay drift**: Pixel-wise drift in overlays

### Stage 7: Ablation & Reporting
**Experiments**: Test all methods at δ ∈ {η - 33ms, η, η + 33ms}

**Decision Criteria**:
- **If** per-frame stereo @ η eliminates drift → Report negative result
- **Else**: Quantify % improvement of temporal head vs both baselines

---

## Technical Insights from Literature

### Challenges in Surgical Depth Estimation
1. **Ambiguous tissue boundaries** (homogeneous textures)
2. **Heavy runtime at HD resolution** (1024×1280)
3. **Small camera pose changes** in confined surgical environment
4. **Occlusion artifacts** from tool motion, tissue deformation
5. **Limited generalization** across different surgical scenes

### Why Temporal Context Matters
1. **Small pose changes** → Need longer temporal windows to avoid occlusions
2. **Endoscope motion** is complex 3D rotation with restricted translation
3. **Occluded pixels in immediate frames** may appear in longer temporal spans
4. **Motion + appearance cues** together predict future scene state

### Key Design Decisions
1. **3-frame context** balances temporal information vs. computational cost
2. **Deterministic output** (not probabilistic) for simplicity and interpretability
3. **Light fine-tuning** of backbone preserves learned features
4. **Latency-matched evaluation** ensures fair comparison under realistic constraints

---

## Expected Outcomes

### Success Criteria
1. **Temporal head reduces drift** compared to per-frame stereo
2. **Motion extrapolation alone** insufficient (validates need for learned forecasting)
3. **Optimal δ** found through ablation study
4. **Quantifiable improvement** in Future-EPE, drift metrics

### Potential Challenges
1. Dataset size (Hamlyn < SCARED)
2. Ground truth depth quality/alignment
3. Computational cost of temporal processing
4. Generalization to unseen surgical scenes

### Broader Impact
If successful, this work:
- Stabilizes image-fusion roadmaps in endoscopic navigation
- Keeps trackers locked during tissue motion
- Reduces manual corrections in catheter/stent alignment
- Improves guidance reliability in surgical AR/VR

---

## Key Differences from Related Work

| Aspect | Streaming Perception | LightEndoStereo | Our Project |
|--------|---------------------|-----------------|-------------|
| **Task** | Object detection in driving | Current-frame stereo depth | **Future depth prediction** |
| **Temporal** | Tracking + forecasting | Single-frame | **3-frame context** |
| **Domain** | Autonomous vehicles | Surgical (current) | **Surgical (future)** |
| **Latency** | General framework | Real-time focus | **Prediction horizon** |
| **Evaluation** | Streaming accuracy | Standard metrics | **Latency-matched** |

---

## Next Steps (Implementation Plan)

1. ✅ **Extract Hamlyn dataset** (DONE)
2. **Measure system delay η** (instrumentation)
3. **Data preprocessing pipeline**:
   - Train/val/test split
   - 3-frame context windows
   - GT alignment at t+δ
4. **Implement baselines**:
   - Baseline A: Optimized RAFT-Stereo
   - Baseline B: Motion extrapolation
5. **Build our model**:
   - Temporal fusion head
   - Training loop
6. **Evaluation framework**:
   - Latency-matched metrics
   - Ablation experiments
7. **Analysis & reporting**

---

## Summary

This project bridges the gap between "see now" and "see next" in surgical vision by leveraging temporal context from stereo video. Drawing on insights from streaming perception (Li et al.), lightweight stereo matching (Ding et al.), and temporal self-supervision (Shi et al.), we propose a focused, reproducible approach to future depth prediction with practical surgical applications.

The switch from SCARED to **Hamlyn dataset** and from fixed 200ms to **light ablations** (±33ms around measured delay η) makes the project more flexible and scientifically rigorous. The deterministic architecture simplifies implementation while maintaining practical utility.

**Core hypothesis**: Temporal context enables better future depth prediction than either per-frame stereo or motion-only extrapolation, reducing drift in surgical guidance systems.

#!/bin/bash
# Setup script for RAFT-Stereo baseline

set -e

echo "Setting up RAFT-Stereo baseline..."
echo "===================================="

# Create third_party directory
mkdir -p third_party
cd third_party

# Clone RAFT-Stereo if not already present
if [ ! -d "RAFT-Stereo" ]; then
    echo "Cloning RAFT-Stereo repository..."
    git clone https://github.com/princeton-vl/RAFT-Stereo.git
    cd RAFT-Stereo
    
    # Apply any necessary patches
    echo "RAFT-Stereo cloned successfully"
else
    echo "RAFT-Stereo already exists"
    cd RAFT-Stereo
fi

cd ../..

# Create checkpoints directory
mkdir -p checkpoints

# Download pretrained model
CHECKPOINT="checkpoints/raftstereo-middlebury.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo ""
    echo "To download pretrained RAFT-Stereo model:"
    echo "  1. Visit: https://github.com/princeton-vl/RAFT-Stereo"
    echo "  2. Download raftstereo-middlebury.pth (or raftstereo-sceneflow.pth)"
    echo "  3. Place in: checkpoints/raftstereo-middlebury.pth"
    echo ""
    echo "Or use this direct link (if available):"
    echo "  wget -O $CHECKPOINT https://drive.google.com/uc?id=1yTfxJcRQAN5KkTWEJZqJIBSEoNHALb0z"
else
    echo "Checkpoint already exists: $CHECKPOINT"
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "  1. Download pretrained weights if not done"
echo "  2. Test baseline: python src/baselines/raft_stereo_baseline.py"
echo "  3. Evaluate on Hamlyn: python src/baselines/evaluate_baseline_a.py"

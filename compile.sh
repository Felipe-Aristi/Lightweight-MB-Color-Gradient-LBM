#!/usr/bin/env bash

set -e

# =======================================================
# Executable name
# =======================================================

EXE=lbm_solver

# =======================================================
# GPU architecture
# =======================================================
# RTX 3050 laptop: SM=86
# RTX 4090:        SM=89

SM=86
LBM_BLOCK_X=${LBM_BLOCK_X:-32}
LBM_BLOCK_Y=${LBM_BLOCK_Y:-4}
LBM_BLOCK_Z=${LBM_BLOCK_Z:-4}

# =======================================================
# Source files
# =======================================================

SRC="main.cu kernels.cu"

# =======================================================
# Compile
# =======================================================

echo "Compiling..."

nvcc $SRC \
    -o $EXE \
    -O3 \
    -std=c++20 \
    --restrict \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -DLBM_BLOCK_X=${LBM_BLOCK_X} \
    -DLBM_BLOCK_Y=${LBM_BLOCK_Y} \
    -DLBM_BLOCK_Z=${LBM_BLOCK_Z} \
    -gencode arch=compute_${SM},code=sm_${SM} \
    -gencode arch=compute_${SM},code=compute_${SM} \
    -lineinfo \
    -Xptxas -v

echo "Compilation finished."

# =======================================================
# Run
# =======================================================

echo "Running ./$EXE ..."
./$EXE

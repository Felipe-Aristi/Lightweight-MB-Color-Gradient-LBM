#!/usr/bin/env bash

set -e

# =======================================================
# Executable name
# =======================================================

STENCIL=${STENCIL:-D3Q27}
RUN=${RUN:-1}

case "$STENCIL" in
    D3Q27)
        EXE=lbm_solver_d3q27
        STENCIL_FLAG=""
        ;;
    D3Q19)
        EXE=lbm_solver_d3q19
        STENCIL_FLAG="-DLBM_D3Q19"
        ;;
    *)
        echo "Unknown STENCIL='$STENCIL'. Use D3Q27 or D3Q19." >&2
        exit 1
        ;;
esac

# =======================================================
# GPU architecture
# =======================================================
# RTX 3050 laptop: SM=86
# RTX 4090:        SM=89

SM=89
LBM_BLOCK_X=${LBM_BLOCK_X:-32}
LBM_BLOCK_Y=${LBM_BLOCK_Y:-4}
LBM_BLOCK_Z=${LBM_BLOCK_Z:-2}

# =======================================================
# Source files
# =======================================================

SRC="main.cu kernels.cu"

# =======================================================
# Compile
# =======================================================

echo "Compiling..."
echo "Stencil: $STENCIL"

nvcc $SRC \
    -o $EXE \
    -O3 \
    -std=c++20 \
    --restrict \
    --extended-lambda \
    --expt-relaxed-constexpr \
    $STENCIL_FLAG \
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

if [ "$RUN" = "1" ]; then
    echo "Running ./$EXE ..."
    ./$EXE
else
    echo "Skipping run because RUN=$RUN."
fi

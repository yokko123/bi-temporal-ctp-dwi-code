#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=30:00:00
#SBATCH --job-name=glcm
#SBATCH --output=glcm_%j.out
#SBATCH --error=glcm_%j.err
#
# FE2: 3D GLCM feature extraction over the spatiotemporal CTP slices.
# Runs on CPU only; the paper used a 64-core node with 64 GB RAM.

set -euo pipefail
cd "$(dirname "$0")"

# Local cohort. Roots come from the environment; see ../README.md.
python 3d_glcm.py

# ISLES'24:
# python 3d_glcm_isles.py

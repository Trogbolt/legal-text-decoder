#!/bin/bash
set -e

cd /app/src

echo "=================================================="
echo "Pipeline started at: $(date)"
echo "=================================================="

echo "=== STEP 0: DOWNLOAD AND UNPACK ==="
python 00_download_and_unpack.py

echo "=== STEP 1: DATA PROCESSING ==="
python 01_data_preprocessing.py

echo "=== HEURISTIC BASELINE (WORD COUNT) ==="
python baseline_heuristic.py

echo "=== TF-IDF BASELINE ==="
python baseline.py

echo "=== STEP 2: TRAINING ==="
python 02_train.py

echo "=== STEP 3: EVALUATION ==="
python 03_evaluation.py

echo "=== STEP 4: INFERENCE ==="
python 04_inference.py

echo "=================================================="
echo "Pipeline finished successfully at: $(date)"
echo "=================================================="

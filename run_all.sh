#!/bin/bash
# run_all.sh - Complete pipeline execution

set -e  # Exit on error

echo "========================================"
echo "ATTRIBUTE AFFINITY PIPELINE"
echo "========================================"

# 1. ETL
echo ""
echo "STEP 1: ETL"
python etl/run_etl.py

# 2. Generate Embeddings
echo ""
echo "STEP 2: Generate Embeddings"
python run_provider.py

# 3. Compare Providers
echo ""
echo "STEP 3: Compare Providers"
python compare_providers.py

# 4. Train Two-Tower (with best provider)
echo ""
echo "STEP 4: Train Two-Tower"
python run_two_tower.py --provider gcl

# 5. Generate Affinity Scores
echo ""
echo "STEP 5: Generate Affinity Scores"
python run_inference.py --provider gcl

echo ""
echo "✅ PIPELINE COMPLETE!"
echo "Results: gs://your-bucket/models/hh_attribute_affinity_gcl.parquet"
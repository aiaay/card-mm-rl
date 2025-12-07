#!/bin/bash
# Run comprehensive evaluation suite for all trained agents
# This script evaluates agents against baselines and each other

set -e  # Exit on error

echo "=================================================="
echo "  Comprehensive Agent Evaluation Suite"
echo "=================================================="
echo ""

# Create results directory
mkdir -p data/results

# 1. Evaluate Baselines
echo "Step 1/4: Evaluating baseline agents..."
python scripts/eval_baselines.py \
    --n-episodes 100 \
    --output data/results/baselines.csv

echo ""
echo "Step 2/4: Evaluating IPPO vs baselines..."
python -m mmrl.eval.compare_agents \
    --agent ippo \
    --episodes 100 \
    --output data/results/ippo_vs_baselines.csv

echo ""
echo "Step 3/4: Evaluating MAPPO vs baselines..."
python -m mmrl.eval.compare_agents \
    --agent mappo \
    --episodes 100 \
    --output data/results/mappo_vs_baselines.csv

echo ""
echo "Step 4/4: Evaluating RL agents against each other..."
python scripts/eval_rl_vs_rl.py \
    --agents ippo mappo \
    --episodes 100 \
    --output data/results/rl_vs_rl.csv

echo ""
echo "=================================================="
echo "  Evaluation Complete!"
echo "=================================================="
echo ""
echo "Results saved to data/results/:"
echo "  - baselines.csv             (baseline comparison)"
echo "  - ippo_vs_baselines.csv     (IPPO vs rule-based)"
echo "  - mappo_vs_baselines.csv    (MAPPO vs rule-based)"
echo "  - rl_vs_rl.csv              (all RL matchups)"
echo ""
echo "View summary:"
echo "  cat data/results/*.csv"


#!/usr/bin/env bash
# Reproduce AutoVLA's reported NAVSIM PDMS with the released checkpoint.
#
# This is the end-to-end validation of the port: component-level checks (strict
# checkpoint load, byte-identical prompt, vendored tokenizer) say the pieces are
# right, but only reproducing the published number says the whole stack is.
#
#   SPLIT=warmup_test_e2e ./scripts/run_navsim_reproduction.sh   # smoke test
#   SPLIT=navtest         ./scripts/run_navsim_reproduction.sh   # the real thing
#
# Reported: PDMS 89 on navtest (the checkpoint is literally AutoVLA_PDMS_89.ckpt).
#
# Storage: metric cache + preprocessed JSON are regenerable, so they live on
# NVMe; scores and logs go to EFS.
set -euo pipefail

AUTOVLA_ROOT=/mnt/efs/users/rod/repos/AutoVLA
VENV=/opt/dlami/nvme/rod/venvs/autovla
SPLIT="${SPLIT:-warmup_test_e2e}"
GPU="${GPU:-1}"
# Metric caching is CPU-bound; sequential is fine for the smoke split but
# hopeless for navtest's 12k tokens.
WORKER="${WORKER:-single_machine_thread_pool}"
OUT_ROOT="${OUT_ROOT:-/mnt/efs/users/rod}"

CACHE_ROOT=/opt/dlami/nvme/rod/cache/autovla_navsim
METRIC_CACHE="$CACHE_ROOT/${SPLIT}_metric_cache"
JSON_DIR="$CACHE_ROOT/${SPLIT}_nocot"
RESULTS="$OUT_ROOT/autovla_navsim_repro/$SPLIT"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$AUTOVLA_ROOT/dataset/nuplan/maps"
export OPENSCENE_DATA_ROOT="$AUTOVLA_ROOT/dataset/nuplan"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA_ROOT/navsim"
export NAVSIM_EXP_ROOT="$RESULTS/exp"
export PYTHONPATH="$AUTOVLA_ROOT:$AUTOVLA_ROOT/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3

CKPT=/mnt/efs/users/rod/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt
PRECONF="dataset/autovla-${SPLIT}-nocot"

mkdir -p "$METRIC_CACHE" "$JSON_DIR" "$RESULTS" "$NAVSIM_EXP_ROOT"
cd "$AUTOVLA_ROOT"
say() { echo "$(date '+%F %T') [repro] $*" | tee -a "$RESULTS/run.log"; }
say "split=$SPLIT gpu=$GPU results=$RESULTS"

# --- 1. metric cache (CPU, parallel) -----------------------------------------
if [[ -z "$(ls -A "$METRIC_CACHE" 2>/dev/null)" ]]; then
  say "building metric cache -> $METRIC_CACHE"
  "$VENV/bin/python" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$SPLIT" cache.cache_path="$METRIC_CACHE" \
    worker="$WORKER" \
    >> "$RESULTS/metric_cache.log" 2>&1
else
  say "metric cache present, skipping"
fi

# --- 2. per-token agent input JSON (no CoT) ----------------------------------
if [[ -z "$(ls -A "$JSON_DIR" 2>/dev/null)" ]]; then
  say "preprocessing -> $JSON_DIR"
  "$VENV/bin/python" tools/preprocessing/nocot_sample_generation.py \
    --config "$PRECONF" --output_dir "$JSON_DIR" \
    >> "$RESULTS/preprocess.log" 2>&1
else
  say "preprocessed JSON present, skipping"
fi
say "tokens preprocessed: $(ls "$JSON_DIR" 2>/dev/null | wc -l)"

# --- 3. PDM score ------------------------------------------------------------
say "scoring with the released checkpoint (LoRA already merged -> use_lora=false)"
CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
  train_test_split="$SPLIT" \
  agent=autovla_agent \
  +agent.config_path="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml" \
  +agent.checkpoint_path="$CKPT" \
  +agent.sensor_data_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  +agent.lora_conf.use_lora=false \
  metric_cache_path="$METRIC_CACHE" \
  json_data_path="$JSON_DIR" \
  experiment_name=autovla_repro_"$SPLIT" \
  >> "$RESULTS/pdm_score.log" 2>&1
say "done -> $RESULTS"

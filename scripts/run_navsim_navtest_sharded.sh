#!/usr/bin/env bash
# Full navtest reproduction of AutoVLA's reported PDMS (~89).
#
# Metric cache and preprocessed agent-input JSON are built ONCE over all of
# navtest; only the GPU-bound scoring is sharded, one shard per GPU. Each shard
# restricts the scene filter to a disjoint set of logs, and run_pdm_score_cot
# scores scene_loader.tokens & metric_cache.tokens -- so the shards partition
# navtest exactly and the per-token rows concatenate back to the whole.
#
#   SHARD_GPUS=1,2,3,4,6,7 ./scripts/run_navsim_navtest_sharded.sh
set -uo pipefail

AUTOVLA_ROOT=/mnt/efs/users/rod/repos/AutoVLA
VENV=/opt/dlami/nvme/rod/venvs/autovla
SHARD_GPUS="${SHARD_GPUS:-1,2,3,4,6,7}"
OUT_ROOT="${OUT_ROOT:-/mnt/efs/users/rod}"
WORKER="${WORKER:-single_machine_thread_pool}"

CACHE_ROOT=/opt/dlami/nvme/rod/cache/autovla_navsim
METRIC_CACHE="$CACHE_ROOT/navtest_metric_cache"
JSON_DIR="$CACHE_ROOT/navtest_nocot"
RESULTS="$OUT_ROOT/autovla_navsim_repro/navtest"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$AUTOVLA_ROOT/dataset/nuplan/maps"
export OPENSCENE_DATA_ROOT="$AUTOVLA_ROOT/dataset/nuplan"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA_ROOT/navsim"
export PYTHONPATH="$AUTOVLA_ROOT:$AUTOVLA_ROOT/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
CKPT=/mnt/efs/users/rod/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt

mkdir -p "$METRIC_CACHE" "$JSON_DIR" "$RESULTS"
cd "$AUTOVLA_ROOT"
say() { echo "$(date '+%F %T') [navtest] $*" | tee -a "$RESULTS/run.log"; }

IFS=',' read -r -a GPUS <<< "$SHARD_GPUS"
N=${#GPUS[@]}
say "navtest reproduction: $N shards on GPUs $SHARD_GPUS"

# --- 1. metric cache over ALL of navtest -------------------------------------
if [[ $(find "$METRIC_CACHE" -name 'metric_cache.pkl' 2>/dev/null | wc -l) -lt 10 ]]; then
  say "building navtest metric cache (worker=$WORKER)"
  NAVSIM_EXP_ROOT="$RESULTS/exp" "$VENV/bin/python" \
    "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest cache.cache_path="$METRIC_CACHE" worker="$WORKER" \
    > "$RESULTS/metric_cache.log" 2>&1 || { say "ABORT: metric caching failed"; exit 1; }
fi
say "metric cache entries: $(find "$METRIC_CACHE" -name 'metric_cache.pkl' 2>/dev/null | wc -l)"

# --- 2. preprocess ALL of navtest --------------------------------------------
if [[ $(ls "$JSON_DIR" 2>/dev/null | wc -l) -lt 10 ]]; then
  say "preprocessing navtest -> $JSON_DIR"
  NAVSIM_EXP_ROOT="$RESULTS/exp" "$VENV/bin/python" \
    tools/preprocessing/nocot_sample_generation.py \
    --config dataset/autovla-navtest-nocot --output_dir "$JSON_DIR" \
    > "$RESULTS/preprocess.log" 2>&1 || { say "ABORT: preprocessing failed"; exit 1; }
fi
say "preprocessed tokens: $(ls "$JSON_DIR" 2>/dev/null | wc -l)"

# --- 3. score, one shard per GPU ---------------------------------------------
PIDS=()
for i in "${!GPUS[@]}"; do
  G="${GPUS[$i]}"; SHARD_RES="$RESULTS/shard$i"; mkdir -p "$SHARD_RES/exp"
  (
    CUDA_VISIBLE_DEVICES="$G" NAVSIM_EXP_ROOT="$SHARD_RES/exp" "$VENV/bin/python" \
      "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
      train_test_split="navtest_shard$i" \
      agent=autovla_agent \
      +agent.config_path="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml" \
      +agent.checkpoint_path="$CKPT" \
      +agent.sensor_data_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
      +agent.lora_conf.use_lora=false \
      metric_cache_path="$METRIC_CACHE" \
      json_data_path="$JSON_DIR" \
      experiment_name="autovla_navtest_shard$i" \
      > "$SHARD_RES/pdm_score.log" 2>&1
    echo "$(date '+%F %T') [navtest] shard$i (gpu=$G) rc=$?" >> "$RESULTS/run.log"
  ) &
  PIDS+=($!); sleep 5
done
say "all $N shards launched; waiting"
for p in "${PIDS[@]}"; do wait "$p"; done

# --- 4. merge ----------------------------------------------------------------
say "merging shard results"
"$VENV/bin/python" scripts/merge_navtest_shards.py "$RESULTS" | tee -a "$RESULTS/run.log"
say "done -> $RESULTS"

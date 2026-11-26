#!/usr/bin/env bash
# Score one navtest shard on one GPU, against the shared metric cache + JSON.
#   ./scripts/score_navtest_shard.sh <shard_idx> <gpu>
set -uo pipefail
AUTOVLA_ROOT=/mnt/efs/users/rod/repos/AutoVLA
VENV=/opt/dlami/nvme/rod/venvs/autovla
SHARD="${1:?shard index}"; GPU="${2:?gpu}"
C=/opt/dlami/nvme/rod/cache/autovla_navsim
R=/mnt/efs/users/rod/autovla_navsim_repro/navtest/shard$SHARD
mkdir -p "$R/exp"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$AUTOVLA_ROOT/dataset/nuplan/maps"
export OPENSCENE_DATA_ROOT="$AUTOVLA_ROOT/dataset/nuplan"
export NAVSIM_DEVKIT_ROOT="$AUTOVLA_ROOT/navsim"
export NAVSIM_EXP_ROOT="$R/exp"
export PYTHONPATH="$AUTOVLA_ROOT:$AUTOVLA_ROOT/navsim:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3
cd "$AUTOVLA_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" \
  "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_cot.py" \
  train_test_split="navtest_shard$SHARD" \
  agent=autovla_agent \
  +agent.config_path="$AUTOVLA_ROOT/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml" \
  +agent.checkpoint_path=/mnt/efs/users/rod/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt \
  +agent.sensor_data_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  +agent.lora_conf.use_lora=false \
  metric_cache_path="$C/navtest_metric_cache" \
  json_data_path="$C/navtest_nocot" \
  experiment_name="autovla_navtest_shard$SHARD" \
  >> "$R/pdm_score.log" 2>&1
echo "$(date '+%F %T') shard$SHARD (gpu=$GPU) rc=$?" >> /mnt/efs/users/rod/autovla_navsim_repro/navtest/run.log

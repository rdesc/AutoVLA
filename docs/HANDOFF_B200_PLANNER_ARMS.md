# Handoff: the four planner arms on B200

Train and evaluate AutoVLA on HUGSIM's PDM-Closed and oracle-future planner rollouts, with and
without reasoning, at **5 epochs**. Written 2026-11-27 for a run on B200 hardware.

Read `FINETUNING_FINDINGS.md` §1, §3 and §8 first — they are short and they are what makes the
numbers below interpretable.

---

## The four arms

| config (`config/training/qwen2.5-vl-3B-hugsim-<name>.yaml`) | data | reasoning | steps |
|---|---|---|---|
| `pdmclosed-5ep` | PDM-Closed, 7,158 train | no | 1,490 |
| `pdmclosed-cotmatched-5ep` | same decisions + captions, 7,155 | **yes** | 1,490 |
| `oraclefuture-5ep` | oracle-future, 6,866 train | no | 1,430 |
| `oraclefuture-cotmatched-5ep` | same decisions + captions, 6,852 | **yes** | 1,425 |

They form two pairs. Within a pair the ONLY difference that reaches the model is whether
`cot_output` is populated — same decisions, same scenarios, same splits, same hyperparameters.
Run all four so each pair is trained under identical conditions; do not pair a new arm against a
previously-trained one.

## ⚠️ The one setting you must change for B200

**Effective batch must be 24.** It is `GPUs x batch_size x accumulate_grad_batches`, and the
configs assume **8 GPUs** with `accumulate_grad_batches: 3`.

    8 GPUs  -> accumulate_grad_batches: 3    (what the configs say)
    4 GPUs  -> accumulate_grad_batches: 6
    2 GPUs  -> accumulate_grad_batches: 12
    16 GPUs -> accumulate_grad_batches: 2  (and batch_size stays 1)

This is not a tuning preference. Measured on identical data, §2:

| effective batch | All* |
|---|---|
| 24 | **39.7** |
| 32 | 32.4 |
| 128 | 31.5 |

**Batch 32 costs 7.3 points** — three times the noise floor, and nearly the whole penalty seen at
128. It is a cliff, not a slope. The published arms used `accumulate: 4` because they ran on SIX
GPUs; inheriting that number on eight silently gives 32. If B200 changes your GPU count, fix
`accumulate` so the product is 24, and re-derive `lr_warmup_step` as 7% of the resulting step count.

If B200 memory lets you raise `batch_size` above 1, **do not** — that changes the effective batch
too. Keep `batch_size: 1` and adjust only `accumulate_grad_batches`.

## Why 5 epochs

§3's convergence result ("epochs 2 and 3 add nothing") was measured on the 36k teacher set, where
one epoch is **1,435 optimizer steps**. These sets are ~7.2k, so one epoch is only ~298 steps. The
finding is *~1,400 steps is enough*, not *one epoch is enough*. Five epochs here is ~1,450 steps,
which lands on the step count where convergence was actually verified. It also matches what
ReCogDrive runs on the same rollouts (`hugsim_pdmclosed_cot_executed_5ep`).

If you change the dataset size, size the run in steps:
`epochs = round(1400 * effective_batch / n_examples)`.

## Running it

    cd /mnt/efs/users/rod/repos/AutoVLA
    ARMS="pdmclosed-5ep pdmclosed-cotmatched-5ep oraclefuture-5ep oraclefuture-cotmatched-5ep" \
      bash /mnt/efs/users/rod/autovla_analysis/scripts/hugsim_arm_chain.sh

That chain trains each arm, evaluates the last checkpoint on the HUGSIM 155, scores it, and fails
loudly on an empty table row. To run a single arm: `ARMS="pdmclosed-5ep" bash .../hugsim_arm_chain.sh`.

On 8xA100, 1 epoch of ~7.2k took ~17 min, so expect **~85 min per arm** at 5 epochs, plus eval:
~20 min for a no-reasoning arm, **~56 min for a reasoning arm** (CoT generation dominates), plus
~12 min scoring. Roughly **7-8 h for all four** on A100; B200 should be faster.

## Traps that have already cost us time

1. **`scripts/run_autovla_hugsim.sh` runs the sweep ONLY — it does not score.** Score with
   `simulators/hugsim_garage/score_realized.py` then `scripts/mila/paper_table_row.py`. A chain can
   log "SCORED" over a 0-byte `table_row.txt`; check the file is non-empty.
2. **`python` is not on PATH in a `nohup`'d shell** (only `python3`). Source `setup/env_a100.sh`
   first, or scoring silently becomes "command not found". This cost 10 h of idle GPU once.
3. **`AGENT` must differ per checkpoint.** `run_hugsim_sweep.sh` skips any scenario that already has
   an `eval.json`, so reusing a finished run's name evaluates NOTHING and reports the old result.
4. **Never wait on a process with `pgrep -f <pattern>` when the waiter's own command line contains
   that pattern** — it matches itself and waits forever. Capture the PID and wait on `/proc/<pid>`.
5. **Identify end-of-epoch checkpoints by mtime**, not by the `epoch=` in the filename: at
   `val_check_interval` < 1 every checkpoint in an epoch carries the same index and two can tie on
   the loss.
6. **The frames live on NVMe**, `/opt/dlami/nvme/rod/datasets/hugsim_oracle/`, which is EPHEMERAL.
   The SFT JSONs reference them by absolute path. If that directory is gone, re-extract from
   `$ROD/drive_staging/oracle_pdm_closed_hd.tar.zst` and Drive's `oracle_future_hd.tar.zst`, and
   rewrite the paths if the root differs.

## What to compare against

Existing numbers, all at 1 epoch / ~300 steps and therefore probably undertrained — treat them as a
floor, not a target:

| arm | All* |
|---|---|
| base `AutoVLA_PDMS_89` | 24.8 |
| PDM-Closed, 1 epoch | 25.5 |
| oracle-future, 1 epoch | 20.1 |
| teacher no-reasoning, 7.6k, 1 epoch | 28.8 |
| teacher no-reasoning, 36k, 1 epoch | 32.4 |
| teacher **with** reasoning, 36k, 1 epoch | **39.7** |

**The noise floor is ±2.4 points on All\*** (§1 — two runs of the same recipe on the same data,
differing only in GPU count, scored 38.6 and 41.0). Do not report a difference smaller than that as
an effect. The reasoning deltas we are chasing here were −3.5 and +1.4 at 1 epoch, i.e. inside the
noise; 5 epochs may or may not change that, and "indistinguishable from zero" is a legitimate
result worth reporting as such.

## Data

### 1. The frames (52 GB) — fetch these FIRST

The SFT JSONs are decisions and labels only. Every training example references **12 camera frames**
by ABSOLUTE PATH — 3 cameras (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`) x 4 context steps
0.5 s apart — and those frames are not in any of the SFT artifacts. That is why the SFT tarball is
9.4 MB while the data it depends on is 52 GB.

    Drive: "agentic driver assets/oracle_pdm_closed_hd.tar.zst"   13.4 GB  -> extracts to 20 GB
           "agentic driver assets/oracle_future_hd.tar.zst"       21.6 GB  -> extracts to 32 GB

    tar --zstd -xf oracle_pdm_closed_hd.tar.zst -C <root>
    tar --zstd -xf oracle_future_hd.tar.zst     -C <root>

A JSON references them like this:

    /opt/dlami/nvme/rod/datasets/hugsim_oracle/oracle_pdm_closed_hd/<scenario>/claude_driver/cameras/0141_CAM_FRONT.jpg
    \________________ extraction root _______________/\_________ fixed by the archive _________/

**If `<root>` differs from `/opt/dlami/nvme/rod/datasets/hugsim_oracle`, symlink rather than
rewrite.** The layout under the root is identical, so one symlink fixes every path in every set:

    mkdir -p /opt/dlami/nvme/rod/datasets
    ln -s <root> /opt/dlami/nvme/rod/datasets/hugsim_oracle

Rewriting the JSONs with `sed` also works but touches ~29k files across four sets and has to be
redone for every rebuild. Verify either way before launching — a missing frame is not a crash, the
builder already resolved these paths and training will simply fail to open them:

    python - <<'EOF'
    import json, glob, os, random
    d='/mnt/efs/users/rod/student_runs/sft_data/autovla_hugsim_pdmclosed_executed5s_train'
    bad=0
    for f in random.sample(glob.glob(d+'/*.json'), 200):
        s=json.load(open(f))
        for k in s:
            if k.endswith('_camera_paths'):
                bad += sum(not os.path.exists(p) for p in s[k])
    print('missing frames in 200 sampled examples:', bad)   # must be 0
    EOF

On this node the frames are already extracted and the paths resolve; `$ROD/drive_staging/` also
holds a durable copy of the PDM-Closed archive with its `.sha256`. **NVMe is EPHEMERAL** — it is
wiped when the instance stops — so on a fresh box assume the frames are gone and start here.

### 2. The SFT sets (9.4 MB)

    $ROD/student_runs/sft_data/autovla_hugsim_{pdmclosed,oraclefuture}_executed5s{,_train,_val}
    $ROD/student_runs/sft_data/autovla_hugsim_{pdmclosed,oraclefuture}_cotmatched_executed5s{,_train,_val}

Also on Drive as `agentic driver assets/autovla_planner_sft_sets.tar.gz` (+ `.README.md`), which is
what to use if `$ROD` is not mounted on the B200 box.

Do NOT use the older `*_cot_executed5s` sets (11,451 / 9,326 examples). They were built by a builder
that does not apply HUGSIM's per-step nc/dac check, so they contain 3,859 / 1,879 steps whose own
plan scored a predicted collision or off-road. The `_cotmatched_` sets replace them.

> Note there is a third archive on Drive, `oracle_predict_hd.tar.zst` (20.9 GB). **Not in scope** —
> we are doing oracle-future and PDM-Closed only.

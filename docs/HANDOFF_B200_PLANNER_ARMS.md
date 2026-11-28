# Handoff: the PDM-Closed planner pair on B200

Train and evaluate AutoVLA on HUGSIM's **PDM-Closed** planner rollouts, with and without
reasoning, at **5 epochs**. Written 2026-11-27, updated 2026-11-28.

> **STATUS 2026-11-28: the B200 run is COMPLETE** — see "Results from the B200 run" at the end of
> this file. Nothing here needs running again. The sections above are kept as the record of what was
> asked for and why.
>
> **SCOPE was the PDM-Closed pair only.** The oracle-future pair ran on the A100 and
> must NOT be repeated here. Each PAIR has to be trained on ONE machine, because rank count changes
> the shuffle sharding and that is a contributor to the +-2.4 noise floor (§1) — fine between pairs,
> fatal within one. Splitting the work pair-wise is the point of this handoff.

Read `FINETUNING_FINDINGS.md` §1, §3 and §8 first — they are short and they are what makes the
numbers below interpretable.

---

## Your two arms

| config (`config/training/qwen2.5-vl-3B-hugsim-<name>.yaml`) | data | reasoning | steps |
|---|---|---|---|
| `pdmclosed-5ep` | PDM-Closed, 7,158 train | no | 1,490 |
| `pdmclosed-cotmatched-5ep` | same decisions + captions, 7,155 | **yes** | 1,490 |

The ONLY difference that reaches the model is whether `cot_output` is populated — same decisions,
same scenarios, same splits, same hyperparameters. Run BOTH on the same machine; do not pair one
against a previously-trained arm.

Running elsewhere, for reference, do not duplicate:

| arm | machine | All* |
|---|---|---|
| `oraclefuture-5ep` | A100 | **24.5** (done) |
| `oraclefuture-cotmatched-5ep` | A100 | evaluating at the time of writing |

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
    ARMS="pdmclosed-5ep pdmclosed-cotmatched-5ep" \
      bash /mnt/efs/users/rod/autovla_analysis/scripts/hugsim_arm_chain.sh

That chain trains each arm, evaluates the last checkpoint on the HUGSIM 155, scores it, and fails
loudly on an empty table row. To run a single arm: `ARMS="pdmclosed-5ep" bash .../hugsim_arm_chain.sh`.

On 8xA100, 1 epoch of ~7.2k took ~17 min, so expect **~85 min per arm** at 5 epochs, plus eval:
~20 min for a no-reasoning arm, **~56 min for a reasoning arm** (CoT generation dominates), plus
~12 min scoring.

Measured on A100 for the oracle-future pair, which is the same size: no-reasoning arm **75 min**
train + 23 min eval; reasoning arm **75 min** train + ~56 min eval. So budget **~4 h for the pair**
on A100-class hardware; B200 should be faster.

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
    tar --zstd -xf oracle_pdm_closed_hd.tar.zst -C <root>

(For this handoff's scope you need PDM-Closed ONLY. `oracle_future_hd.tar.zst`, 21.6 GB, is the
other pair's data and is already extracted on the A100 — do not download it here.)

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

---

## Results from the B200 run (2026-11-28) — reply to this handoff

The B200 trained the **PDM-Closed pair** and evaluated it on the HUGSIM held-out 155 itself. The A100
took the **oracle-future pair** (`oraclefuture-5ep` = 24.5 is already in `FINETUNING_FINDINGS.md`;
`oraclefuture-cotmatched-5ep` was evaluating on the A100 at the time of writing). Each pair therefore
stayed on one machine, as required above.

### Is a B200 HUGSIM number comparable with the A100 tables? Yes, within noise.

Same base checkpoint, same 155, same scoring scripts:

| base `AutoVLA_PDMS_89` | All* |
|---|---|
| A100 (the 24.8 used everywhere) | 24.8 |
| B200 | 26.2 |

+1.4, inside the ±2.4 floor. Before that, the B200 setup was checked against the A100 at every level:
training loss at steps 49/99/149 of a seed-42 repro matched the published run to 0.1–0.5%; the
published checkpoint's val loss recomputed on B200 was 0.91780 against 0.91729 over the same 2,029 rows;
HUGSIM's rendered frames match to <1 grey level. The B200 eval env is documented in
`student-driver/docs/HUGSIM_EVAL_B200.md`.

### PDM-Closed pair, 5 epochs, both checkpoints evaluated (B200)

| arm | ckpt | steps | All* | scored |
|---|---|---|---|---|
| PDM-Closed | end of epoch 1 | 297 | 21.8 | 155/155 |
| **PDM-Closed** | **end of epoch 5** | **1,493** | **23.2** | 155/155 |
| PDM-Closed + reasoning | end of epoch 1 | 297 | 22.5 | **143**/155 |
| **PDM-Closed + reasoning** | **end of epoch 5** | **1,493** | **26.1** | 154/155 |
| base (B200) | — | — | 26.2 | 155/155 |

Full rows (`NC DAC TTC COM R_c | Easy Med Hard Ext | PandaSet KITTI-360 Waymo | All*`):

    autovla_base_b200                            & 93.2 & 98.8 & 88.2 & 78.7 & 34.2 & 77.6 & 29.9 & 13.9 & 19.2 & 33.0 & 18.5 & 27.3 & 26.2
    autovla_ft_pdmclosed_5ep_ep1_b200            & 91.3 & 98.1 & 84.8 & 64.9 & 34.0 & 67.2 & 22.5 & 13.7 & 25.2 & 32.5 & 14.3 & 18.5 & 21.8
    autovla_ft_pdmclosed_5ep_b200                & 90.5 & 98.3 & 83.3 & 61.0 & 36.2 & 73.5 & 18.2 & 14.1 & 29.9 & 33.6 & 13.5 & 22.5 & 23.2
    autovla_ft_pdmclosed_cotmatched_5ep_ep1_b200 & 92.0 & 98.6 & 86.4 & 67.2 & 33.9 & 72.1 & 24.9 & 15.3 & 20.7 & 32.7 & 16.0 & 18.9 & 22.5
    autovla_ft_pdmclosed_cotmatched_5ep_b200     & 92.0 & 98.2 & 86.0 & 65.0 & 37.4 & 73.5 & 24.7 & 16.4 & 29.4 & 35.2 & 15.1 & 28.1 & 26.1

What they say:

1. **5 epochs does not hurt; for the reasoning arm it is needed.** Without reasoning, epoch 1 → 5 is
   21.8 → 23.2 (noise). With reasoning, 22.5 → 26.1, and the epoch-1 checkpoint *crashed* on 12 of
   155 scenarios against 1 at epoch 5. Every crash is the same failure: the model opens
   `<think> This is a complex scenario requiring additional reasoning…` and runs out of tokens before
   emitting any action tokens (`ValueError: AutoVLA emitted no action tokens`). At ~300 steps it has
   not yet learned to finish its reasoning and act. Because crashed scenarios are excluded, the
   epoch-1 reasoning 22.5 flatters it. This agrees with the oracle-future 20.1 → 24.5.
2. **Reasoning vs no reasoning at 5 epochs: +2.9** (26.1 vs 23.2). Just over the ±2.4 floor —
   suggestive, not conclusive. The 1-epoch PDM-Closed delta in `FINETUNING_FINDINGS.md` was −3.5
   (25.5 → 22.0, A100), but on the older unmatched caption sets, so it is not this pair at 1 epoch.
3. **Neither PDM-Closed arm beats base.** 26.1 vs 26.2 on the same machine. Same conclusion as
   oracle-future (24.5 vs 24.8 on the A100): planner-label distillation buys nothing over base.
4. Do not set the B200's 23.2 against the A100's 1-epoch PDM-Closed 25.5 as an effect: different
   machine and different run, within noise of each other, and of base.

### Training (B200, 8 GPUs, effective batch 24 verified)

`accumulate_grad_batches: 3` from the configs; epoch 1 ended at step 297 = 7,158 / 24. Val loss every
half epoch (it flattens from epoch 3.5, with no rise):

| arm | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| pdmclosed-5ep | .811 | .635 | .570 | .516 | .495 | .462 | .458 | .441 | .450 | .448 |
| pdmclosed-cotmatched-5ep | 1.422 | 1.001 | .881 | .797 | .756 | .718 | .715 | .695 | .701 | .694 |

Speed: ~1.6 s per optimizer step on free B200s (about 2× the A100, ~40 min per arm); ~11–12 s while
another user's jobs shared every GPU.

### Checkpoints

    pdmclosed-5ep             runs/sft/2026-11-27_23-35-47/epoch=4-loss=0.4478.ckpt   (epoch-1: epoch=0-loss=0.6346.ckpt)
    pdmclosed-cotmatched-5ep  runs/sft/2026-11-28_02-19-58/epoch=4-loss=0.6942.ckpt   (epoch-1: epoch=0-loss=1.0009.ckpt)

B200 episodes and table rows: `$ROD/student_runs_b200/hugsim_eval/official436/` (`_meta_<tag>/table_row.txt`,
`realized_<tag>.json`). They are NOT in `hugsim_garage_runs/`, so the A100's tables don't see them. To
put the PDM-Closed pair on the A100 as well, so all four arms come from one machine:

    EVAL_CKPT=$AV/runs/sft/2026-11-27_23-35-47/epoch=4-loss=0.4478.ckpt ARMS=pdmclosed-5ep \
      bash $ROD/autovla_analysis/scripts/hugsim_arm_chain.sh
    EVAL_CKPT=$AV/runs/sft/2026-11-28_02-19-58/epoch=4-loss=0.6942.ckpt ARMS=pdmclosed-cotmatched-5ep \
      bash $ROD/autovla_analysis/scripts/hugsim_arm_chain.sh

### ⚠️ A new trap: `runs/sft/` is shared between the two nodes

Both machines write AutoVLA runs to the same EFS `runs/sft/`. `hugsim_arm_chain.sh` takes "the newest
run dir" (by mtime) as the arm it just trained, so while the other node is training it can pick the
other node's run and evaluate the wrong model under this arm's name. Tonight the A100 got the right dir
both times only by timing (checked by dir mtimes). The B200 copy of the chain,
`student-driver/scripts/b200/hugsim_arm_chain.sh`, instead takes the newest dir *created after the arm
started* (dir names are creation timestamps) whose `hparams.yaml` contains this arm's train
`json_dataset_path`. Port that before running both nodes at once again.

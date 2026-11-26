# Fine-tuning AutoVLA: what we measured

Findings from distilling simulator rollouts into `AutoVLA_PDMS_89` on HUGSIM and Alpasim,
2026-11-20 to 2026-11-26. These are properties of **this model and this codebase** — the
optimiser it wants, the limits of its action codebook, and two bugs in the training path.
Where the data lives is recorded separately, in `$ROD/README_runs_locations.md`.

Scores below are HUGSIM closed-loop `hdscore_realized` (the **driven** trajectory), column
`All*` = the mean of the three per-dataset means over the 155-scenario held-out half of
`nuscenes-train-wk-80-20-panda-eval-byscene.yaml`. Not HUGSIM's native `hdscore`, which
scores the *planned* trajectory and differs a lot for a model that emits a 5 s plan.

---

## 1. There is a ±2.4 point noise floor. Read every single number against it.

`conv3ep1` and the published `autovla_ft` are the **same recipe on the same data** —
effective batch 24, lr 2e-5 held flat, 1 epoch, same by-scenario split. They differ only in
running on 8 GPUs versus 6, which changes how `DistributedSampler` shards the shuffle.

| | All* |
|---|---|
| `autovla_ft` (published, 6 GPUs) | 38.6 |
| `conv3ep1` (same recipe, 8 GPUs) | 41.0 |

**2.4 points of pure run-to-run variance**, which is three times the spread across epochs 1–3
of a single run. At n=1 on 155 scenarios, differences under ~2.5 points are not
distinguishable from noise. Per-difficulty columns are worse — `Hard` swings 16.4 / 9.4 / 13.3
across one run's own epochs, because it is 39 scenarios.

## 2. ReCogDrive's optimiser makes AutoVLA worse. Do not "match for consistency".

Matching a comparison model's hyperparameters sounds fair and is not. We ran it:
`rcdmatch` is our teacher data under ReCogDrive's recorded recipe, read off
`official436/_meta_recogdrive_vlm_ft_teacher_rewind/vlm_checkpoint_overlay/{provenance.json,trainer_state.json}`
— 1 epoch, global batch 128, lr 4e-5, 10% warmup then cosine to exactly 0, 316 steps. Our
schedule reproduces theirs within ~1 pp at every point of the run (logged `lr-AdamW`
3.9196e-05 at step 49 against a predicted 3.9190e-05).

| | recipe | All* |
|---|---|---|
| `autovla_ft_rcdmatch` | batch 128, lr 4e-5 cosine | **31.5** |
| `autovla_ft` | batch 24, lr 2e-5 flat | 38.6 |
| `conv3ep1` | batch 24, lr 2e-5 flat | 41.0 |

**Cost: 7–10 points**, far outside the noise floor. The mechanism is almost certainly step
count, not the schedule: one epoch at batch 128 is **269 optimizer steps** where batch 24
gives **1,435**, and the cosine spends the last third of those few steps at almost no
learning rate. ReCogDrive tolerates 316 steps; AutoVLA evidently needs the updates.

Consequence for experiment design: hold the **data** constant across students (same rollouts,
same filters, same split, same eval) and let each model have an optimiser that suits it.
Forcing a shared optimiser attributes to architecture a gap that is partly a hyperparameter
choice favouring the other model.

## 3. One epoch is enough, and val loss does not track the closed-loop score.

`3ep-flat`: the published arm's optimiser extended to 3 epochs, LR held **flat** on purpose
(`lr_step_frequency: 1000000`). Under a cosine, "epoch 2 beats epoch 1" would conflate more
training with a lower LR, and the end-of-run flattening is guaranteed by construction rather
than evidence of anything.

| end of epoch | All* | val loss |
|---|---|---|
| 1 | 41.0 | 0.9158 |
| 2 | 41.8 | 0.8264 |
| 3 | 41.1 | 0.8140 |

The score moves 0.8 points with no trend while held-out loss falls 11%. `rcdmatch` makes the
same point from the other side: its val loss fell monotonically to 1.0437 and it scored 31.5.
**Imitation loss on held-out teacher tokens is not the closed-loop objective and must not be
used to pick a checkpoint.**

Also note the per-step `train_loss` in the Lightning CSV is a **single micro-batch of 8
samples** (`models/autovla.py`, `batch_size: 1` per rank, `sync_dist=True`), written every 50
steps. It is far too noisy to read as a curve. Two AutoVLA runs correlate at r = 0.983 on it
only because `seed_everything(42)` gives them the same shuffle, so the same hard rows land at
the same fraction of the epoch. HuggingFace's Trainer logs the mean over the logging interval
at global batch 128, which is why a ReCogDrive curve looks smooth next to ours — that is a
logging difference, not a training one.

## 4. The action codebook has a hard speed ceiling, and no vocabulary for a fast turn.

`codebook_cache/agent_vocab.pkl` → `['token_all']['veh']`, shape `(2048, 6, 4, 2)`. Taking
each primitive's centre displacement over its 0.5 s:

```
max implied speed   31.34 m/s      p99 24.68      median 6.56
above 25 m/s        20 / 2048  (0.98%)
above 15 m/s        277
of the 188 primitives above 17 m/s, ALL 188 are straight (|heading change| < 2 deg)
```

So a **fast turn is unrepresentable**, and above 31.34 m/s nothing is. Measured quantisation
floor per training set (`$ROD/autovla_analysis/scripts/codebook_quantisation_error.py`, using
the dataset's own `TrajectoryTargetBuilder`, so these are exactly the labels training saw):

| set | ADE floor | FDE floor | rows >1 m FDE |
|---|---|---|---|
| HUGSIM teacher | 0.045 m | 0.036 m | 0.0% |
| Alpasim teacher (`texecnr_m`) | 0.145 m | 0.205 m | 2.6% |
| Alpasim logged ego (`bclog`) | 0.421 m | 0.824 m | 9.5% |

The floor is flat at ~4 cm up to 15 m/s and then collapses — `bclog` averages a 3.1 m FDE
floor above 15 m/s, worst 28 m. HUGSIM rollouts never exceed ~10 m/s so they never reach it;
Alpasim has real highway driving and **20% of sampled `bclog` rows are in the broken regime**.
`bclog` has 548 rows (3.14%) above the 31.34 m/s ceiling outright.

**Remedy if the fast regime matters:** re-cluster the codebook on the target distribution
(`tools/action_token/action_token_cluster.py`). Filtering fast rows is a workaround — it
removes degenerate labels but leaves the model unable to drive a fast curve at all.

## 5. The output format collapses — and it is a consequence of §4, not of the prompt.

Base AutoVLA completes 100/100 Alpasim val100 rollouts. Every Alpasim-fine-tuned arm crashes
6–11 outright with `ValueError: AutoVLA emitted no action tokens`: the model writes prose
where the ten action tokens belong, sometimes in a repetition loop, and often closes
`</answer>` cleanly — so these are genuine completions, not truncation.

Failures per 10,000 steps, by ego speed:

```
speed m/s    base    bclog   texec-m  texecnr-m
0-5          0           0         0          0
5-10         0         6.5         0        2.3
10-15        0         3.2         0          0
20-25        0         189       357        159
25-30        0        1000       732        556
30+          0         556       455        500
```

Base: **0 failures in 17,854 steps**, including 153 above 20 m/s. ~0.4% of eval steps are
above 25 m/s and they carry 8 of the 24 distinct failures — a ~150× enrichment, in exactly
the band where §4 says the codebook runs out. In the 30+ m/s bin, 74% of sampled `bclog` rows
have **all ten label tokens identical** — a constant, image-independent target. HUGSIM: zero
such rows.

**Mechanism.** One epoch at lr 2e-5 with the whole LM backbone trainable
(`train_lm_backbone: true`), on a set whose fast regime is supervised by a constant ten-token
string, erodes the base checkpoint's margin between `<action_*>` and the ordinary English
continuation of `"The final output action is: "` in that regime. The base checkpoint never
loses it.

**Ruled out, with counts** (`$ROD/autovla_analysis/scripts/sft_target_audit.py` rebuilds the
exact assistant target from `dataset_utils/sft_dataset.py` for 1200 rows × 5 sets and
tokenises it):

- *Prompt format / empty `cot_output` under `use_cot: true`.* 6000 of 6000 rows carry exactly
  ten action tokens. The no-CoT target is byte-identical in shape to the HUGSIM no-reason set,
  which trains fine. Each arm fails inside its own **correctly learned** template and only in
  the action slot.
- *Loss masking.* `DataCollator` masks only tokens before `<|im_start|>assistant`; the action
  span is always supervised and nothing is truncated.
- *Instruction vocabulary.* All sets use exactly `{keep forward, turn left, turn right}`, and
  the `.stats.json` sidecars record zero unmapped-command drops.

**Careful with the HUGSIM control.** HUGSIM teacher data tops out at 9.65 m/s over 36,476
rows. That HUGSIM arms completed 155/155 shows the builders and prompt plumbing are sound; it
is **not** evidence that Alpasim data is uniquely bad, because the HUGSIM arms were never
probed in the failing regime.

**Fix without retraining:** constrain the answer slot at decode time — once
`"The final output action is: "` has been emitted, mask logits to ids ≥ `ACTION_START_ID`
(151665) and require ten of them, e.g. via `prefix_allowed_tokens_fn`. Also prefer
`max_new_tokens` to `max_length`, which is a *total* budget shared with the video prompt.
This turns a hard crash into a possibly-poor plan, which is what an arms comparison wants —
but it changes eval semantics, so say so wherever the numbers are reported.

## 6. Two bugs in the training path

**a. The action-token loss is gated on reasoning.** In `models/autovla.py`, the extra
action-token cross-entropy is added only when `hascot[0]` is true. Any arm built with
`--no-cot` therefore receives **no action-token-specific loss at all**. HUGSIM's no-reasoning
arm has the same gap and still trains, so this is not the cause of §5 — but it leaves exactly
the slot that breaks without its dedicated supervision, and un-gating it is free.

**b. Training and inference feed different video resolutions.** `dataset_utils/sft_dataset.py`
hardcodes `min_pixels = max_pixels = 28*28*128 = 100352` in all three prompt-construction
branches and **ignores** the configs' `model.video.{min,max}_pixels: 109760`, which the eval
driver does use. A real train/test mismatch, affecting both simulators equally.

> **Not fixed as of 2026-11-26, deliberately.** Every arm trained to date used 100352, so
> changing it breaks comparability with every published row. Fix it together with a re-run of
> the baseline arms, not alone.

## 7. `cot_loss_scale`

Upstream multiplies a reasoning row's loss by 40 so that scarce CoT data is not drowned out.
In our difficulty-gated sets ~75% of rows carry reasoning, so that factor does the opposite of
its purpose: with `gradient_clip_val=1.0` clipping **by value**, every CoT step saturates the
clip and the easy rows — the whole point of gating reasoning on difficulty — barely train.
All our configs set `cot_loss_scale: 1.0`. Set it back to 40 only to reproduce upstream.

## 8. Practical notes

- `scripts/run_autovla_hugsim.sh` runs the **sweep only — it does not score**. Scoring is
  `simulators/hugsim_garage/score_realized.py` then `scripts/mila/paper_table_row.py`. A chain
  that logs "SCORED" can still have written a 0-byte `table_row.txt`; check it is non-empty.
- `AGENT` must differ per checkpoint. `run_hugsim_sweep.sh` skips any scenario that already
  has an `eval.json`, so re-running under a finished run's name evaluates **nothing** and
  silently reports the old result.
- At `val_check_interval: 0.25`, all four of an epoch's checkpoints carry the same `epoch=`
  index in the filename and two can tie on the loss. Identify end-of-epoch checkpoints by
  **mtime**, not by the filename.
- `every_n_epochs=1` (the stock `ModelCheckpoint` default) silently defeats a sub-epoch
  `val_check_interval`: you validate 12 times and keep 3 epoch-end files. `tools/run_sft.py`
  now takes `checkpoint_every_n_epochs` from the config; set it to `null`.
- The step decay is usually a **no-op** at these run lengths: `lr_step_frequency` is 2000 and
  these runs are a few hundred to ~1400 steps, so `gamma ** 0 == 1` and the LR is constant
  after warmup. Do not read such a run as "decayed".

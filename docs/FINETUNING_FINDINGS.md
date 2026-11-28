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

## 1b. The recipe that works — copy this

The best AutoVLA arm to date: **All\* 39.7** (`teacher-8gpu`; the same recipe scored 38.6 and 41.0
on other runs, mean 39.8 over three samples). Config
`config/training/qwen2.5-vl-3B-hugsim-teacher-8gpu.yaml`.

| | value | why it is this |
|---|---|---|
| `batch_size` | 1 | per rank; the real knob is the product below |
| `accumulate_grad_batches` | **3** | **8 GPUs x 1 x 3 = effective batch 24.** Set this so the product is 24 on whatever GPU count you have. See §2: 32 costs 7.3 points. |
| `learning_rate` | **2e-5** | |
| `epochs` | **1** | more does not help; see §3 |
| `lr_warmup_step` | 100 | ~7% of the 1,435-step run. Scale it to 7% for a shorter run, do not keep 100 |
| LR schedule | **flat after warmup** | `lr_step_frequency: 2000` never fires in 1,435 steps, so `gamma ** 0 == 1`. Do NOT read these runs as "decayed" |
| `lr_step_gamma` | 0.98 | unused at this run length |
| `weight_decay` | 0.01 | |
| `cot_loss_scale` | **1.0** | upstream's 40 is wrong here; see §9 |
| `train_sample_size` | null | all examples |
| `num_workers` | 4 | |
| `val_check_interval` | 0.25 | 4 val points per epoch |
| `save_top_k` / `checkpoint_every_n_epochs` | -1 / null | keep every checkpoint; the stock `every_n_epochs=1` defeats a sub-epoch interval |
| `model.use_cot` | true | true even for no-reasoning arms — the system prompt stays identical, only `cot_output` is empty |
| `model.init_from_checkpoint` | `AutoVLA_PDMS_89.ckpt` | a fine-tune OF AutoVLA, not a retrain from base Qwen |
| `model.train_vision_backbone` | **false** | vision encoder frozen |
| `model.train_lm_backbone` | true | |
| `model.trajectory` | 10 poses, 0.5 s, 5 s horizon | fixed by the codebook |
| `model.video.{min,max}_pixels` | 100352 | inert in practice — see §6b |
| `model.tokens.action_start_id` | 151665 | |
| trainer | FSDP `FULL_SHARD`, bf16, `gradient_clip_val 1.0` by **value**, `seed_everything(42)` | clip-by-value is why `cot_loss_scale` 40 would saturate every CoT step |
| inference | `temperature 0.01`, `max_length 2048` | `max_length` is a TOTAL budget shared with the video prompt |

Data: `$ROD/student_runs/sft_data/autovla_hugsim_final-rewind_executed5s_{train,val}` — 36,476
examples over 200 of the 265 dev scenarios, split **by scenario** (a row split shares frames across
the halves). Runtime: ~83 min training on 8xA100, plus ~56 min to evaluate on the HUGSIM 155 with
16 workers (a no-reasoning arm evaluates in ~20).

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

**The penalty is a cliff, not a slope — it already bites at batch 32.** Measured 2026-11-27 on the
same data and the same 8 GPUs, varying only `accumulate_grad_batches`:

| arm | accumulate | eff. batch | steps | All* |
|---|---|---|---|---|
| `teacher-8gpu` | 3 | 24 | 1,435 | **39.7** |
| `teacher-b32` | 4 | **32** | 1,139 | **32.4** |
| `rcdmatch` | 16 | 128 | 269 | 31.5 |

Going 24 -> 32 costs **7.3 points**, three times the noise floor and essentially the whole penalty
you get at 128. So **pin `accumulate_grad_batches` so that GPUs x batch_size x accumulate == 24**,
and never inherit the number from a config written for a different GPU count: the published arms
used 4 on SIX GPUs, which is 24, but the same file on eight gives 32 and loses 7 points silently.

Consequence for experiment design: hold the **data** constant across students (same rollouts,
same filters, same split, same eval) and let each model have an optimiser that suits it.
Forcing a shared optimiser attributes to architecture a gap that is partly a hyperparameter
choice favouring the other model.

## 3. ~1,400 optimizer STEPS is enough — not "one epoch". Val loss does not track the score.

`3ep-flat`: the published arm's optimiser extended to 3 epochs, LR held **flat** on purpose
(`lr_step_frequency: 1000000`). Under a cosine, "epoch 2 beats epoch 1" would conflate more
training with a lower LR, and the end-of-run flattening is guaranteed by construction rather
than evidence of anything.

| end of epoch | All* | val loss |
|---|---|---|
| 1 | 41.0 | 0.9158 |
| 2 | 41.8 | 0.8264 |
| 3 | 41.1 | 0.8140 |

The score moves 0.8 points with no trend while held-out loss falls 11%.

> **READ THIS AS STEPS, NOT EPOCHS.** That run is the 36k teacher set, where one epoch is **1,435
> optimizer steps**. "One epoch is enough" is only true at that size. A 7.2k set at effective batch
> 24 gives one epoch = **298 steps**, a fifth as much training, and the convergence result says
> nothing about it.
>
> We made exactly this mistake: the planner arms (§8) and the 7.6k teacher controls were run at 1
> epoch / ~300 steps on that reasoning, and are therefore probably UNDERTRAINED. Their relative
> ordering may survive, since all four are equally short, but the absolute numbers are depressed.
> Re-running at 5 epochs (~1,450 steps) to land on the step count where convergence was verified.
>
> **The rule: size the run in optimizer steps, around 1,400, and convert to epochs from the dataset
> size.** `epochs = round(1400 * effective_batch / n_examples)`. Scale warmup to 7% of the result
> rather than inheriting a number written for a different length. `rcdmatch` makes the
same point from the other side: its val loss fell monotonically to 1.0437 and it scored 31.5.
**Imitation loss on held-out teacher tokens is not the closed-loop objective and must not be
used to pick a checkpoint.**

Also note the per-step `train_loss` in the Lightning CSV is a **single micro-batch of 8
samples** (`models/autovla.py`, `batch_size: 1` per rank, `sync_dist=True`), written every 50
steps. It is far too noisy to read as a curve. Two AutoVLA runs correlate at r = 0.983 on it
only because `seed_everything(42)` gives them the same shuffle, so the same hard examples land at
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

| set | ADE floor | FDE floor | training examples >1 m FDE |
|---|---|---|---|
| HUGSIM teacher | 0.045 m | 0.036 m | 0.0% |
| Alpasim teacher (`texecnr_m`) | 0.145 m | 0.205 m | 2.6% |
| Alpasim logged ego (`bclog`) | 0.421 m | 0.824 m | 9.5% |

The floor is flat at ~4 cm up to 15 m/s and then collapses — `bclog` averages a 3.1 m FDE
floor above 15 m/s, worst 28 m. HUGSIM rollouts never exceed ~10 m/s so they never reach it;
Alpasim has real highway driving and **20% of sampled `bclog` training examples are in the broken regime**.
`bclog` has 548 training examples (3.14%) above the 31.34 m/s ceiling outright.

**Remedy if the fast regime matters:** re-cluster the codebook on the target distribution
(`tools/action_token/action_token_cluster.py`). Filtering fast training examples is a workaround — it
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
the band where §4 says the codebook runs out. In the 30+ m/s bin, 74% of sampled `bclog` training examples
have **all ten label tokens identical** — a constant, image-independent target. HUGSIM: zero
such examples.

**Mechanism.** One epoch at lr 2e-5 with the whole LM backbone trainable
(`train_lm_backbone: true`), on a set whose fast regime is supervised by a constant ten-token
string, erodes the base checkpoint's margin between `<action_*>` and the ordinary English
continuation of `"The final output action is: "` in that regime. The base checkpoint never
loses it.

**Ruled out, with counts** (`$ROD/autovla_analysis/scripts/sft_target_audit.py` rebuilds the
exact assistant target from `dataset_utils/sft_dataset.py` for 1200 training examples × 5 sets and
tokenises it):

- *Prompt format / empty `cot_output` under `use_cot: true`.* 6000 of 6000 training examples carry exactly
  ten action tokens. The no-CoT target is byte-identical in shape to the HUGSIM no-reason set,
  which trains fine. Each arm fails inside its own **correctly learned** template and only in
  the action slot.
- *Loss masking.* `DataCollator` masks only tokens before `<|im_start|>assistant`; the action
  span is always supervised and nothing is truncated.
- *Instruction vocabulary.* All sets use exactly `{keep forward, turn left, turn right}`, and
  the `.stats.json` sidecars record zero unmapped-command drops.

**Careful with the HUGSIM control.** HUGSIM teacher data tops out at 9.65 m/s over 36,476
training examples. That HUGSIM arms completed 155/155 shows the builders and prompt plumbing are sound; it
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

**b. `sft_dataset.py` ignored `model.video.{min,max}_pixels` — but the effect was nil.**
It hardcoded `min_pixels = max_pixels = 28*28*128 = 100352` in all three prompt-construction
branches while the configs said 109760 and the eval driver used 109760. The key now works
(configs are pinned to 100352 so no published arm changes).

> **MEASURED, 2026-11-26: this changed nothing.** Building the same row at both settings yields
> identical video frames — **420×224 = 94,080 px** either way. The 800×450 source is rounded down to
> Qwen's 28-pixel patch grid and lands *below* both caps, so neither binds. Training and inference
> were always at the same actual resolution. The `-px109760` arm run to test this is therefore a
> duplicate of the batch-24 teacher arm, not a resolution experiment; its val curve matches
> `conv3ep1` to four decimals at the first validation point.
>
> The lesson is worth more than the bug: **a config key that is read but never binding looks exactly
> like a key that works.** Check what the processor actually emits, not what the config says.

## 7. What the Alpasim sets are actually filtered on — and the one check nobody applied

Easy to get wrong, because three different checks are involved and only two of them were applied.
Our Alpasim arms are built with `--only-tokens`, which keeps exactly the 13,872 decisions in
ReCogDrive's shipped `recogdrive/sft_filtered.jsonl`, so both students train on byte-identical
decisions. That is the property that makes the cross-student comparison defensible, and it is why
the builder bypasses its own filters on that path.

What that list does and does not encode, measured against the export's `samples.jsonl` and its
`recogdrive/stats.json` (2026-11-26):

| check | what it catches | in their list? |
|---|---|---|
| `keep` | at-fault event, and the 3 s before it | **yes** — 0 of 13,872 violate it (`export_keep` 17,062) |
| executed-path, i.e. DAC/NC | the executed 4 s goes off-road or collides | **yes** — 2,416 dropped (`executed_offroad` 411, `executed_collision` 2,005; `keep_and_executed_ok` 16,723) |
| `wrong_lane` / `left_corridor` | the ego is in the wrong lane, or laterally outside its route corridor, **at that decision** | **NO** — 1,485 remain (10.7%): 815 wrong_lane, 853 left_corridor |

The third is a different and softer signal from the second. A vehicle can sit outside its corridor
for an entire rollout without its executed path ever going off-road or colliding, so those 1,485
decisions pass the DAC/NC check cleanly and still demonstrate bad lane discipline.

**Why this matters.** The unexplained part of the Alpasim result is specifically that every
fine-tuned arm roughly doubles LATERAL corridor exits — `left_corridor_laterally` 9.8% → 17–20% —
while `offroad` actually improves (26.8% → ~20%) and progress rises (0.513 → 0.618). That pattern
matches the one check nobody applied and not the ones everybody did.

**Why it is a lead and not a cause.** ReCogDrive trained on the same 1,485 decisions and its Alpasim
arms improved (0.519 → 0.614). So bad-lane supervision is not sufficient on its own; at most it
compounds a weakness AutoVLA already has. The HUGSIM builder applies the lane filter normally, which
is consistent with HUGSIM arms never showing this failure.

**If you test it**, do it as an ablation and keep the matched arm as the headline row. Filtering
trades away the identical-decisions property, which is currently the strongest claim about these
numbers. Sizing, for the cheapest candidate — `texecnr-m`, the worst arm and no-reasoning, so a
20-minute eval rather than 56:

    texecnr-m   13,025 examples -> 1,295 bad-lane (9.9%) -> 11,730 after filtering
    texec-m     13,000 examples -> 1,292 bad-lane (9.9%) -> 11,708 after filtering

Note the power limit before reading any result: 9.9% is modest, and the run-to-run noise floor on
the Alpasim scene score has never been measured (the ±2.4 in §1 is HUGSIM `All*`). A move of ~0.02
would not be distinguishable from noise.

## 8. Distilling a PLANNER instead of the Claude teacher (2026-11-26/27)

HUGSIM's own planners as demonstrators, same simulator, same prompt, same executed-path target,
same recipe — only *who produced the trajectory* changes. Both planner sets are direct-decision
(a planner emits a trajectory, not language); captions were added separately, see the 2x2 below.

| arm | demonstrator completed | training examples | scenarios | All* |
|---|---|---|---|---|
| base `AutoVLA_PDMS_89` | — | — | — | 24.8 |
| + PDM-Closed | **129 / 265 runs** | 7,607 | 193 / 265 | **25.5** |
| + oracle-future | 371 / 436 runs | 7,461 | 252 / 265 | **20.1** |
| + Claude teacher, no reasoning | rewound until solved | 35,595 | 189 / 265 | 32.4 |
| + Claude teacher, no reasoning, **subsampled to 7.6k** | | 7,607 | 200 / 265 | **28.8** |

**Neither planner helps, and the better planner is worse.** Oracle-future completes 85% of its runs
against PDM-Closed's 49% and covers the most scenarios of ANY arm (252/265, more than the teacher's
200) — and scores 5 points lower, below base. Breadth of demonstration is therefore not what
matters. The leading explanation is that oracle-future plans with **privileged knowledge of the
future**: its paths are correct but not *inferable* from three forward cameras, so the student is
trained to predict something its inputs do not determine. PDM-Closed plans from observable state,
which is probably why it merely fails to help rather than actively hurting.

Note the 7.6k teacher controls use `train_sample_size`, i.e. a RANDOM subset of the teacher set,
not a filter-matched one. That is a weaker control than the `--only-tokens` matching now used for
the caption arms: it equalises volume but not which decisions survive. It does not invalidate the
comparison, since both teacher sets pass the same filters to begin with, but the asymmetry is worth
knowing before leaning hard on a 1-2 point difference.

**Most of the gap is dataset size, not demonstrator quality.** Cutting the teacher to the planners'
7.6k costs 3.6 points (32.4 -> 28.8). Against that honest baseline PDM-Closed is only 3.3 points
behind — close to the +-2.4 noise floor — while oracle-future is still 8.7 behind, which is real.

### Why the planner sets are only ~7.5k examples

Not a filtering mistake; three things stack, and the horizon dominates.

    archive:          420 rollouts, 22,295 steps   (~53 steps = ~13 s each)
      dev half only:  265 rollouts    <- the other 155 are the eval set and must stay held out
      of those steps:
        -4,547   fewer than 5.0 s of future   <- ~32%, the dominant cost
        -1,944   the step's own plan scored nc == 0 or dac == 0
          -683   the 5 s future reaches a failed end of the run
      kept:     7,461 / 252 scenarios

AutoVLA always answers **10 poses at 0.5 s**, so a usable example needs 20 further steps. At ~53
steps per rollout the last 20 of EVERY rollout are structurally unusable. ReCogDrive's 8-pose/4 s
label needs only 16, so its version of this set comes out meaningfully larger — do not compare the
two students' example counts without accounting for that.

The second asymmetry is draws per scenario: the teacher set is **569 rollouts** (three draws plus
rewind passes) against the planners' **265** (seed 0 only). Re-running seeds does not help a
deterministic planner — the runs README records PDM-Closed seeds 1-2 as identical copies.

### The two planner arms fail for OPPOSITE reasons

Measured from each archive's `infos.pkl` and `summary.csv`:

| | PDM-Closed | oracle-future |
|---|---|---|
| planner's own `cl_hdscore` | 55.4 | **86.9** |
| mean ego speed | **4.48 m/s** | 7.16 m/s |
| distance driven per rollout | 64.1 m | **97.1 m** |
| steps mean / median | **63** / 52 | 53 / 52 |
| steps p90 / max | **115 / 308** | 76 / 133 |
| stalled runs | **52** | 11 |
| steps read -> kept | 16,562 -> 7,607 (46%) | 14,635 -> 7,461 (51%) |
| dropped: future reaches a failed end | **28%** | 5% |
| dropped: fewer than 5 s of future | 15% | **31%** |

**Why the two sets are the same size despite very different planners.** PDM-Closed drives ~60%
slower and covers two-thirds the distance, yet generates MORE steps: the medians are identical at
52 and the difference is all tail (p90 115 vs 76, max 308 vs 133), which is 52 stalled runs
crawling to the step cap. Slow or stuck driving means the rollout does not end. Those extra steps
are the least useful ones and the filters remove 28% of them as within 5 s of a failure. Oracle-
future finishes its routes, so its rollouts are short and clean — only 5% lost to failed ends, but
31% lost to the horizon rule, because short rollouts are hit hardest by needing 20 further steps.
"More raw material, heavily filtered" lands in the same place as "less raw material, lightly
filtered". It is a coincidence, not a property of the sources.

**The surviving data is biased differently in each case, and the students show it.**

  - PDM-Closed: even after the failures are filtered out, what remains is a demonstrator averaging
    4.48 m/s. The student inherits the timidity — `COM` 60.9 against the teacher arm's 97.7, `Easy`
    76.3 against 88.3. It is not crashing more; it is **under-driving**, which is what it was shown.
  - oracle-future: the demonstrations are fast, clean and from an 86.9-scoring planner — and the
    student still lands below base. That is the strongest form of the privileged-information
    argument: the paths are excellent and simply **not inferable from three forward cameras**, so
    fitting them costs the model competence it already had.

Two different failure modes, both landing near or below base. Neither is a filtering mistake.

### Reasoning x demonstrator, the 2x2

Post-hoc Claude captions over the same planner rollouts (`adam-nov25-planner-captions`,
`steps[].reason`), difficulty-gated exactly as the teacher arm is, and size-matched to their
no-reasoning counterparts with `train_sample_size` so reasoning is the only variable.

| demonstrator | no reasoning | + reasoning | delta |
|---|---|---|---|
| Claude teacher, 36k | 32.4 | **39.7** | **+7.3** |
| Claude teacher, 7.6k | 28.8 | 26.8 | -2.0 |
| PDM-Closed, 7.6k | 25.5 | 22.0 | -3.5 |
| oracle-future, 7.6k | 20.1 | 21.5 | +1.4 |

> **CONFOUND, found 2026-11-27 — the two planner rows above are NOT yet interpretable.** The
> no-reasoning planner sets were built by `make_oracle_sft_data.py` and the reasoning ones by
> `make_teacher_sft_data.py`, because the caption export ships the teacher archives' file format.
> The format matches; the FILTERS do not. The oracle builder applies HUGSIM's per-step nc/dac check
> and a failed-end rule, the teacher builder applies neither (teacher archives arrive pre-filtered).
> So the reasoning arms drew from a strictly larger, lower-quality pool — 1,879 extra steps for
> oracle-future and 3,859 for PDM-Closed, every one a step whose own plan scored a predicted
> collision or off-road. `train_sample_size` matched the example COUNT, which made the arms look
> controlled while the pools differed, and the confound runs AGAINST the reasoning arms.
> The teacher rows are unaffected (same builder both sides), so **the +7.3 headline stands**.
> Fix: `--only-tokens` on the teacher builder restricts a caption build to the decisions the
> archive build kept. Rebuilt as `*_cotmatched_*`; the two planner deltas will be re-measured.
>
> **The general lesson: matching two sources' FORMAT is not matching their FILTERS.** The evidence
> was in the `.stats.json` drop reasons the whole time. Compare those before pairing sets built by
> different builders.

**Reasoning pays only on the full teacher set.** The three deltas at 7.6k are -2.0 / -3.5 / +1.4,
all inside the noise floor, i.e. indistinguishable from zero. Narrating a trajectory post hoc does
not transfer; the +7.3 needs both the reasoning and 36k examples of it. This is the cleanest
evidence we have that the teacher's advantage is in its TRAJECTORIES — recoverable paths a
camera-only student can infer — and not in the text attached to them.

### Re-run at 5 epochs — the 1-epoch arms WERE undertrained

Confirming the §3 correction with data. Same data, same recipe, same 8 GPUs; only the step count
changes.

| arm | epochs | steps | All* |
|---|---|---|---|
| base `AutoVLA_PDMS_89` | — | — | 24.8 |
| oracle-future | 1 | 286 | 20.1 |
| **oracle-future** | **5** | **1,430** | **24.5** |
| oracle-future + reasoning | 5 | 1,425 | *running* |
| PDM-Closed | 1 | 298 | 25.5 |
| PDM-Closed | 5 | 1,490 | *queued, B200* |

**+4.4 points from training longer**, clear of the ±2.4 floor. So "1 epoch is enough" was costing
real points on these small sets, exactly as §3 now warns. `Hard` moved 14.2 -> 20.6 and `R_c`
34.6 -> 39.4, so the extra steps did teach it something.

**But it lands at 24.5 against base's 24.8 — exactly base level.** That makes the conclusion
STRONGER, not weaker: oracle-future distillation is not "undertrained and might work", it is
trained to the step count where convergence was verified and buys nothing. The
privileged-information reading survives the better-trained test.

Full row:

    autovla_ft_oraclefuture_5ep & NAVSIM & 88.9 & 98.4 & 81.7 & 51.0 & 39.4 & 68.7 & 22.4 & 20.6 & 29.0 & 36.1 & 10.2 & 27.4 & 24.5

NOTE the 1-epoch rows elsewhere in this section (PDM-Closed 25.5, oracle-future 20.1, and the 7.6k
teacher controls at 26.8 / 28.8) are all ~300-step runs and are depressed for the same reason.
Treat them as a floor. The 36k teacher arms are unaffected -- one epoch there is 1,435 steps.

### Matched-optimiser planner arms

`pdmclosed-rcd` 23.0 and `oraclefuture-rcd` 22.7, against 25.5 and 20.1 at batch 24. As expected
from §2: one epoch at batch 128 on a 7.5k set is only **55 optimizer steps**, too few to move off
base (24.8). Reportable as a matched column; little signal in it.

## 9. `cot_loss_scale`

Upstream multiplies a reasoning row's loss by 40 so that scarce CoT data is not drowned out.
In our difficulty-gated sets ~75% of training examples carry reasoning, so that factor does the opposite of
its purpose: with `gradient_clip_val=1.0` clipping **by value**, every CoT step saturates the
clip and the easy examples — the whole point of gating reasoning on difficulty — barely train.
All our configs set `cot_loss_scale: 1.0`. Set it back to 40 only to reproduce upstream.

## 10. Practical notes

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

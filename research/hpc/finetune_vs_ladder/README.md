# finetune_vs_ladder — capacity/existence control vs the scripted eval population

Purpose: a capacity-control experiment. The best-config self-play policies
plateau against the scripted eval ladder; the open question is whether that
is a *capacity* limit of the architecture or a *training-distribution* limit
of self-play. This campaign warm-starts the SAME architecture from the
best-config iter_1200 checkpoints and trains it directly against the eval
population (each episode: one learned seat, three cadenced-fast scripted
opponents of one archetype drawn uniformly from {v2,v3,v4,v5} — exactly the
`eval/checkpoint_scripted_diagnostics.py` lineup, generalized over seats and
archetypes). One success proves the existence claim; 8 warm starts are run
for redundancy.

## Design

- **Env** (`scripted_opponent_env.py`): `ScriptedOpponentEnv` wraps the
  training env `NashHungerMultiAgentEnv` by composition — observations,
  action masks, terminal-only rewards (+1/0/-1, death delivered on the
  death step) and infos are computed by the unmodified training code; the
  wrapper only injects the three scripted seats' actions inside `step()`
  and filters everything to the focal agent. MAPPO's centralized critic
  observation is byte-identical to training time (global encoder reads the
  full game state, scripted seats included). Focal seat: uniform per
  episode (`focal_selection="rotate"` available). Scripted seeds:
  `episode_seed * 17 + seat` (the eval-harness convention). The RLlib
  episode ends when the focal seat terminates (death or horizon).
- **Warm start** (`run_finetune.py`): loads the checkpoint's RLModule
  standalone (`learner_group/learner/rl_module/shared_policy`), asserts
  the module's declared architecture/observation contract against the CLI
  flags (loud failure on mismatch), injects the state into the fresh
  algorithm's learner, syncs env runners, and runs a logits-identity check
  on a fixed probe batch before (must differ) and after (must match) the
  load. Only module weights are transferred — the optimizer restarts,
  which is intended for a distribution shift of this size.
- **Batching**: episode-matched conventions — 45 complete games per update
  at 180 s = 8100 env steps, minibatch 2025, 4 epochs, 2 env runners.
  With a single learned seat an env step carries ONE agent transition
  (self-play carried four), so an update sees ~8100 agent steps from >= 45
  games (episodes shorter than 180 when the focal seat dies early).
- **Optimization**: entropy 0.001 and lr 1e-4 held constant — the TERMINAL
  values of the best-config anneals, i.e. what the checkpoint was training
  with at iter 1200. `--entropy-anneal-to/--lr-anneal-to` switch to
  schedules over the fine-tune budget. gamma = lambda = 1, terminal-only
  reward, qty 5 / price 50 — unchanged.
- **Checkpoint numbering** continues the warm start: iter_1250, 1300, ...,
  1500 under `<HUNGER_CAMPAIGN_ROOT>/<algo>_seed<N>_headmlp/checkpoints/`.
  Chosen (over restarting at 0050) because
  `eval/checkpoint_scripted_diagnostics.py --checkpoint-start 1250
  --checkpoint-stop 1500 --checkpoint-every 50` then works unchanged, and
  plots line up on the same absolute-iteration axis as the source runs.
- **Telemetry**: `iters.jsonl` per task, mirroring `eval/long_baseline.py`
  field names, plus `focal_survival_rate`, `focal_win_rate`,
  `focal_terminal_reward`, per-archetype episode counts and per-archetype
  survival, and per-iteration wall time.

## Tasks

Array 0-7 (see `campaign.py` / `run_array.sbatch`):

| task | algo  | seed | checkpoint root          |
|------|-------|------|--------------------------|
| 0-4  | ippo  | 0-4  | 2SEED (0,1) / S234 (2-4) |
| 5    | mappo | 0    | 2SEED                    |
| 6    | mappo | 2    | S234                     |
| 7    | mappo | 3    | S234                     |

All warm starts are `default_180_mlp` arms at `iter_1200`.

## Configure, validate, submit (on Resnick)

```bash
cd /resnick/home/$USER/Hunger_games_2/research
cp hpc/finetune_vs_ladder/campaign.env.example hpc/finetune_vs_ladder/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/finetune_vs_ladder/campaign.env
source hpc/finetune_vs_ladder/campaign.env

PYTHONPATH="$HUNGER_RESEARCH_ROOT" "$HUNGER_HPC_VENV/bin/python" \
  -m pytest hpc/finetune_vs_ladder/test_campaign_plan.py -q

bash hpc/finetune_vs_ladder/submit_all.sh
```

Walltime: 12 h per task (300 iters; sampling cost per iteration is close to
the self-play campaigns' ~1.5 min/iter at the same 8100-env-step batch, and
the update is ~4x smaller).

## After training

```bash
PYTHONPATH="$HUNGER_RESEARCH_ROOT" "$HUNGER_HPC_VENV/bin/python" \
  -m eval.checkpoint_scripted_diagnostics \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" \
  --output-dir "$HUNGER_CAMPAIGN_ROOT/_scripted_eval_100/v4" \
  --algorithms ippo --seeds 0,1,2,3,4 \
  --checkpoint-start 1250 --checkpoint-stop 1500 --checkpoint-every 50 \
  --n-games 100 --opponent v4
```

(repeat per opponent / for `--algorithms mappo --seeds 0,2,3`). Compare
against the same checkpoints' iter_1200 scripted-eval scores from the
best-config campaign.

## Decisions to make at deploy time

- Entropy/lr: hold at terminal values (default) vs re-anneal
  (`--entropy-anneal-to 0.0005 --lr-anneal-to 5e-5` or similar).
- `--focal-selection rotate` for exact per-seat balance instead of uniform
  random (random is the default; over 45+ games/update the difference is
  noise).

# RLlib PPO Baseline Guide

This document explains the multi-agent PPO training stack added to the Python
research environment and how to use it.

## What was implemented

### Shared local observation encoder

Each seat receives a 171-dimensional local observation built from information
that player is legally allowed to see:

- Own cash, reserved cash, and available cash.
- Own total, reserved, and available inventory for all four foods.
- Produced-food indicators.
- Public top-of-book data.
- Public opponent alive/production information.
- The player's own open orders.
- Public trade matrices from the previous tick.
- Elapsed-game fraction.

The encoder supports the current rule where every player produces two foods.
It lives in `env/rl/observation_encoder.py`.

### Simultaneous market clearing

The research simulator now defaults to simultaneous order clearing. On each
environment step, all seats submit one action, valid trade/cancel updates are
applied to the books, and then crossed books clear before production,
consumption, deaths, and end checks advance the clock.

This means each tick behaves like everyone submitting quotes into the same
market-clearing batch rather than lower-index seats trading before later seats
act. Execution priority is deterministic from the environment seed: new
same-tick orders receive seeded sequence priority, foods clear in seeded order,
and crossed books use price-time priority with self-trade prevention.

The legacy immediate-matching path is still available for compatibility with
the TypeScript/live simulator by constructing a config with:

```python
GameConfig(simultaneous_order_clearing=False)
```

### Centralized critic state

The critic receives a 305-dimensional vector generated directly from the
complete `GameState`. It does not reconstruct state by combining partial player
observations.

For the player whose value is being estimated, all players are rotated so that
the target player is always slot 0. The vector contains:

- Four players × 20 features:
  - Alive indicator.
  - Four produced-food indicators.
  - Cash, reserved cash, and available cash.
  - Total inventory A-D.
  - Reserved inventory A-D.
  - Available inventory A-D.
- Every possible active order slot:
  - Player × food × bid/ask = 32 fixed slots.
  - Existence, price, original quantity, remaining quantity, partial-fill
    status, creation time, and sequence priority.
- Elapsed-game fraction.

The critic therefore predicts the target player's expected discounted future
shaped reward using information from the entire game:

```text
V(global_state, target_player)
    ≈ E[sum gamma^k * target_player_reward]
```

The actor never receives this global state.

### RLlib multi-agent environment

`NashHungerMultiAgentEnv` exposes four RLlib agents:

```text
seat_0, seat_1, seat_2, seat_3
```

Each active seat receives:

```python
{
    "observations": float32[171],
    "action_mask": float32[410],
    "critic_observations": float32[305],
}
```

All four seats map to one shared policy. Dead seats receive one terminal
transition and are then removed from subsequent action requests. The underlying
`FoodTradingEnv` remains the sole authority for game rules.

### Reward calculation

The Gymnasium and RLlib adapters use the same stateful reward tracker:

```text
+ 0.01 × min(available required food) while alive
+ 5.00 / k for winning, where k is tied winners
+ 1.00 for surviving to game end without winning
```

The dense survival reward uses the uncapped minimum available inventory across
the player's required foods. It intentionally ignores cash and total food, so
policies are rewarded for maintaining a balanced buffer of the resources that
directly prevent starvation. There are no dense negative death or starvation
penalties; a dead seat simply stops receiving the positive survival reward.

The final evaluation statistics remain based on real game results rather than
only shaped return.

### Shared actor and centralized critic

The custom PyTorch RLModule supports two actor types:

- `mlp`: feed-forward local policy.
- `lstm`: recurrent local policy with independent hidden state for each seat.

Both use:

- Local 171D actor input.
- Legal-action masking over 410 discrete actions.
- A separate feed-forward critic using the 305D complete state.

Default model sizes are constants at the top of
`env/rl/rllib_module.py`:

```python
ACTOR_HIDDEN_SIZE = 256
CRITIC_HIDDEN_SIZES = [512, 512, 256]
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
MAX_SEQ_LEN = 64
```

`NashHungerPPOTorchRLModule` delegates network logic to top-level architecture
constants: `ACTOR_ARCHITECTURE = LSTMActor` and
`CRITIC_ARCHITECTURE = MLPCritic`. The actor outputs final 410-action logits
directly. The critic outputs final scalar values directly from the 305D
centralized critic observation.

Architecture choices and layer sizes are changed in code by editing those
constants, not via CLI flags.

### PPO configuration

The initial shared PPO configuration uses:

```text
gamma                 0.995
GAE lambda             0.95
learning rate          3e-4
PPO clip               0.2
KL coefficient         0.0
entropy coefficient    0.01
value loss coefficient 0.5
gradient clipping      0.5
train batch            16,384 agent steps
minibatch              2,048
epochs                 4
LSTM sequence length   64
```

`kl_coeff` is disabled because the heavily masked action distributions can
produce non-finite KL values in small batches. PPO clipping and entropy
regularization remain active.

### Handcrafted bot compatibility

The greedy, old-greedy, and buffer agents were updated to support two produced
foods. Learned policies can be evaluated against any agent in
`AGENT_REGISTRY`.

## Installation

From the `research` directory:

```bash
python -m pip install -e '.[rl]'
```

This installs RLlib, PyTorch, TensorBoard, Gymnasium, and the research package.

## Training

### Small CPU smoke run

```bash
python train_ppo.py \
  --iterations 2 \
  --num-env-runners 1 \
  --num-gpus 0 \
  --train-batch-size 1024 \
  --minibatch-size 256 \
  --max-seq-len 64 \
  --num-epochs 1
```

### Normal LSTM run

```bash
python train_ppo.py \
  --iterations 100 \
  --num-env-runners 4 \
  --num-gpus 1 \
  --max-seq-len 64
```

The GPU is assigned to the learner. Game simulation remains on CPU environment
runners.

### Useful training flags

```text
--iterations              PPO training iterations
--seed                    global training seed
--output                  root output directory
--resume                  RLlib checkpoint to restore
--num-env-runners         CPU simulation workers
--num-gpus                learner GPU allocation
--evaluation-interval     RLlib evaluation frequency
--evaluation-episodes     episodes per evaluation
--checkpoint-interval     checkpoint frequency
--train-batch-size        agent steps per update
--minibatch-size          PPO minibatch size
--num-epochs              passes over each training batch
--max-seq-len             LSTM training sequence length
```

## Training outputs

Each run creates a timestamped directory under `runs/` by default:

```text
runs/ppo-<timestamp>/
├── run_config.json
├── iterations.jsonl
├── tensorboard/
├── checkpoints/
└── latest_checkpoint.txt
```

- `run_config.json`: resolved CLI configuration.
- `iterations.jsonl`: complete JSON-serializable RLlib result per iteration.
- `tensorboard/`: scalar event files.
- `checkpoints/`: RLlib checkpoints.
- `latest_checkpoint.txt`: absolute path to the newest checkpoint.

View metrics with:

```bash
tensorboard --logdir runs
```

## Evaluation

The keyword `policy` means the learned checkpoint. Other values are
handcrafted names from `AGENT_REGISTRY`.

### Shared-policy self-play

```bash
python evaluate_ppo.py \
  --checkpoint <checkpoint-path> \
  --agents policy,policy,policy,policy \
  --episodes 100
```

### Mixed learned and handcrafted agents

```bash
python evaluate_ppo.py \
  --checkpoint <checkpoint-path> \
  --agents policy,greedy,buffer,random \
  --episodes 100
```

Another example:

```bash
python evaluate_ppo.py \
  --checkpoint <checkpoint-path> \
  --agents policy,greedy,greedy,noop \
  --episodes 100
```

By default, the requested lineup is rotated through all four physical seats.
Thus `--episodes 100` runs 100 games for each of four rotations. Use
`--no-rotate` to evaluate only the supplied seat assignment.

Evaluation is deterministic by default. Add `--stochastic` to sample from the
policy distribution.

### Evaluation output

The evaluator prints and writes:

- Win rate.
- Survival rate.
- Mean final cash.
- Mean death time.
- Mean episode length.
- Mean number of trades.
- Aggregates by agent type.
- Aggregates by physical seat.
- Full per-game results.

Specify the output path with:

```bash
--output evaluation/my_run.json
```

Otherwise, `evaluation.json` is written beside the checkpoint.

## Tests

Focused RL tests:

```bash
python -m pytest -q \
  tests/test_observation_encoder.py \
  tests/test_global_observation_encoder.py \
  tests/test_rllib_env.py \
  tests/test_rllib_module.py \
  tests/test_rllib_config.py \
  tests/test_rllib_callbacks.py \
  tests/test_evaluate_ppo.py
```

Trainer/checkpoint integration tests:

```bash
NASH_HUNGER_RUN_RLLIB_INTEGRATION=1 \
python -m pytest -q tests/test_rllib_training_integration.py
```

The integration test trains both MLP and LSTM policies for one minimal PPO
iteration, saves checkpoints, restores them, and verifies the actor type.

## Current status and caveats

- Focused RL tests pass.
- MLP and LSTM trainer/checkpoint/restore smoke tests pass.
- The real training CLI and mixed policy/bot evaluator have completed smoke
  runs successfully.
- RLlib emits a deprecation warning related to its internal legacy
  `RLModuleConfig` constructor path. The custom module itself uses the new
  observation/action/model configuration interface.
- RLlib may suggest installing `gputil`; this is optional and only improves GPU
  monitoring.
- Cloud packaging, Docker images, distributed Ray clusters, opponent pools,
  and hyperparameter sweeps are not implemented yet.

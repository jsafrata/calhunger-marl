# research/ — the CalHunger simulator, agents, and RL stack

This is the frozen code base every result in the papers was produced with. The top-level
[README](../README.md) covers installation, the reproduction map, and the data archive; this file
documents the package itself. (The code calls the game by its internal name, *Nash Hunger*.)

## Layout

```
env/
  config.py          GameConfig: 4 players, 4 foods, $100 cash, 100 food units (>=20 per food), T=180 s,
                     feeding every 10 s (produce 2 / eat 2 per food), order limits
  types.py           Player, Order, Trade, OrderBook, GameState
  rng.py             one seeded PCG64 generator per environment (same seed -> same game)
  setup.py           producer assignment (two permutations: every food has exactly two producers), dealing
  market.py          order books, price-time priority, maker-price matching, reservations (locked cash/food)
  tick.py            production, consumption, death, winner (max cash among survivors; ties by total food)
  actions.py         discrete action space (wait / post / take / cancel; qty 1..Q, price 1..P) + legality mask
                     Q and P are read from HG_MAX_QTY / HG_MAX_PRICE before any RL import (defaults 5 / 50
                     -> 2,049 actions, the papers' setting)
  observation.py     per-player partial observation (own state, public books, trades, who is alive)
  env.py             FoodTradingEnv: reset / step / observe / legal_actions / is_done / get_result
  small_env.py       duration-scaled variants: small (30 s), small_60 ... small_150, with rescaled endowments
  logging.py         per-game JSON log
  mappo_env.py       centralized-critic observation for MAPPO
env/rl/
  rllib_env.py       NashHungerMultiAgentEnv (RLlib MultiAgentEnv; terminal-only reward -1 / 0 / +1)
  observation_encoder.py, global_observation_encoder.py   local / centralized feature encoders
  factored_actor.py  factored masked action heads (type, food, side, quantity, price)
  rllib_module.py    PPO RLModule: MLP or LSTM torso, factored actor, centralized MLP critic
  rllib_config.py    build_ppo_config(): IPPO / MAPPO, episode-matched batching, schedules
  callbacks.py, rewards.py
agents/
  scripted_v1.py ... scripted_v5.py   the calibrated ladder (reference bot, market maker, JIT procurer,
                     storage trader, edge-only taker); scripted_common.py shared machinery
  cadenced_scripted.py   decision-cadence wrapper every scripted seat plays through ("fast" in all papers)
  checkpoint_policy_agent.py   seats an RLlib checkpoint in the plain game runners (policy:NAME)
  greedy_agent.py, buffer_agent.py, random_agent.py, noop_agent.py, old_greedy_agent.py   legacy baselines
eval/
  long_baseline.py   the training entry point used by every Slurm campaign (IPPO/MAPPO, checkpoints every N)
  scaled_checkpoint_scripted_diagnostics.py   checkpoint vs three copies of one archetype, per checkpoint
  checkpoint_selfplay_survival_diagnostics.py four-seat self-play survival + per-head entropy fingerprints
  scripted_version_tournament.py   the ladder calibration tournament (Appendix A)
  deep_eval.py, pilot.py, run_eval.py, configurations.py   earlier diagnostic harnesses
  plot_*.py          campaign-level plots
hpc/
  <campaign>/        campaign.json (locked hyperparameters + library versions), preflight.py, run_task.py,
                     run_array.sbatch, submit_all.sh, README/RUNBOOK; DEPLOY.md is the deployment checklist
tests/               pytest suite: rules, invariants, determinism, encoders, actor, RL module, harness plans
bugs/                minimal reproductions of simulator bugs found and fixed during development
parity/              bridge to the original TypeScript implementation of the game (not included here; the
                     parity test skips automatically when the toolchain is absent)
run_game.py, run_batch.py, watch_game.py, train_ppo.py   command-line runners
```

## Using the environment directly

```python
from env.env import FoodTradingEnv
from agents import AGENT_REGISTRY

env = FoodTradingEnv(seed=42)
agents = [AGENT_REGISTRY[name]() for name in ("scripted_v4", "scripted_v4", "scripted_v4", "scripted_v2")]

obs = env.reset()
while not env.is_done():
    actions = {i: agents[i].act(obs[i], env.legal_actions(i)) for i in range(4)}
    obs, rewards, done, info = env.step(actions)

print(env.get_result())   # winner, end reason, per-player cash / inventory / survival
```

`env.legal_actions(i)` returns a boolean mask over the discrete action space; illegal actions are
no-ops. For the 30–150 s variants use `env.small_env.SCALED_ENVIRONMENTS[name]`, which returns the
`(config, environment class)` pair, or set `HUNGER_ENVIRONMENT_VARIANT` for the checkpoint harnesses.

## Training and evaluating checkpoints

```bash
# Local smoke run of the campaign code path (30 s variant, 2 iterations, 1 env runner)
HG_MAX_QTY=5 HG_MAX_PRICE=50 python -m eval.long_baseline --algos ippo --env_variant small \
  --iters 2 --games_per_update 4 --num_env_runners 1 --checkpoint_every 1 --out_root /tmp/calhunger_smoke

# A run directory looks like <out_root>/<algo>_seed<k>_headmlp/{checkpoints/iter_0100, ..., iters.jsonl}
# Evaluate every checkpoint of a run against three storage traders (v4), 100 paired games each:
HUNGER_ENVIRONMENT_VARIANT=default HG_MAX_QTY=5 HG_MAX_PRICE=50 \
python -m eval.scaled_checkpoint_scripted_diagnostics --campaign-root <dir containing the run dirs> \
  --algorithms ippo --seeds 0 --checkpoint-start 100 --checkpoint-stop 1200 --checkpoint-every 100 \
  --n-games 100 --game-seed-base 900000 --opponent v4 --output-dir <out>
```

The `*.sbatch` files under `hpc/` are the exact invocations behind every campaign; `hpc/README.md`
files describe each campaign's arms, budgets, and outputs. Legacy notes: `RLLIB_PPO_GUIDE.md`
(the RLlib integration) and `GREEDY_BOT.md` (the original live-game bot that `scripted_v1` ports).

## Tests

```bash
pytest -q                      # full suite (~45 files; RLlib integration tests take a few minutes)
pytest -q -m "not slow"        # skip the Ray-starting tests
```

# Caltech Resnick v2 five-seed CPU campaign

This directory prepares, but does not submit, the following frozen campaign:

- IPPO and MAPPO
- seeds 0, 1, 2, 3, 4 for each algorithm
- 1,500 training iterations per run
- full RLlib checkpoints every 100 iterations from 100 through 1500
- ten independent CPU jobs, each with 6 CPUs, 16 GB RAM, and a 12-hour
  walltime
- no GPU allocation

`campaign.json` is the machine-readable source of truth. The array ordering
is IPPO seeds 0-4 at task IDs 0-4, followed by MAPPO seeds 0-4 at task IDs
5-9.

## Safety properties

- `run_task.py` only prints a plan unless `--execute` is explicitly supplied.
- Fresh runs fail if their seed-specific output directory already exists.
- Every job runs a strict package/config/git preflight before training.
- Each attempt records a JSON manifest with the command, git revision, SLURM
  environment, and frozen campaign configuration.
- Completed jobs verify all 1,500 metric rows and all 15 checkpoint directories.
- Parallel jobs write to distinct directories, so seeds cannot append to one
  another's metrics.

## 1. Put the repository in persistent storage

This campaign is sized to fit in the submitting user's 60 GB personal VAST
allocation. Use a clean committed snapshot under
`/resnick/home/$USER/Hunger_games_2/research`. The login shell may display
`/home/$USER`; `readlink -f "$HOME"` shows the canonical VAST path. Group
storage under `/resnick/groups/<group>/...` is also accepted.

Check available capacity with `hpcquota`. Do not copy the local macOS
`.venv-mappo`; it is not portable to the cluster's x86 Linux nodes.

The strict preflight rejects an uncommitted worktree by default. This prevents
ten expensive runs from being produced from code that cannot later be
identified. `HUNGER_ALLOW_DIRTY=1` exists as an emergency override, but should
not be used for the actual campaign.

## 2. Fill in the local campaign environment

From the research repository root on Resnick:

```bash
cp hpc/v2_5seed_cpu/campaign.env.example hpc/v2_5seed_cpu/campaign.env
```

Edit `campaign.env`, replacing `REPLACE_ACCOUNT`, then:

```bash
source hpc/v2_5seed_cpu/campaign.env
sacctmgr show user "$USER" withassoc
```

The account in `HUNGER_SLURM_ACCOUNT` must be one listed by `sacctmgr`. The
default filesystem paths use the submitting user's personal VAST allocation;
they may instead all point into an authorized group allocation. `campaign.env`
is gitignored.

## 3. Create the locked CPU environment

The campaign environment names Resnick's available Python module and the setup
script loads it automatically. Confirm it before installation:

```bash
source hpc/v2_5seed_cpu/campaign.env
module load "$HUNGER_PYTHON_MODULE"
python3 --version
bash hpc/v2_5seed_cpu/setup_env.sh
```

`python3 --version` must report Python 3.11.6. The setup script refuses to
modify an existing environment. Resnick does not provide the local development
machine's Python 3.12, so the HPC lock uses its supported Python 3.11.6 module;
all pinned binary dependencies publish CPython 3.11 Linux wheels. Package
versions remain Ray 2.53.0, PyTorch 2.12.0 CPU, NumPy 2.4.6, Gymnasium 1.1.1,
PyArrow 25.0.0, and SciPy 1.17.1. The script deliberately does not install from
the repository's broader and currently divergent dependency declarations.

If the exact CPU PyTorch wheel is unavailable from the configured index, stop
and resolve the lock rather than substituting another Torch or Ray version.

## 4. Validate without training

```bash
source hpc/v2_5seed_cpu/campaign.env
export HG_MAX_QTY=5
export HG_MAX_PRICE=50
"$HUNGER_HPC_VENV/bin/python" -m hpc.v2_5seed_cpu.preflight --strict
bash hpc/v2_5seed_cpu/validate_plan.sh
```

`validate_plan.sh` prints all ten fully resolved commands. It cannot start
training because it never supplies `--execute`.

## 5. Recommended smoke test

This is the first command in this guide that submits computation. Do not run it
until the campaign is approved:

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --export=ALL \
  hpc/v2_5seed_cpu/smoke_test.sbatch
```

The smoke test runs a separate MAPPO seed 999 for 100 iterations. It does not
touch any of the ten campaign directories. After it finishes, inspect:

```bash
sacct -j <job-id> -o JobID,State,Elapsed,AllocCPUS,MaxRSS
seff <job-id>
```

Confirm that it used no GPU and stayed below 16 GB. Smoke job 65381253 took
40:56 wall-clock for 100 MAPPO iterations and peaked at 4.23 GB. Its linear
1,500-iteration projection is about 10 hours 14 minutes, so the production
request uses a 12-hour walltime. The smoke output lives under
`$HUNGER_CAMPAIGN_ROOT/_smoke` and a second smoke submission intentionally
fails rather than overwriting it.

## 6. Submit the ten-run campaign

This is the production submission command. It is documented here but has not
been run:

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --export=ALL \
  hpc/v2_5seed_cpu/run_array.sbatch
```

The array may run all ten tasks concurrently (60 allocated CPUs total). Queue
start times can be staggered by fairshare. Monitor with:

```bash
squeue -u "$USER"
sacct -j <array-job-id> -o JobID,State,Elapsed,AllocCPUS,MaxRSS
```

SLURM stdout/stderr files land in `hpc/v2_5seed_cpu/slurm/` and are gitignored.

## 7. Verify the finished campaign

```bash
"$HUNGER_HPC_VENV/bin/python" -m hpc.v2_5seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```

A successful report requires ten complete runs, iteration counters 1 through
1500 with no gaps or duplicates, exactly the requested checkpoint set, and an
RLlib marker in every checkpoint directory.

Expected run directories are:

```text
ippo_seed0_headmlp ... ippo_seed4_headmlp
mappo_seed0_headmlp ... mappo_seed4_headmlp
```

Attempt manifests are written to `$HUNGER_CAMPAIGN_ROOT/_manifests`.

## Resuming one interrupted array task

If a task stops exactly at a checkpoint, resubmit only that array index:

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --array=<task-id> \
  --export=ALL,HUNGER_RESUME=1 hpc/v2_5seed_cpu/run_array.sbatch
```

Resume is intentionally conservative. It refuses to proceed if `iters.jsonl`
contains rows after the latest checkpoint, because blindly appending after a
restore would duplicate iteration numbers. In that situation, archive and
reconcile the partial metrics before retrying. Never point a fresh task at an
existing seed directory.

## Files

- `campaign.json`: frozen algorithms, seeds, hyperparameters, checkpoints,
  versions, and requested resources
- `requirements-v2-cpu.txt`: critical environment lock
- `setup_env.sh`: one-time environment creation
- `preflight.py`: read-only environment/config/git checks
- `validate_plan.sh`: dry-run expansion of all ten tasks
- `run_array.sbatch`: production ten-task array
- `smoke_test.sbatch`: isolated 100-iteration performance check
- `run_task.py`: guarded task mapping, execution, manifests, and resume logic
- `verify_campaign.py`: artifact completeness verification

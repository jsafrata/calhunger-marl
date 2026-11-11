# Caltech Resnick entropy-0.05 one-seed pilot

This campaign runs two independent CPU jobs:

- array task 0: IPPO, seed 0
- array task 1: MAPPO, seed 0
- 1,500 iterations per task
- entropy coefficient 0.05
- full checkpoints at iterations 100, 200, ..., 1500

All other training, reward, model, and environment settings match the v2
five-seed campaign. The output root is separate, so this pilot cannot append
to or overwrite the entropy-0.01 results.

## Configure and validate on the HPC

From the research repository root:

```bash
cp hpc/v2_entropy005_seed0/campaign.env.example \
  hpc/v2_entropy005_seed0/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' \
  hpc/v2_entropy005_seed0/campaign.env
source hpc/v2_entropy005_seed0/campaign.env

bash hpc/v2_entropy005_seed0/validate_plan.sh
```

The plan must show exactly one IPPO command and one MAPPO command, each with
`--iters 1500`, `--entropy_coeff 0.05`, and the new campaign root. The plan is
read-only and does not start training.

## Submit both runs

This is the command that starts computation:

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --export=ALL \
  hpc/v2_entropy005_seed0/run_array.sbatch
```

Monitor it with:

```bash
squeue -r -j <array-job-id> -o "%.18i %.2t %.10M %.10l %.4C %R"
```

The two output directories will be:

```text
$HUNGER_CAMPAIGN_ROOT/ippo_seed0_headmlp
$HUNGER_CAMPAIGN_ROOT/mappo_seed0_headmlp
```

SLURM logs are written under `hpc/v2_entropy005_seed0/slurm/`.

## Verify after both tasks finish

```bash
"$HUNGER_HPC_VENV/bin/python" \
  -m hpc.v2_entropy005_seed0.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"
```

## Resume a timed-out task

Task 0 is IPPO and task 1 is MAPPO. Resume only the interrupted task:

```bash
sbatch --account="$HUNGER_SLURM_ACCOUNT" --array=<task-id> \
  --export=ALL,HUNGER_RESUME=1 \
  hpc/v2_entropy005_seed0/run_array.sbatch
```

Resume is accepted only when the metrics file ends exactly at the latest
checkpoint, preventing duplicated iteration rows.

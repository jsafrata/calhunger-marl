# Runbook: deploy and launch the best-config campaign

Paste-ready instructions for the executing agent. Follow `hpc/DEPLOY.md`
conventions throughout: every step's check must pass before the next step;
on any mismatch, STOP and report the output instead of improvising.

## 1. On the Mac — commit, bundle, verify, transfer

```bash
cd <repo-root>/research
git rev-parse --show-toplevel     # MUST end in /research (nested-repo trap)
git status --short                # review; then stage this change set:
git add env eval hpc tests
git diff --cached --stat          # eyeball: intended files only
git commit -m "Best-config bundle: lr anneal, LSTM eval state-carry, campaign + eval harnesses"
git log --oneline -1              # note <COMMIT> for step 2's match check

git bundle create ~/hunger-best-config.bundle main
git bundle verify ~/hunger-best-config.bundle
scp ~/hunger-best-config.bundle hpc:~/        # `hpc` alias only — never the full hostname
```

## 2. On Resnick — update the clone

```bash
ssh hpc
cd ~/Hunger_games_2/research
git status --short                            # MUST be empty
git bundle verify ~/hunger-best-config.bundle
git fetch ~/hunger-best-config.bundle main
git merge --ff-only FETCH_HEAD                # STOP on any refusal
git log --oneline -1                          # MUST equal <COMMIT> from step 1
```

## 3. Configure and echo the campaign environment

```bash
cp hpc/best_config_2seed_cpu/campaign.env.example \
  hpc/best_config_2seed_cpu/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/best_config_2seed_cpu/campaign.env
source hpc/best_config_2seed_cpu/campaign.env
echo "$HUNGER_SLURM_ACCOUNT"; echo "$HUNGER_RESEARCH_ROOT"
echo "$HUNGER_CAMPAIGN_ROOT";  echo "$HUNGER_HPC_VENV"     # all non-empty
```

## 4. Preflight and plan validation

```bash
"$HUNGER_HPC_VENV/bin/python" -m hpc.best_config_2seed_cpu.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" --strict \
  > ~/best_config_preflight.json 2>&1
echo "preflight exit: $?"                                  # MUST be 0
grep -E '"ok"|"errors"' ~/best_config_preflight.json       # ok true, errors []

bash hpc/best_config_2seed_cpu/validate_plan.sh \
  > ~/best_config_plan.txt 2>&1
echo "plan exit: $?"                                       # MUST be 0
grep -c '"task_id":' ~/best_config_plan.txt                # MUST print 16
grep -c -- '--include_survival_features' ~/best_config_plan.txt   # MUST print 16
grep -c -- '--torso lstm' ~/best_config_plan.txt           # MUST print 8
```

## 5. Submit

```bash
bash hpc/best_config_2seed_cpu/submit_all.sh
# record the four job ids it prints
```

## 6. Monitor

```bash
watch -n 15 'squeue -u $USER -o "%.18i %.22j %.2t %.10M %.10l %R"'
tail -n 3 hpc/best_config_2seed_cpu/slurm/*<jobid>*.out
# after completion:
"$HUNGER_HPC_VENV/bin/python" -m hpc.best_config_2seed_cpu.verify_campaign \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT"      # exit 0, "ok": true
```

Expected wall-clock: small_90_mlp ~10 h, small_90_lstm up to ~24 h,
default_180_mlp ~20 h, default_180_lstm up to ~48 h. A task killed at its
wall is NOT lost: resubmit that arm with
`HUNGER_RESUME=1` in the `--export` list (see README) and it resumes from
the latest checkpoint.

## 7. After training completes — evaluations

```bash
source hpc/best_config_2seed_cpu/campaign.env
bash hpc/best_config_eval/submit_evals.sh
# done when:
ls "$HUNGER_CAMPAIGN_ROOT"/_scripted_eval_100/*/*/checkpoint_diagnostics.json | wc -l   # 16
ls "$HUNGER_CAMPAIGN_ROOT"/_selfplay_survival_50/*/selfplay_survival_diagnostics.json | wc -l  # 4
```

## 8. Sync results back (on the Mac)

```bash
rsync -av hpc:"<HUNGER_CAMPAIGN_ROOT>/_scripted_eval_100/" \
  research/eval/results/best_config_scripted_eval_100/
rsync -av hpc:"<HUNGER_CAMPAIGN_ROOT>/_selfplay_survival_50/" \
  research/eval/results/best_config_selfplay_survival_50/
```
(Expand `<HUNGER_CAMPAIGN_ROOT>` to the literal Resnick path.)

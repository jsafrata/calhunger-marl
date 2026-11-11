# Deploying code and campaigns to Resnick — the checked workflow

Every HPC deployment follows these steps in order. None are optional; each
one exists because skipping it has already caused (or nearly caused) a
silent failure.

## 0. Repository layout — read this first

The git repository that matters is rooted at **`Hunger_games_2/research`**
(commits like `618cf08` live here). The parent `Hunger_games_2` directory
on the laptop also contains a **stale outer repo** whose `main` is frozen
at `f2e2139` ("Latest changes", July 2026). Bundling from the wrong root
once deployed month-old code to the cluster. Always verify before
bundling:

```bash
cd <repo-root>/research
git rev-parse --show-toplevel   # must end in /research
git log --oneline -1            # must be the commit you just made
```

On Resnick the repo root is `/resnick/home/jsafrata/Hunger_games_2/research`
(the parent directory there is NOT a git repo — that is correct).

## 1. On the Mac: commit, bundle, verify, transfer

```bash
cd <repo-root>/research

# Stage the specific files for this change — not `git add -A`.
git add <files...>
git diff --cached --stat        # eyeball: only intended files, sane sizes
git commit -m "<message>"

git bundle create ~/hunger-<change-name>.bundle main
git bundle verify ~/hunger-<change-name>.bundle

# ALWAYS use the `hpc` ssh alias — it shares one authenticated connection
# (ControlMaster, persists 8h), so only the first command of a session
# asks for password + Duo. Using the full hostname bypasses this.
scp ~/hunger-<change-name>.bundle hpc:~/
```

## 2. On Resnick: update the clone

```bash
ssh hpc
cd ~/Hunger_games_2/research

git status --short              # MUST be empty; if not, stop and resolve
git bundle verify ~/hunger-<change-name>.bundle
git fetch ~/hunger-<change-name>.bundle main
git merge --ff-only FETCH_HEAD  # if this fails, STOP — histories diverged
git log --oneline -1            # must match the laptop commit id exactly
```

`campaign.env` files are untracked and survive updates. If the merge
refuses to fast-forward, do not force anything — diagnose which side is
stale.

## 3. Configure and echo the campaign environment

```bash
test -e hpc/<campaign>/campaign.env || \
  cp hpc/<campaign>/campaign.env.example hpc/<campaign>/campaign.env
sed -i 's/REPLACE_ACCOUNT/hippo/' hpc/<campaign>/campaign.env
source hpc/<campaign>/campaign.env

echo "$HUNGER_SLURM_ACCOUNT"; echo "$HUNGER_RESEARCH_ROOT"
echo "$HUNGER_CAMPAIGN_ROOT";  echo "$HUNGER_HPC_VENV"
```

All four must print non-empty, correct paths.

## 4. Preflight and plan validation (training campaigns)

```bash
"$HUNGER_HPC_VENV/bin/python" -m hpc.<campaign>.preflight \
  --research-root "$HUNGER_RESEARCH_ROOT" \
  --campaign-root "$HUNGER_CAMPAIGN_ROOT" --strict \
  > ~/<campaign>_preflight.json 2>&1
echo "preflight exit: $?"                      # must be 0
grep -E '"ok"|"errors"' ~/<campaign>_preflight.json

bash hpc/<campaign>/validate_plan.sh > ~/<campaign>_plan.txt 2>&1
echo "plan exit: $?"                           # must be 0
grep -c '"task_id":' ~/<campaign>_plan.txt     # must equal the array size
```

Eval-only harnesses (scripted eval, self-play survival) have no separate
preflight — they validate every expected checkpoint path at startup and
fail fast — but steps 0-3 still apply in full.

## 5. Submit

Use the harness's `submit_*.sh` (they pass `--account` and any per-horizon
walltimes). Record the returned job ids.

## 6. Monitor

```bash
watch -n 15 'squeue -r -j <ids> -o "%.18i %.22j %.2t %.10M %.10l %R"'
tail -n 3 hpc/<harness>/slurm/*<jobid>*.out    # per-task progress lines
sacct -j <ids> --format=JobID,JobName%22,State,Elapsed,MaxRSS   # afterwards
```

Diagnostics cache per-checkpoint raw JSONs: a task killed by walltime
resumes from the cache on plain resubmission.

## 7. Sync results back (run on the Mac)

```bash
rsync -av hpc:<campaign-root>/<output-dir>/ \
  research/eval/results/<local-name>/
```

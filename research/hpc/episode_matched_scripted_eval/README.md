# Episode-matched campaign: checkpoint diagnostics against v2–v5

This can evaluate all 20 training runs at checkpoints 100, 200, ..., 1200.
The initial submission covers the 16 completed runs at 60, 90, 120, and 150
seconds; the four 180-second runs are submitted separately later. For every
checkpoint it plays 100 stochastic games with the learned policy in seat 0
and three copies of one scripted opponent in seats 1–3.

`submit_all.sh` currently submits four Slurm arrays for the completed 60, 90,
120, and 150-second horizons. `submit_180.sh` submits the 180-second array
separately once its training finishes. Each array has four tasks corresponding
to opponents v2, v3, v4, and v5. A task evaluates both IPPO and MAPPO, seeds 0
and 1, and records:

- average survival time;
- average terminal cash;
- average completed trades involving the learned seat;
- win percentage;
- survival percentage.

Results are resumable and stored under:

```text
$HUNGER_CAMPAIGN_ROOT/_scripted_eval_100/<horizon>/<opponent>/
├── checkpoint_diagnostics.csv
├── checkpoint_diagnostics.json
└── raw/
```

No graphs or Matplotlib files are created on the HPC. Copy the result tree to
the Mac and run the plotting code locally after all requested tasks finish.

Submit after sourcing the training campaign's `campaign.env`:

```bash
bash hpc/episode_matched_scripted_eval/submit_all.sh
```

After the 180-second training campaign finishes and verifies:

```bash
bash hpc/episode_matched_scripted_eval/submit_180.sh
```

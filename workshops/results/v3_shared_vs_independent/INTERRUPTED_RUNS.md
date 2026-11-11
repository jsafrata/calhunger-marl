# v3 campaign stop record

The campaign was stopped at the user's request at `2026-07-21T16:16:01Z`.
No checkpoints, metric rows, completed manifests, or v2 artifacts were deleted.

## Complete training runs

| Algorithm | Condition | Seed | Iterations | Status |
| --- | --- | ---: | ---: | --- |
| IPPO | Shared | 0 | 1100 | Exact read-only v2 reference |
| IPPO | Shared | 1 | 1100 | Complete |
| IPPO | Shared | 2 | 1100 | Complete |
| IPPO | Shared+ID | 0 | 1100 | Complete |
| IPPO | Shared+ID | 1 | 1100 | Complete |
| MAPPO | Shared | 0 | 1100 | Exact read-only v2 reference |

No full v3 evaluation jobs had started when the campaign was stopped.

## Partial training runs

| Algorithm | Condition | Seed | Last metric row | Last checkpoint | Environment transitions | Agent transitions | Wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IPPO | Shared+ID | 2 | 474 | 400 | 1,941,504 | 6,551,549 | 5,303.853 |
| IPPO | Independent | 0 | 237 | 200 | 970,752 | 3,253,335 | 4,328.775 |

For Shared+ID seed 2, exact resumable state ends at iteration 400; rows 401–474
are retained as training telemetry. For Independent seed 0, exact resumable state
ends at iteration 200; rows 201–237 are retained as telemetry. Resuming should
truncate only those post-checkpoint telemetry rows and continue from the paths in
each run's `partial_progress.resume_checkpoint` field.

## Independent-policy accounting at stop

At iteration 237, unique trajectory samples were:

| Network | Unique samples | Learner samples processed | Optimizer steps |
| --- | ---: | ---: | ---: |
| `pi_0` | 767,217 | 3,662,848 | 3,577 |
| `pi_1` | 799,358 | 3,662,848 | 3,577 |
| `pi_2` | 834,841 | 3,662,848 | 3,577 |
| `pi_3` | 851,919 | 3,662,848 | 3,577 |

This confirms that each policy collected only its own seat trajectory while the
cyclic PPO minibatcher applied the same number of optimizer updates to each
independent network.

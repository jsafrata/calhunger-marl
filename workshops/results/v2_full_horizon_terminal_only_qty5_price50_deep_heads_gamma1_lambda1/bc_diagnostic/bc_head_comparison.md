# BC linear-vs-MLP head comparison (v2, MAX_QTY=5, MAX_PRICE=50)

Dataset: 2000/300/300 games of Scripted-v1 Fast under v2 env. Same dataset used for both head kinds. Early-stopping patience=5 on val flat_accuracy.

## Training

| metric | linear head | mlp head |
|---|---|---|
| epochs run | 28 | 28 |
| best val flat_accuracy | 0.3481 | 0.4151 |
| test flat_accuracy | 0.3518 | 0.4167 |
| test type accuracy | 0.9977 | 0.9944 |
| test food accuracy | rel | 0.5493 | 0.5561 |
| test side accuracy | rel | 0.7467 | 1.0000 |
| test price accuracy | rel | 0.6377 | 0.7019 |
| test qty accuracy | rel | 1.0000 | 1.0000 |
| post side accuracy | POST | 0.7467 | 1.0000 |
| post food accuracy | POST | 0.5493 | 0.5561 |
| post price exact | POST | 0.6377 | 0.7019 |
| post qty exact | POST | 0.0000 | 0.0000 |

## Deterministic self-play + 1-vs-3-scripted rollouts

### Lineup: bc_selfplay (500 games)

| metric | linear | mlp |
|---|---|---|
| mean survivors / game | 1.64 | 3.94 |
| reach-180 fraction | 0.8520 | 0.9940 |
| mean trades / game | 119.9 | 116.7 |
| mean per-seat survival | 0.4090 | 0.9840 |

### Lineup: 1bc_3fast (500 games)

| metric | linear | mlp |
|---|---|---|
| mean survivors / game | 3.24 | 3.94 |
| reach-180 fraction | 1.0000 | 1.0000 |
| mean trades / game | 103.4 | 106.7 |
| seat_0 (BC) survival rate | 0.4820 | 0.9460 |

### Lineup: scripted_v1_fast_reference (500 games)

| metric | linear | mlp |
|---|---|---|
| mean survivors / game | 4.00 | 4.00 |
| reach-180 fraction | 1.0000 | 1.0000 |
| mean trades / game | 105.8 | 105.8 |
| mean per-seat survival | 0.9990 | 0.9990 |

## Recommendation

**Winner: `mlp` heads.**

MLP heads reach val flat_accuracy 0.4151 vs linear 0.3481. Adopt MLP heads (`head_kind='mlp'`) as the v2 default — larger head capacity, at negligible parameter cost, gives at least parity on BC and more headroom for the RL run. Self-play mean survivors: linear=1.64, mlp=3.94.
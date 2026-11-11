# bc_vs_ladder — BC capacity control evaluated in cross-play

Evaluates the v2-era behavior-cloning actor (mlp heads; the "98.4% self-play survival"
control) against each scripted archetype v2-v5: 1 BC seat + 3 cadenced (fast) copies,
seat-balanced (4 rotations x 25 games), fixed seeds (base 700000, identical across
opponents), plus a 4x-BC self-play reference on the same seeds. 100 games per lineup.

Motivation: the papers' BC control should not itself rest on self-play numbers, since the
papers argue self-play metrics are unreliable. This provides the cross-play row.

Deploy: standard flow. The BC actor weights are data (not in git) — scp
research/eval/results/v2_full_horizon_terminal_only_qty5_price50_deep_heads_gamma1_lambda1/bc_diagnostic/mlp/bc_actor.pt
to the cluster and set HUNGER_BC_ACTOR to its path; set HUNGER_OUTPUT_DIR.
Local 4-game smoke (Aug 13): mechanics verified; BC 100% self-play, 0% vs each of
v2-v5 — expect the full run to show the BC control is population-bound too.

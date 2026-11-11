# best_config_peer_crossplay — peers experiment on the best architecture

Purpose: the IAEval/Who-Verifies "cross-play against independently trained peers" figure,
redone on the agreed best architecture only (default_180_mlp best-config arms: episode-
matched batching, survival features, trade-history critic, entropy/lr anneals, MLP heads).
Replaces the earlier v3-campaign matrix, which mixed in the deprecated shared+ID (agent
one-hot) condition and an off-topic iter-1400 checkpoint.

Design: 10 policies (IPPO/MAPPO x seeds 0-4, all iter_1200) — identical architecture and
training config, differing ONLY in seed and critic input. Full ordered 1v3 pair matrix
(90 jobs) + mandatory self-play diagonal (10 jobs), 100 seat-balanced games each on paired
seeds (base 900000), stochastic sampling. One Slurm array task per focal policy; jobs are
cache-resumable (raw JSON per job); tasks shard disjoint focals so a shared output dir is
safe. Output: $HUNGER_CAMPAIGN_ROOT/_peer_crossplay_100/{crossplay_matrix.csv,raw/}.

Harness: checkpoint_policy_crossplay.py (copy of the workshops eval harness; reads each
checkpoint's declared obs contract, so survival-features/trade-history policies encode
correctly). Estimated ~30-60 min per task; walltime 4 h.

Deploy: standard DEPLOY.md flow; copy campaign.env.example -> campaign.env, set account
hippo. After completion, rsync _peer_crossplay_100/ to workshops/results/seeds234/ and
regenerate the F3 figure with the best-config labels.

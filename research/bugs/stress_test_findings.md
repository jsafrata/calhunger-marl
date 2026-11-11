# Stress Test Findings

Run the stress suite with:

```bash
python -m pytest research/tests/test_env_stress.py
```

Latest local result:

```text
7 passed, 4 xfailed
```

The four `xfail` cases are intentional known-bug catches. They keep the stress
suite runnable today while preserving executable checks that should flip once
the environment is fixed.

## Bugs Currently Caught

1. Replacement post mask ignores reusable reserves.
   - Stress test: `test_stress_replacement_posts_are_legal_when_prior_reserves_cover_them`
   - MRE: `research/bugs/replacement_mask_mre.md`

2. Legacy immediate `post_order()` validates before locating the replaceable
   prior order.
   - Stress test: `test_stress_legacy_replacements_reuse_prior_reserves`
   - MRE: `research/bugs/legacy_post_replacement_mre.md`

3. Simultaneous stale replacement posts can raise after another same-tick action
   consumes the prior order.
   - Stress test: `test_stress_simultaneous_stale_replacements_are_rejected_not_crashes`
   - MRE: `research/bugs/simultaneous_stale_replacement_mre.md`

4. `NashHungerGym.reset(seed=None)` repeats seed `0`.
   - Stress test: `test_stress_gym_no_seed_resets_do_not_repeat_same_episode`
   - MRE: `research/bugs/gym_reset_seed_mre.md`

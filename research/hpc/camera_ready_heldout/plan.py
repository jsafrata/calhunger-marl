"""Print the task -> job mapping of both arrays (mirrors the sbatch arithmetic).

    python hpc/camera_ready_heldout/plan.py
"""
ROOTS = [("2seed", "0,1"), ("s234", "2,3,4")]
OPPONENTS = ["v2", "v3", "v4", "v5"]
ALGOS = ["ippo", "mappo"]
RUNS = ["ippo_s0", "ippo_s1", "ippo_s2", "ippo_s3", "ippo_s4",
        "mappo_s0", "mappo_s1", "mappo_s2", "mappo_s3", "mappo_s4"]
POPULATIONS = ["3x ep_ippo_s2", "3x ep_mappo_s2", "mixed {ep_ippo_s0, ep_mappo_s1, ep_ippo_s4}"]
CKPTS = list(range(100, 1201, 100))


def scripted_tasks():
    out = []
    for t in range(16):
        root, seeds = ROOTS[t // 8]
        opp = OPPONENTS[(t % 8) // 2]
        algo = ALGOS[t % 2]
        out.append((t, root, seeds, opp, algo, len(seeds.split(",")) * len(CKPTS)))
    return out


def peer_tasks():
    return [(t, POPULATIONS[t // 10], RUNS[t % 10], len(CKPTS)) for t in range(30)]


if __name__ == "__main__":
    print("Part A (scripted, held-out seeds): 16 tasks")
    cells = 0
    for t, root, seeds, opp, algo, n in scripted_tasks():
        cells += n
        print(f"  task {t:2d}: root={root:<5} seeds={seeds:<5} opponent={opp} algorithm={algo:<5} cells={n}")
    print(f"  total cells: {cells} (expect 10 runs x 12 ckpts x 4 opponents = 480)")
    print("Part B (held-out learned peers): 30 tasks")
    cells = 0
    for t, pop, run, n in peer_tasks():
        cells += n
        print(f"  task {t:2d}: population={pop:<44} run={run:<8} cells={n}")
    print(f"  total cells: {cells} (expect 10 runs x 12 ckpts x 3 populations = 360)")
    assert cells == 360
    assert sum(n for *_, n in scripted_tasks()) == 480
    # every (root, opponent, algorithm) combination appears exactly once
    combos = {(r, o, a) for _, r, _, o, a, _ in scripted_tasks()}
    assert len(combos) == 16
    print("plan OK")

"""
Runs the three planners on the toy regrasp map and prints what each one does.
Useful as a smoke test until the real regrasp map (from Jinjou/Hailey) is in.

    .venv/bin/python demo_planners.py
"""

from planners import baseline_plan, global_plan, greedy_plan
from regrasp_map import toy_map


def main():
    m = toy_map()
    for name, fn in [("baseline", baseline_plan),
                     ("greedy", greedy_plan),
                     ("global", global_plan)]:
        print(f"=== {name} ===")
        plan = fn(m)
        print(plan.describe(m))
        print()


if __name__ == "__main__":
    main()

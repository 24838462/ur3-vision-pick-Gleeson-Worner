"""Launch the UR3 vision-guided stacking simulation."""
import argparse

from simulation import COMPLETE, run_scene


def main():
    parser = argparse.ArgumentParser(description="UR3 search, PBVS pickup and placement.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not COMPLETE.exists():
        raise FileNotFoundError("Run python .\\compile_scene.py first.")
    run_scene(args.seed)


if __name__ == "__main__":
    main()

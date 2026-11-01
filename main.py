"""Launch the UR3 vision-guided stacking simulation."""
import argparse

from simulation import COMPLETE, run_scene


def main():
    parser = argparse.ArgumentParser(description="UR3 search, PBVS pickup and placement.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier (e.g. 2.0 or 3.0)")
    parser.add_argument("--fast", action="store_true", help="Run simulation at maximum uncapped CPU/GPU speed")
    args = parser.parse_args()
    if not COMPLETE.exists():
        raise FileNotFoundError("Run python .\\compile_scene.py first.")
    speed = 0.0 if args.fast else args.speed
    run_scene(args.seed, sim_speed=speed)


if __name__ == "__main__":
    main()

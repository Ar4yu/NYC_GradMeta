import argparse
import yaml
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    out = os.path.join(results_dir, "train_log.txt")
    with open(out, "w") as f:
        f.write("Training placeholder: pipeline wiring OK.\n")

    print(f"[train] Wrote {out}")

if __name__ == "__main__":
    main()

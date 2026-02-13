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

    out = os.path.join(results_dir, "eval_log.txt")
    with open(out, "w") as f:
        f.write("Eval placeholder: next step is real metrics + plots.\n")

    print(f"[eval] Wrote {out}")

if __name__ == "__main__":
    main()

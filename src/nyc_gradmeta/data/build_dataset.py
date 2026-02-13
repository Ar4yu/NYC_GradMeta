import argparse
import os
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "nyc_train.csv")
    test_path  = os.path.join(processed_dir, "nyc_test.csv")

    if not os.path.exists(train_path):
        with open(train_path, "w") as f:
            f.write("epiweek,cases,deaths\n")
    if not os.path.exists(test_path):
        with open(test_path, "w") as f:
            f.write("epiweek,cases,deaths\n")

    print(f"[build_dataset] Wrote placeholders:\n  {train_path}\n  {test_path}")

if __name__ == "__main__":
    main()

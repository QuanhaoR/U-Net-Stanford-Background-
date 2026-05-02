import os
import subprocess
import sys


def run_experiment(loss_name):
    cmd = [
        sys.executable, "main.py",
        "--loss", loss_name,
        "--epochs", "60",
        "--batch_size", "16",
        "--lr", "1e-3",
        "--img_size", "256",
    ]
    print(f"\n{'='*60}")
    print(f"Running experiment with loss: {loss_name}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Run all three loss configurations sequentially
    for loss in ["ce", "dice", "combined"]:
        run_experiment(loss)
    print("\nAll experiments completed!")

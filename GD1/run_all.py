import os
import subprocess
import sys
import time
from datetime import datetime

def run_cmd(cmd):
    print(f"\n{'='*50}\nRUN: {cmd}\n{'='*50}")
    start_time = time.time()
    res = subprocess.run([sys.executable] + cmd.split(), check=True)
    elapsed = time.time() - start_time
    print(f"DONE in {elapsed:.2f}s\n")

def main():
    print(f"START ALL EXPERIMENTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_start = time.time()

    # 1. Run BreastMNIST (Full ~780 samples, Binary)
    print("\n>>> [1/2] RUN BREASTMNIST (Binary)...")
    run_cmd("src/train.py --dataset breastmnist --epochs 30")

    # 2. Run OCTMNIST (5000 samples subset, Multi-class)
    print("\n>>> [2/2] RUN OCTMNIST (Multi-class, 5000 samples)...")
    run_cmd("src/train.py --dataset octmnist --max_samples 5000 --epochs 30")

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_time/60:.2f} minutes!")
    print(f"Results saved in 'results/' directory.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

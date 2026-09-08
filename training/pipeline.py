import subprocess
import sys
from train import train
from promote import promote_best_model

def run_training():

    print("\n==============================")
    print("STEP 1: TRAINING")
    print("==============================\n")

    result  = subprocess.run(
        [sys.executable, "training/train.py"],
        check=True
    )

    return result.returncode

def run_promotion():

    print("\n==============================")
    print("STEP 2: PROMOTION")
    print("==============================\n")

    result = subprocess.run(
        [sys.executable, "training/promote.py"],
        check=True
    )

    return result.returncode

def main():

    candidate_version = train()
    promote_best_model(candidate_version)

    print("\n==============================")
    print("PIPELINE COMPLETED")
    print("==============================\n")

if __name__ == "__main__":
    main()
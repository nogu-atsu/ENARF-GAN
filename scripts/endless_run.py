import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default="", required=True)

    args = parser.parse_args()

    while True:
        subprocess.run(args.command, shell=True)

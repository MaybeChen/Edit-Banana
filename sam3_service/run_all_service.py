#!/usr/bin/env python3
"""
Launch multiple SAM3 HTTP service workers (and optionally RMBG workers).
Use when you want one process per GPU or multiple ports for load balancing.

Usage:
  python -m sam3_service.run_all_service --workers 2
  python -m sam3_service.run_all_service --workers 2 --rmbg 1
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start multiple SAM3 (and optional RMBG) service workers")
    parser.add_argument("--workers", type=int, default=1, help="Number of SAM3 server processes")
    parser.add_argument("--rmbg", type=int, default=0, help="Number of RMBG server processes (0 = none)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--device", type=str, default="auto", help="Device for SAM3 (e.g. auto, cuda, cpu)")
    parser.add_argument("--base-port", type=int, default=8001, help="First SAM3 port (default 8001)")
    parser.add_argument("--rmbg-base-port", type=int, default=9101, help="First RMBG port (default 9101)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config = args.config or os.path.join(project_root, "config", "config.yaml")
    if not os.path.isfile(config):
        print(f"Config not found: {config}", file=sys.stderr)
        sys.exit(1)

    procs = []

    for i in range(args.workers):
        port = args.base_port + i
        env = os.environ.copy()
        if args.workers > 1 and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            sys.executable, "-m", "sam3_service.server",
            "--config", config,
            "--port", str(port),
            "--device", args.device,
        ]
        print(f"Starting SAM3 worker {i + 1}/{args.workers} on port {port} ...")
        procs.append(subprocess.Popen(cmd, env=env, cwd=project_root))

    for i in range(args.rmbg):
        port = args.rmbg_base_port + i
        cmd = [
            sys.executable, "-m", "sam3_service.rmbg_server",
            "--config", config,
            "--port", str(port),
        ]
        print(f"Starting RMBG worker {i + 1}/{args.rmbg} on port {port} ...")
        procs.append(subprocess.Popen(cmd, cwd=project_root))

    if not procs:
        print("No workers started (--workers and --rmbg)", file=sys.stderr)
        sys.exit(1)

    print("All workers started. Press Ctrl+C to stop.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Start the whole application on this machine (no Docker).

    python scripts/run_local.py                 # 3 processes: detector (8001), grouping (8002), gateway (8000)
    python scripts/run_local.py --mode local    # 1 process: everything inside the gateway
    python scripts/run_local.py --device cpu    # force CPU

Open http://localhost:8000 when it says "Ready". Ctrl+C stops everything.
The first start downloads the models (~0.7 GB) into ./models; later starts are fast.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shelfai.config import load_config  # noqa: E402


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for(url: str, proc: subprocess.Popen, name: str, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"{name} exited during startup (code {proc.returncode}). See its log above.")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"  {name} is up ({time.time() - t0:.0f} s)")
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise SystemExit(f"{name} did not start within {timeout:.0f} s.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["http", "local"], default="http")
    ap.add_argument("--device", help="auto | cpu | cuda | 0")
    ap.add_argument("--config", help="path to a config YAML (default: configs/default.yaml)")
    ap.add_argument("--startup-timeout", type=float, default=900, help="seconds (first run downloads models)")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               SHELFAI_SERVICES_MODE=args.mode, PYTHONUNBUFFERED="1")
    if args.device:
        env["SHELFAI_DEVICE"] = args.device
    if args.config:
        env["SHELFAI_CONFIG"] = str(Path(args.config).resolve())
    os.environ.update({k: v for k, v in env.items() if k.startswith("SHELFAI_")})
    cfg = load_config()
    svc = cfg["services"]

    procs: list[tuple[str, subprocess.Popen]] = []

    def start(name, module):
        p = subprocess.Popen([sys.executable, "-m", module], cwd=str(ROOT), env=env)
        procs.append((name, p))
        return p

    def stop(*_):
        for name, p in procs:
            if p.poll() is None:
                p.terminate()
        for name, p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(0)))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))

    ports = [svc["gateway_port"]] + ([svc["detector_port"], svc["grouping_port"]] if args.mode == "http" else [])
    busy = [p for p in ports if port_in_use(p)]
    if busy:  # an older copy of the app may still be running; it would answer for us and serve old code
        raise SystemExit(f"Port(s) {', '.join(map(str, busy))} already in use - is the app already running? Stop it "
                         f"(Ctrl+C in its terminal, or find the process with: netstat -ano | findstr :{busy[0]}) and retry.")

    try:
        print(f"Starting in '{args.mode}' mode (models are loaded once per service; first run downloads them)...")
        if args.mode == "http":
            d = start("detector", "services.detector.app")
            g = start("grouping", "services.grouping.app")
            wait_for(f"http://127.0.0.1:{svc['detector_port']}/health", d, "detector", args.startup_timeout)
            wait_for(f"http://127.0.0.1:{svc['grouping_port']}/health", g, "grouping", args.startup_timeout)
        gw = start("gateway", "services.gateway.app")
        wait_for(f"http://127.0.0.1:{svc['gateway_port']}/api/v1/health", gw, "gateway", args.startup_timeout)
        print(f"\nReady: open http://localhost:{svc['gateway_port']}   (Ctrl+C to stop)\n")
        while all(p.poll() is None for _, p in procs):
            time.sleep(1)
        dead = [n for n, p in procs if p.poll() is not None]
        print(f"Stopped: {', '.join(dead)} exited.")
    finally:
        stop()


if __name__ == "__main__":
    main()

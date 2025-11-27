import os
import time
import yaml
import socket
import torch
from torch.utils.data import DataLoader
from Dataset.data_loader import CellDataset


def main():
    # ----------------------------------------------------------
    # Locate config file RELATIVE to this script
    # ----------------------------------------------------------
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    CFG_PATH = os.path.join(CURRENT_DIR, "Todo_List", "test.yaml")

    # Load YAML
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # ----------------------------------------------------------
    # Auto-detect machine and set data_root
    # ----------------------------------------------------------
    hostname = socket.gethostname()

    if "anncollin" in hostname:
        data_root = "/home/anncollin/Desktop/Nucleoles/dataset/MyDB/"
    elif "orion" in hostname:
        data_root = "/DATA/annso/MyDB"
    else:
        raise RuntimeError(f"Unknown hostname '{hostname}'. Add mapping.")

    cfg["data_root"] = data_root

    # ----------------------------------------------------------
    # Build dataset (IO only)
    # ----------------------------------------------------------
    dataset = CellDataset(
        root_dir=cfg["data_root"],
        transform=None,              # IMPORTANT: IO-only benchmark
        normalize_16bit=True,
        synthetic_length=5000
    )

    # Warm-up test
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=4,
        shuffle=True
    )

    print("Warm-up:")
    t0 = time.time()
    n_batches = 5
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
    dt = time.time() - t0
    print(f"{n_batches} batches in {dt:.1f}s → {dt/n_batches:.3f} s/batch\n")

    # Benchmark different worker counts
    for nw in [2, 4, 5, 6, 7, 8, 12, 16]:
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=nw,
            shuffle=True,
            pin_memory=True
        )

        t0 = time.time()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

        dt = time.time() - t0
        print(f"workers={nw:2d} → {dt/n_batches:.3f} s/batch")


# ----------------------------------------------------------
# Entry point (REQUIRED for spawn multiprocessing)
# ----------------------------------------------------------
if __name__ == "__main__":
    main()

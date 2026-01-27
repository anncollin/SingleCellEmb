import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# Add the path to: /home/anncollin/Desktop/Nucleoles/dataset/
# so we can import createDB_utils.py
# -------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

#######################################################################################################
# DATASET FOR MyDB_npy  (parallel to CellDataset 
#######################################################################################################
class CellDataset(Dataset):

    def __init__(self, root_dir, transform=None, synthetic_length=100_000, in_chans=2):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.synthetic_length = synthetic_length
        self.in_chans = in_chans

        # find all .npy files
        self.npy_files = []
        for r, d, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    self.npy_files.append(os.path.join(r, f))

        if len(self.npy_files) == 0:
            raise RuntimeError(f"[NpyCellDataset] No .npy files found in {root_dir}.")

        print(f"[NpyCellDataset] Found {len(self.npy_files)} npy files under {root_dir}.")

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, idx):

        while True:  # keep trying until a valid file is found
            path = random.choice(self.npy_files)

            try:
                arr = np.load(path)
                if self.in_chans == 1:
                    arr = arr[0:1]

                tensor = torch.from_numpy(arr).float()
                return tensor

            except Exception:
                # silently drop broken file
                if path in self.npy_files:
                    self.npy_files.remove(path)

                if len(self.npy_files) == 0:
                    raise RuntimeError("[CellDataset] All .npy files are broken.")


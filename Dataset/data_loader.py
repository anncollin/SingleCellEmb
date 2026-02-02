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
# DATASET FOR MyDB_npy  
#######################################################################################################
class CellDataset(Dataset):

    def __init__(self, root_dir, transform=None, synthetic_length=100_000, in_channels="both"):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.synthetic_length = synthetic_length
        self.in_channels = in_channels.lower()

        assert self.in_channels in {"egfp", "dapi", "both"}, \
            f"Invalid in_channels={in_channels}"

        # channel index map (EXPLICIT)
        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        self.npy_files = []
        for r, d, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    self.npy_files.append(os.path.join(r, f))

        if len(self.npy_files) == 0:
            raise RuntimeError(f"[CellDataset] No .npy files found in {root_dir}.")

        print(
            f"[CellDataset] Found {len(self.npy_files)} npy files | "
            f"in_channels={self.in_channels}"
        )

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, idx):

        while True:
            path = random.choice(self.npy_files)

            try:
                arr = np.load(path)  # (2, H, W)

                chans = self.channel_map[self.in_channels]
                arr = arr[chans]     # (C, H, W)

                print(arr.shape)

                tensor = torch.from_numpy(arr).float()

                if self.transform is not None:
                    tensor = self.transform(tensor)

                return tensor

            except Exception:
                if path in self.npy_files:
                    self.npy_files.remove(path)

                if len(self.npy_files) == 0:
                    raise RuntimeError("[CellDataset] All .npy files are broken.")


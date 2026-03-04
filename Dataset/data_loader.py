import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


#######################################################################################################
# DATASET FOR MyDB  (N,2,96,96)
#######################################################################################################
class CellDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        cells_per_well=None,
        wells_csv=None,
        in_channels="both"
    ):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.cells_per_well = cells_per_well
        self.in_channels = in_channels.lower()

        assert self.in_channels in {"egfp", "dapi", "both"}

        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        # ------------------------------------------------------------------
        # collect npy wells
        # ------------------------------------------------------------------

        all_wells = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    all_wells.append(os.path.join(r, f))

        if len(all_wells) == 0:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        # ------------------------------------------------------------------
        # optional CSV filtering
        # ------------------------------------------------------------------

        if wells_csv is not None:

            if wells_csv == "unique_drugs":
                wells_csv = os.path.join(self.root_dir, "..", "labels", "unique_drugs.csv")

            elif wells_csv == "callibration":
                wells_csv = os.path.join(self.root_dir, "..", "labels", "callibration.csv")

            df = pd.read_csv(wells_csv, header=None, dtype=str)
            df.columns = ["plate", "well_code", "drug"]

            allowed = {
                f"{plate}/{well_code}.npy"
                for plate, well_code in zip(df["plate"], df["well_code"])
            }

            wells = []
            for p in all_wells:

                plate = os.path.basename(os.path.dirname(p))
                well  = os.path.splitext(os.path.basename(p))[0]

                key = f"{plate}/{well}.npy"

                if key in allowed:
                    wells.append(p)

            self.npy_files = sorted(wells)

        else:
            self.npy_files = sorted(all_wells)

        if len(self.npy_files) == 0:
            raise RuntimeError("No wells selected")

        print(f"[CellDataset] Using {len(self.npy_files)} wells")

        # ------------------------------------------------------------------
        # build index list
        # each entry = (well_path)
        # ------------------------------------------------------------------

        if cells_per_well is None:
            self.index_list = self.npy_files
        else:
            self.index_list = []
            for p in self.npy_files:
                for _ in range(cells_per_well):
                    self.index_list.append(p)

    def __len__(self):

        return len(self.index_list)

    def __getitem__(self, idx):

        path = self.index_list[idx]

        data = np.load(path, mmap_mode="r")

        if data.ndim != 4:
            raise ValueError(f"Expected (N,2,96,96), got {data.shape}")

        # random cell
        i = np.random.randint(0, data.shape[0])

        arr = data[i]  # (2,96,96)

        arr = arr[self.channel_map[self.in_channels]]

        tensor = torch.from_numpy(np.asarray(arr)).float()

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


#######################################################################################################
# CELL DATASET
# ----------------------------------------------------------------------------------------------------
# Dataset used for self-supervised training (DINO).
#
# This dataset loads wells stored as `.npy` files containing cell populations with shape:
#
#     (N_cells, 2, 96, 96)
#
# where:
#     channel 0 = EGFP
#     channel 1 = DAPI
#
# Each dataset sample corresponds to **one randomly selected cell** from a well.
# The random cell selection is performed at every `__getitem__` call.
#
# The dataset therefore behaves as a large pool of cells sampled across wells.
#
# Optional features:
#     - Restrict wells using a CSV file (e.g. unique_drugs.csv or callibration.csv)
#     - Sample a fixed number of cells per well (`cells_per_well`)
#     - Select input channels (EGFP, DAPI, or both)
#     - Apply augmentations through `transform`
#
# This dataset is intended for **training** where stochastic sampling and
# data augmentation are required.
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

        all_wells = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    all_wells.append(os.path.join(r, f))

        if len(all_wells) == 0:
            raise RuntimeError(f"No .npy files found in {root_dir}")

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

        i = np.random.randint(0, data.shape[0])

        arr = data[i]

        arr = arr[self.channel_map[self.in_channels]]

        tensor = torch.from_numpy(np.asarray(arr)).float()

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor


#######################################################################################################
# POPULATION DATASET
# ----------------------------------------------------------------------------------------------------
# Dataset used for evaluation and embedding computation.
#
# This dataset loads the **entire population of cells from a well** stored as:
#
#     (N_cells, 2, 96, 96)
#
# Unlike `CellDataset`, this dataset does **not randomly sample cells**.
# Instead, it returns all cells from a well at once.
#
# Each sample returned by the dataset is:
#
#     tensor : (N_cells, C, 96, 96)
#     drug   : str
#
# where:
#     C depends on the selected input channels (EGFP, DAPI, or both).
#
# The drug label is read from a CSV file with columns:
#
#     plate, well_code, drug_name
#
# This dataset is typically used during evaluation to:
#
#     - compute embeddings for all cells in a well
#     - aggregate them into population representations
#     - compute distances between drug profiles
#
# No data augmentation or random sampling is performed.
#######################################################################################################

class PopulationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        wells_csv,
        in_channels="both"
    ):
        super().__init__()

        self.root_dir = root_dir
        self.in_channels = in_channels.lower()

        assert self.in_channels in {"egfp", "dapi", "both"}

        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        df = pd.read_csv(wells_csv, header=None, dtype=str)
        df.columns = ["plate", "well_code", "drug"]

        self.samples = []

        for plate, well_code, drug in zip(df["plate"], df["well_code"], df["drug"]):

            npy_path = os.path.join(root_dir, plate, f"{well_code}.npy")

            if os.path.isfile(npy_path):
                self.samples.append((npy_path, drug.strip()))

        if len(self.samples) == 0:
            raise RuntimeError("No wells found for PopulationDataset")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        path, drug = self.samples[idx]
        data       = np.load(path, mmap_mode="r")

        if data.ndim != 4:
            raise ValueError(f"Expected (N,2,96,96), got {data.shape}")

        arr    = data[:, self.channel_map[self.in_channels]]

        ########################
        if "siRNA" in path:

            brightness = 0.5
            arr[:, 1] = np.clip(arr[:, 1] * brightness, 0.0, None)  # DAPI only

        ########################
        tensor = torch.from_numpy(np.asarray(arr)).float()

        return tensor, drug
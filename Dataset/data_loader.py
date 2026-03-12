import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

#######################################################################################################
# CELL DATASET
# ----------------------------------------------------------------------------------------------------
# Dataset used for self-supervised training (DINO).
#######################################################################################################

class CellDataset(Dataset):

    ###################################################################################################
    # INITIALIZATION
    ###################################################################################################
    def __init__(
        self,
        root_dir,
        transform=None,
        cells_per_well=None,
        wells_csv=None,
        in_channels="both",
        cfg=None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.cells_per_well = cells_per_well
        self.in_channels = in_channels.lower()
        self.cfg = cfg or {}

        assert self.in_channels in {"egfp", "dapi", "both"}

        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        # -------------------- batch correction parameters --------------------
        self.zoom_factor     = self.cfg.get("zoom_factor", None)
        self.brightness_dapi = self.cfg.get("brightness_dapi", None)
        self.brightness_egfp = self.cfg.get("brightness_egfp", None)

        # -------------------- discover all wells --------------------
        all_wells = []
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    all_wells.append(os.path.join(r, f))

        if len(all_wells) == 0:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        # -------------------- optional CSV filtering --------------------
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

        # -------------------- sampling strategy --------------------
        if cells_per_well is None:
            self.index_list = self.npy_files
        else:
            self.index_list = []
            for p in self.npy_files:
                for _ in range(cells_per_well):
                    self.index_list.append(p)

    ###################################################################################################
    # DATASET LENGTH
    ###################################################################################################
    def __len__(self):

        return len(self.index_list)

    ###################################################################################################
    # GET ONE CELL
    ###################################################################################################
    def __getitem__(self, idx):

        # -------------------- load well --------------------
        path = self.index_list[idx]
        data = np.load(path, mmap_mode="r")

        if data.ndim != 4:
            raise ValueError(f"Expected (N,2,96,96), got {data.shape}")

        # -------------------- random cell sampling --------------------
        i = np.random.randint(0, data.shape[0])
        arr = data[i]

        # -------------------- channel selection --------------------
        arr = arr[self.channel_map[self.in_channels]]

        # -------------------- batch correction for siRNA plates --------------------
        if "siRNA" in path:

            if self.brightness_dapi is not None and arr.shape[0] > 1:
                arr[1] = arr[1] * self.brightness_dapi

            if self.brightness_egfp is not None:
                arr[0] = arr[0] * self.brightness_egfp

            if self.zoom_factor is not None:

                zoomed = np.stack([
                    cv2.resize(arr[c], (0, 0), fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_LINEAR)
                    for c in range(arr.shape[0])
                ])

                h, w = arr.shape[1:]
                zh, zw = zoomed.shape[1:]

                start_h = (zh - h) // 2
                start_w = (zw - w) // 2

                arr = zoomed[:, start_h:start_h+h, start_w:start_w+w]

        # -------------------- convert to tensor --------------------
        tensor = torch.from_numpy(np.asarray(arr)).float()

        # -------------------- optional augmentation --------------------
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor


#######################################################################################################
# POPULATION DATASET
#######################################################################################################

class PopulationDataset(Dataset):

    ###################################################################################################
    # INITIALIZATION
    ###################################################################################################
    def __init__(
        self,
        root_dir,
        wells_csv=None,
        in_channels="both",
        cfg=None
    ):
        super().__init__()

        self.root_dir = root_dir
        self.in_channels = in_channels.lower()
        self.cfg = cfg or {}

        assert self.in_channels in {"egfp", "dapi", "both"}

        self.channel_map = {
            "egfp": [0],
            "dapi": [1],
            "both": [0, 1],
        }

        # -------------------- batch correction parameters --------------------
        self.zoom_factor = self.cfg.get("zoom_factor", None)
        self.brightness_dapi = self.cfg.get("brightness_dapi", None)
        self.brightness_egfp = self.cfg.get("brightness_egfp", None)

        self.samples = []

        # -------------------- case 1: CSV restriction --------------------
        if wells_csv is not None:

            df = pd.read_csv(wells_csv, header=None, dtype=str)
            df.columns = ["plate", "well_code", "drug"]

            for plate, well_code, drug in zip(df["plate"], df["well_code"], df["drug"]):

                npy_path = os.path.join(root_dir, plate, f"{well_code}.npy")

                if os.path.isfile(npy_path):
                    self.samples.append((npy_path, drug.strip()))

        # -------------------- case 2: scan whole dataset --------------------
        else:

            for plate in os.listdir(root_dir):

                plate_dir = os.path.join(root_dir, plate)

                if not os.path.isdir(plate_dir):
                    continue

                for f in os.listdir(plate_dir):

                    if f.endswith(".npy"):

                        npy_path = os.path.join(plate_dir, f)
                        self.samples.append((npy_path, None))

        if len(self.samples) == 0:
            raise RuntimeError("No wells found for PopulationDataset")

        print("[PopulationDataset] Using full cell populations")

    ###################################################################################################
    # DATASET LENGTH
    ###################################################################################################
    def __len__(self):

        return len(self.samples)

    ###################################################################################################
    # GET ONE WELL
    ###################################################################################################
    def __getitem__(self, idx):

        # -------------------- load population --------------------
        path, drug = self.samples[idx]
        data = np.load(path, mmap_mode="r")

        if data.ndim != 4:
            raise ValueError(f"Expected (N,2,96,96), got {data.shape}")

        # -------------------- channel selection --------------------
        arr = data[:, self.channel_map[self.in_channels]]

        # -------------------- batch correction for siRNA plates --------------------
        if "siRNA" in path:

            if self.brightness_dapi is not None and arr.shape[1] > 1:
                arr[:, 1] = arr[:, 1] * self.brightness_dapi

            if self.brightness_egfp is not None:
                arr[:, 0] = arr[:, 0] * self.brightness_egfp

            if self.zoom_factor is not None:

                zoomed = np.stack([
                    np.stack([
                        cv2.resize(arr[i, c], (0, 0), fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_LINEAR)
                        for c in range(arr.shape[1])
                    ])
                    for i in range(arr.shape[0])
                ])

                h, w = arr.shape[2:]
                zh, zw = zoomed.shape[2:]

                start_h = (zh - h) // 2
                start_w = (zw - w) // 2

                arr = zoomed[:, :, start_h:start_h+h, start_w:start_w+w]

        # -------------------- convert to tensor --------------------
        tensor = torch.from_numpy(np.asarray(arr)).float()

        return tensor, drug
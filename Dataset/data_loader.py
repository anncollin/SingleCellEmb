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
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DATASET_UTILS_DIR = os.path.join(PROJECT_ROOT, "Dataset")   

sys.path.append(DATASET_UTILS_DIR)

from createDB_utils import load_from_zip

#######################################################################################################
# DATASET: LOAD SINGLE-CELL IMAGES FROM MULTIPLE ZIP FILES (MyDB format)
# ----------------------------------------------------------------------------------------------------
#    Parameters:
#    ---------------------------
#    root_dir : str
#        Path to MyDB directory containing Plate001, Plate002, ..., siRNA_Plate003, etc.
#
#    transform : callable or None
#        A transform applied to each sample. For DINO, use MultiCropTransform.
#
#    normalize_16bit : bool
#        If True, scales intensities by dividing by 65535.0 when needed.
#
#    synthetic_length : int
#        The "virtual" length of the dataset. Because each __getitem__ selects a random ZIP file
#        and loads a random cell from it, we simulate an infinite dataset by choosing a large length.
#
#    Returns:
#    ---------------------------
#    sample : Any
#        Usually a list of crops (for DINO), depending on the transform passed.
#
#######################################################################################################
class CellDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        transform=None,
        normalize_16bit: bool = True,
        synthetic_length: int = 100_000,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.normalize_16bit = normalize_16bit
        self.synthetic_length = synthetic_length

        # -------------------------------------------------------------------------
        # FIND ALL ZIP FILES RECURSIVELY
        # -------------------------------------------------------------------------
        self.zip_files = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".zip"):
                    self.zip_files.append(os.path.join(root, f))

        if len(self.zip_files) == 0:
            raise RuntimeError(f"[CellZipDataset] No ZIP files found in {root_dir}.")

        print(f"[CellZipDataset] Found {len(self.zip_files)} zip files under {root_dir}.")

    def __len__(self):
        # We simulate a large dataset composed of random samples drawn on-the-fly
        return self.synthetic_length

    ###################################################################################################
    # LOAD 1 RANDOM CELL FROM A ZIP FILE
    ###################################################################################################
    def _load_random_cell_from_zip(self, zip_path: str) -> torch.Tensor:
        """
        Loads exactly one random 96x96x2 cell from a ZIP file using load_from_zip().
        """
        imgs, _, _ = load_from_zip(
            zip_filename=zip_path,
            batch_size=1,      
            read_all=False,    # random sampling inside the ZIP
            channel="both"
        )

        if imgs.shape[0] == 0:
            raise RuntimeError(f"[CellZipDataset] ZIP {zip_path} contains no images.")

        # imgs shape: (1, 2, 96, 96)
        img = imgs[0].astype(np.float32)    # -> (2, 96, 96)

        # Normalize if required
        if self.normalize_16bit and img.max() > 1.0:
            img = img / 65535.0

        return torch.from_numpy(img)  # (2, 96, 96)

    ###################################################################################################
    # GET ITEM: SELECT RANDOM ZIP, LOAD RANDOM CELL
    ###################################################################################################
    def __getitem__(self, idx):
        zip_path = random.choice(self.zip_files)

        img = self._load_random_cell_from_zip(zip_path)

        if self.transform is not None:
            return self.transform(img)

        return img

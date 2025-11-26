import time
from torch.utils.data import DataLoader
from Dataset.data_loader import CellZipDataset  # or your dataset class
from SSL.transforms import MultiCropTransform   # if needed
import yaml

# Adjust this to your config path
cfg_path = "/home/anncollin/Desktop/Nucleoles/SingleCellEmb/configs/your_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

dataset = CellZipDataset(cfg)  # or however you build it
loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=4, shuffle=True)

t0 = time.time()
n_batches = 50
for i, batch in enumerate(loader):
    if i >= n_batches:
        break
dt = time.time() - t0
print(f"{n_batches} batches in {dt:.1f}s → {dt/n_batches:.3f} s / batch")

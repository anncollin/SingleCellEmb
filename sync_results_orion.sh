#!/bin/bash

rsync -avh \
  -e "ssh" \
  --exclude="*/cached_embeddings.pt" \
  anncollin@orion:SingleCellEmb/Results/ \
  /home/anncollin/Desktop/Nucleoles/SingleCellEmb/Results/


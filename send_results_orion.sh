#!/bin/bash

rsync -avh \
  -e "ssh" \
  --include="AA_*/" \
  --include="AA_*/**" \
  --exclude="*/cached_embeddings.pt" \
  --exclude="*" \
  /home/anncollin/Desktop/Nucleoles/SingleCellEmb/Results/ \
  anncollin@orion:SingleCellEmb/Results/ 


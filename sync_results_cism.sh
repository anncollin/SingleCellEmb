rsync -avh \
  --exclude="*block*" \
  --exclude="*cached_embeddings*" \
  manneback:/home/ucl/elen/acollin/SingleCellEmb/Results/ \
  /home/anncollin/Desktop/Nucleoles/SingleCellEmb/Results/